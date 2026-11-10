import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.block import Block
from transformer.sinusoidal_embeddings import SinusoidalPositionalEmbedding
import math

class GPT(nn.Module):
    """
    Choose 
        d,h,L = 128,4,4 - local, 384,6,6 - gpu
        block, batch = 128, 64, 256,64 - gpu
        d_ff = 512, 1536 - gpu
        params = .82M, 10.7M - gpu
        dropout = .1, .2 - gpu
    """
    def __init__(self, V, d, h, dff, L, block, dropout=0.0):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(V, d)
        
        # Learned positional embeddings
        # self.pos = nn.Embedding(block, d)
        # Sinusoidal positional embeddings
        self.pos = SinusoidalPositionalEmbedding(block, d)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d,h,dff,block,dropout) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.apply(self._init)
        for n, p in self.named_parameters():    # residual writers start small
            if n.endswith("attn.wo.weight") or n.endswith("mlp.w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * L))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):   # idx: (B,T) int64
        B, T = idx.shape
        # learned position
        # x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        # Sinusoidal position
        x = self.drop(self.pos(self.tok(idx)))

        # Attention + MLP
        for b in self.blocks: x = b(x)

        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0):
        for _ in range(n):
            logits, _ = self(idx[:, -self.block:]) # crop to the context window
            p = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
        return idx
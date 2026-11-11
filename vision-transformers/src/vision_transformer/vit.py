import torch
import torch.nn as nn
from vision_transformer.block import Block
from vision_transformer.patch_embed import PatchEmbed

class ViT(nn.Module):
    def __init__(self, img=28, patch=7, in_ch=1, d=64, depth=4, heads=4, n_cls=10):
        super().__init__()
        self.patch = PatchEmbed(img, patch, in_ch, d)
        N = self.patch.n                                   # 16
        self.cls = nn.Parameter(torch.zeros(1, 1, d))      # learnable chair token
        self.pos = nn.Parameter(torch.randn(1, N + 1, d) * 0.02)  # one learned vector per position, incl. [CLS]
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)                        # final LN, as in pre-LN GPT
        self.head = nn.Linear(d, n_cls)                    # d -> class logits

    def forward(self, x):                                  # (B, 1, 28, 28)
        x = self.patch(x)                                  # (B, 16, 64)
        cls = self.cls.expand(x.shape[0], -1, -1)          # copy chair token for each image: (B, 1, 64)
        x = torch.cat([cls, x], dim=1)                     # (B, 17, 64): [CLS] goes first
        x = x + self.pos                                   # add position stickers (broadcast over B)
        x = self.blocks(x)                                 # L rounds of attention + MLP, (B, 17, 64)
        x = self.norm(x)                                   # normalise
        return self.head(x[:, 0])                          # read token 0 only: (B, 10)
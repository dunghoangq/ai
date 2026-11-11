import torch.nn as nn
from vision_transformer.attention import Attention

class Block(nn.Module):
    
    def __init__(self, d, heads, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d)                          # LN before attention
        self.attn = Attention(d, heads)
        self.n2 = nn.LayerNorm(d)                          # LN before MLP
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d),                   # expand 64 -> 256
            nn.GELU(),                                     # smooth non-linearity
            nn.Linear(mlp_ratio * d, d),                   # project back 256 -> 64
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))                      # residual 1: attention
        x = x + self.mlp(self.n2(x))                       # residual 2: per-token MLP
        return x                                           # shape unchanged
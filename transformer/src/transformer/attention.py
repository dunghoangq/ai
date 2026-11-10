import math, torch, torch.nn as nn, torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, d, h, block, dropout=0.0):
        super().__init__()
        assert d % h == 0
        self.h, self.dh = h, d // h
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(block, block, dtype=torch.bool), 1))

    def forward(self, x):
        B, T, d = x.shape
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2) # (B,h,T,dh)
        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        s = s.masked_fill(self.mask[:T, :T], float("-inf"))
        p = self.drop(s.softmax(dim=-1))
        o = (p @ v).transpose(1, 2).reshape(B, T, d)
        return self.wo(o)

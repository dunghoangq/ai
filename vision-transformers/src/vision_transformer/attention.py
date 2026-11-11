import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.dk = d // heads            # per-head width: 64 // 4 = 16
        self.qkv = nn.Linear(d, 3*d)    # one matmul produces Q, K and V together
        self.out = nn.Linear(d, d)      # output projection W_O

    def forward(self, x):
        B, N, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1) # three tensors, each (B, N, d)
        q, k, v = [t.view(B, N, self.h, self.dk).transpose(1, 2)
                   for t in (q, k, v)]         # split heads -> (B, h, N, dk)
        att = (q @ k.transpose(-2, -1)) / self.dk ** 0.5   # scores (B, h, N, N), scaled
        att = att.softmax(dim=-1)                          # each row sums to 1; NO causal mask
        y = att @ v                                        # weighted sum of values (B, h, N, dk)
        y = y.transpose(1, 2).reshape(B, N, d)             # merge heads -> (B, N, d)
        return self.out(y)                                 # mix heads
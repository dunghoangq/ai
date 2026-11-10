import torch
import torch.nn as nn
from transformer.attention import Attention
from transformer.mlp import MLP

class Block(nn.Module):

    def __init__(self, d, h, dff, block, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attention(d, h, block, dropout), MLP(d, dff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

def load_block(blk, p):
    """
    Loads pretrained weights from a NumPy dict/obj into a PyTorch Transformer block.
    """
    t = lambda a: torch.from_numpy(a).double()
    with torch.no_grad():
        for name in ["Wq", "Wk", "Wv", "Wo"]:
            getattr(blk.attn, "w" + name[1].lower()).weight.copy_(t(p[name]).T)
        blk.mlp.w1.weight.copy_(t(p["W1"]).T); blk.mlp.w1.bias.copy_(t(p["b1"]))
        blk.mlp.w2.weight.copy_(t(p["W2"]).T); blk.mlp.w2.bias.copy_(t(p["b2"]))
        blk.ln1.weight.copy_(t(p["ln1_g"])); blk.ln1.bias.copy_(t(p["ln1_b"]))
        blk.ln2.weight.copy_(t(p["ln2_g"])); blk.ln2.bias.copy_(t(p["ln2_b"]))

"""
Test load_block()

rng = np.random.default_rng(0)
X = rng.standard_normal((2, 4, 8))
blk = Block(d=8, h=2, dff=32, block=4).double().eval()
load_block(blk, params)                      # your NumPy dict, same keys as your forward
y_np, _ = transformer_block_forward(X, params, 2, 4)
y_pt = blk(torch.from_numpy(X)).detach().numpy()
print(np.abs(y_np - y_pt).max())
"""
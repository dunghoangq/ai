import math, torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, d, dff):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d, dff), nn.Linear(dff, d)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
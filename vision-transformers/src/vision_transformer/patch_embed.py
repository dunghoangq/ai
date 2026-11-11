import torch.nn as nn

class PatchEmbed(nn.Module):

    def __init__(self, img_size=28, patch=7, in_ch=1, d=64):
        super().__init__()
        self.n = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, d, kernel_size=patch, stride=patch)

    def forward(self, x):                   # x: (B, C, 28, 28)
        x = self.proj(x)                    # (B, 64, 4, 4)
        return x.flatten(2).transpose(1, 2) # flatten(2): (B, 64, 16), transpose: (B, 16, 64)
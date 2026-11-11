from vision_transformer.vit import ViT
import torch
import torch.nn.functional as F

torch.manual_seed(0)

model = ViT()
print(sum(p.numel() for p in model.parameters()))

x = torch.randn(8, 1, 28, 28)
print(model(x).shape)

y = torch.randint(0, 10, (8,))
print(F.cross_entropy(model(x), y).item())
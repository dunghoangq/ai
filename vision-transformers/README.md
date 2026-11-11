# Vision Transformers (ViT): a one-evening primer


## 1. The problem ViT solves

For about a decade, images meant convolutional neural networks (CNNs). A convolution hard-codes two assumptions: **locality** (nearby pixels matter most) and **weight sharing** (a pattern is the same wherever it appears). These are good assumptions, but they were chosen by humans.

Meanwhile, transformers were winning in language by being generic: few built-in assumptions, lots of data, lots of compute. The question ViT (Dosovitskiy et al., 2020) asked was: **can we feed an image to a nearly unmodified transformer and have it learn what it needs?**

The obstacle is cost. Attention compares every token with every other token, so cost grows with the **square** of the sequence length N. If each *pixel* of a 224×224 image is a token:

- N = 224 × 224 = 50,176 tokens
- one attention matrix = 50,176² ≈ **2.5 billion entries**, per head, per layer, per image

That is hopeless. ViT's answer: **don't make pixels the tokens, make patches the tokens.** Cut the image into 16×16 tiles and you get N = 196 tokens, so the attention matrix has 196² ≈ 38,000 entries. That is about 65,000 times smaller.

Remember this number: **patching is a way of controlling N.** It is the idea the Patchout paper builds on.

---

## 2. Stage 1: the analogy

Imagine a committee deciding what a photograph shows.

1. You cut the photo into a grid of **postcards**.
2. A clerk reads each postcard and writes a short **summary note** (a fixed-length list of numbers).
3. Each note gets a **sticker saying where its postcard came from** (top-left, middle, and so on).
4. You add one extra **blank note for the chair**.
5. The committee holds several rounds of discussion. In each round, every note can listen to every other note (**attention**) and then think privately (**MLP**).
6. At the end, you read only the **chair's note** and ask: "Cat or dog?"

## 3. Stage 2: naming the parts

| Analogy | Technical component | Job |
|---|---|---|
| Cutting into postcards | **Patchify** | Reshape the image into N small tiles |
| Clerk writing a summary | **Linear projection** (patch embedding) | Map each flattened tile to a d-dimensional vector |
| Location sticker | **Learned positional embedding** | Tell the model where each token came from |
| Chair's blank note | **[CLS] token** | A learnable vector that gathers information from all patches |
| Discussion rounds | **Transformer encoder blocks** (LayerNorm, attention, MLP, residuals) | Mix information between tokens |
| Reading the chair's note | **Classification head** | Linear layer from the [CLS] output to class logits |

**Where the analogy breaks down:** a real clerk *knows* the postcards form a 2D grid. The model does not. The sticker is just a learned vector, and the transformer must discover from data that sticker 5 and sticker 6 are neighbours. This is the root of ViT's biggest drawback (section 7).

---

## 4. Stage 3: the algorithm on one running example

I'll use a tiny example that runs on your laptop: **MNIST** (28×28 greyscale digits), with patch size 7, d = 64.

| Step | Operation | Shape |
|---|---|---|
| Input | image | `(B, 1, 28, 28)` |
| Patchify | 28/7 = 4 tiles per side, so 16 tiles of 7×7 = **49 numbers** each | `(B, 16, 49)` |
| Linear projection | 49 → 64 | `(B, 16, 64)` |
| Prepend [CLS] | 16 + 1 tokens | `(B, 17, 64)` |
| Add position embedding | learned table | `(B, 17, 64)` |
| L encoder blocks | shape-preserving | `(B, 17, 64)` |
| Final LayerNorm, take token 0 | the [CLS] output | `(B, 64)` |
| Head | 64 → 10 | `(B, 10)` |

For reference, the original ViT-Base on ImageNet is: 224×224×3 image, 16×16 patches, so 196 tiles of 16·16·3 = **768 numbers** each, and d = 768, giving 197 tokens.

Key facts:

- **The encoder is exactly what you built, minus the causal mask.** Every token attends to every token (bidirectional, like BERT, not like GPT).
- **Pre-LN:** `x = x + Attn(LN(x))`, then `x = x + MLP(LN(x))`.
- **Why [CLS]?** The output must be one vector per image, but the encoder outputs 17. [CLS] has no pixels of its own, so it is a neutral "blackboard" that attention writes into. (Averaging all patch outputs also works about as well.)
- **Why is patch embedding a convolution in code?** A `Conv2d` with `kernel_size = stride = patch` applies the *same* linear map to each non-overlapping tile. That is exactly "cut into tiles, flatten, linear layer", just faster. (This is the exercise at the end.)

---

## 5. PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
Standard imports. `F` gives us `cross_entropy` later.

### 5.1 Patch embedding (cut, flatten, project)

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch=7, in_ch=1, d=64):
        super().__init__()
        self.n = (img_size // patch) ** 2   # number of patches: (28//7)^2 = 16
        self.proj = nn.Conv2d(in_ch, d, kernel_size=patch, stride=patch)
        # kernel = stride = 7: a 7x7 window jumps 7 pixels, so tiles never overlap.
        # d output channels = d numbers per tile = the linear projection 49 -> 64.

    def forward(self, x):                     # x: (B, 1, 28, 28)
        x = self.proj(x)                      # (B, 64, 4, 4): a 4x4 grid of 64-d vectors
        return x.flatten(2).transpose(1, 2)   # flatten(2): (B, 64, 16); transpose: (B, 16, 64)
```
`flatten(2)` merges the two spatial dimensions (4×4) into one token axis of length 16, in row-major order. The transpose puts tokens before features, which is the `(B, N, d)` layout your GPT used.

### 5.2 Attention (yours, without the mask)

```python
class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.dk = d // heads                  # per-head width: 64 // 4 = 16
        self.qkv = nn.Linear(d, 3 * d)        # one matmul produces Q, K and V together
        self.out = nn.Linear(d, d)            # output projection W_O

    def forward(self, x):                                  # (B, N, d)
        B, N, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)             # three tensors, each (B, N, d)
        q, k, v = [t.view(B, N, self.h, self.dk).transpose(1, 2)
                   for t in (q, k, v)]                     # split heads -> (B, h, N, dk)
        att = (q @ k.transpose(-2, -1)) / self.dk ** 0.5   # scores (B, h, N, N), scaled
        att = att.softmax(dim=-1)                          # each row sums to 1; NO causal mask
        y = att @ v                                        # weighted sum of values (B, h, N, dk)
        y = y.transpose(1, 2).reshape(B, N, d)             # merge heads -> (B, N, d)
        return self.out(y)                                 # mix heads
```
The only difference from GPT is the missing `masked_fill`. Row i of `att` says "how much token i listens to each other token".

### 5.3 Encoder block

```python
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
```
I've omitted dropout to keep the code minimal. The ViT paper uses it.

### 5.4 The full ViT

```python
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
```
Compare line by line with the table in section 4: every row of that table is one line here. With these settings, the parameter count is about 205k.

### 5.5 Sanity checks first (do these before training)

```python
torch.manual_seed(0)
model = ViT()
print(sum(p.numel() for p in model.parameters()))          # ~205,000
x = torch.randn(8, 1, 28, 28)
print(model(x).shape)                                       # expect torch.Size([8, 10])
y = torch.randint(0, 10, (8,))
print(F.cross_entropy(model(x), y).item())                  # expect close to ln(10) = 2.303
```
**Predict before running:** an untrained model guessing among 10 classes should have loss near ln 10 ≈ 2.30. If you see 5 or 0.1, something is wrong.

### 5.6 Training on CPU

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])  # MNIST mean/std
train = DataLoader(datasets.MNIST("data", train=True,  download=True, transform=tf),
                   batch_size=128, shuffle=True)
test  = DataLoader(datasets.MNIST("data", train=False, download=True, transform=tf),
                   batch_size=512)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

for epoch in range(3):
    model.train()
    for x, y in train:
        loss = F.cross_entropy(model(x), y)   # forward + loss
        opt.zero_grad()                       # clear old gradients
        loss.backward()                       # backprop
        opt.step()                            # update
    model.eval()
    correct = 0
    with torch.no_grad():                     # inference: no graph needed
        for x, y in test:
            correct += (model(x).argmax(dim=1) == y).sum().item()
    print(f"epoch {epoch+1}: test acc {correct / len(test.dataset):.4f}")
```
Why this is laptop-friendly: only 17 tokens, so the 17×17 attention matrices are trivial, and the model is small. I haven't run it here, so treat these as expectations. I'd expect each epoch to take on the order of a minute on a modern CPU, and accuracy in the high 90s after a few epochs. Write your own prediction down first, then compare. If you want to see ViT's data hunger (next section), swap in CIFAR-10 (`in_ch=3, img=32, patch=4`) and compare it with a small CNN. I'd expect the CNN to learn faster at this scale.

---

## 6. How this connects to the Patchout paper

The paper applies exactly this recipe to audio. A **spectrogram** (time × frequency, like a greyscale image) is cut into patches, which become tokens. The problem is that spectrograms of 10-second clips produce *far* more patches than 196, and attention cost scales with N². Patchout's idea is to **randomly drop a subset of patches during training**. This cuts compute and memory, and it acts as a regulariser too. As I recall, they also split positional information into separate time and frequency components, but verify that against the paper. Once you read it with "N is the enemy" in mind, most design choices will make sense.

---

## 7. Drawbacks of ViT

1. **Data hunger.** With no locality or translation prior, ViT must *learn* that neighbouring patches are related. Trained on mid-sized datasets alone, it tends to lose to CNNs. The original paper needed very large pre-training (JFT-300M). Later work (DeiT) closed much of the gap with heavy augmentation and distillation.
2. **Quadratic attention cost** in N. Higher resolution or smaller patches means many more tokens, and cost explodes. Patchout is one remedy.
3. **Patches destroy fine detail.** Everything inside a 16×16 tile is squashed by one linear map, with no hierarchy of fine-to-coarse features.
4. **Fixed resolution.** The position table has a fixed length. A different image size means interpolating the embeddings.
5. **Single scale.** There are no multi-scale feature maps, which is awkward for detection and segmentation (hierarchical variants like Swin address this).
6. **Training sensitivity.** Warmup, weight decay and augmentation matter more than for CNNs.

A common misconception is that ViT "sees" like a CNN, only with attention. It doesn't. In early layers, attention can learn local, CNN-like patterns, but nothing *forces* it to. Its flexibility is both its strength and its cost.

---

## 8. Suggested plan for your evening

1. **20 min:** reread sections 2 to 4 above, and draw the shape table from memory.
2. **40 min:** type the code (don't paste), run the sanity checks, and predict the loss before each run.
3. **30 min:** train 3 epochs while reading the ViT section of the Patchout paper.
4. **Remaining time:** the exercise below, then the rest of the paper.

---

## 9. Exercise (reproduce, not recognise)

**Part A (prediction):** your `ViT` currently uses a learned `self.pos`. Predict what happens to MNIST accuracy if you *delete* `x = x + self.pos`. Does the model lose all spatial information? Write down your answer and your reasoning before testing it. (Hint: think about what attention is permutation-equivariant to, and whether [CLS] changes this.)

**Part B (build):** re-implement `PatchEmbed` **without** `Conv2d`, using `x.unfold(2, 7, 7).unfold(3, 7, 7)`, a reshape to `(B, 16, 49)`, and an `nn.Linear(49, 64)`. Copy the conv's weights into your linear layer (`conv.weight.view(64, 49)`) and assert that both versions give identical outputs with `torch.allclose`. If you get this right, you've proved to yourself that "conv with stride = kernel" is just "cut, flatten, project".

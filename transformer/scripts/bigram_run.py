import torch, math, time, urllib.request, os
from transformer.gpt import GPT

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"): urllib.request.urlretrieve(URL, "input.txt")
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text)); V = len(chars)                  # expect 65
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

d, h, L, block, batch, dropout = 128, 4, 4, 128, 64, 0.1   # config A
lr, min_lr, warmup, iters, eval_every, eval_iters = 1e-3, 1e-4, 100, 5000, 250, 100

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block - 1, (batch,))
    x = torch.stack([src[i:i + block] for i in ix])
    y = torch.stack([src[i + 1:i + block + 1] for i in ix])  # shifted by one
    return x.to(device), y.to(device)

model = GPT(V, d, h, 4 * d, L, block, dropout).to(device)
print(sum(p.numel() for p in model.parameters()))

decay   = [p for p in model.parameters() if p.ndim >= 2]
nodecay = [p for p in model.parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": nodecay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))

def lr_at(it):
    if it < warmup: return lr * (it + 1) / warmup
    t = (it - warmup) / (iters - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * t))

@torch.no_grad()
def estimate_loss():
    model.eval(); out = {}
    for split in ["train", "val"]:
        out[split] = sum(model(*get_batch(split))[1].item() for _ in range(eval_iters)) / eval_iters
    model.train(); return out

history = []
t0 = time.time()
for it in range(iters + 1):
    for g in opt.param_groups: g["lr"] = lr_at(it)
    if it % eval_every == 0:
        e = estimate_loss(); history.append((it, e["train"], e["val"]))
        print(f"{it:5d}  train {e['train']:.4f}  val {e['val']:.4f}  {time.time()-t0:.0f}s")
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
print("".join(itos[i] for i in model.generate(ctx, 500)[0].tolist()))
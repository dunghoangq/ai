from transformer.gpt import GPT

V = 50257
d = 128
h = 4
L = 4
dff = 512
block = 128
dropout = .1

model = GPT(V=V, d=d, h=h, dff=dff, L=L, block=block, dropout=dropout)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")
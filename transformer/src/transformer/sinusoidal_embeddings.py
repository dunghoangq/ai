import torch
import torch.nn as nn
import math

class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()
        # Create a tensor shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term
        # 10000^(2i / d_model) in log space
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (1, max_len, d_model)
        # to broadcast across batch dimensions.
        self.register_buffer('pe', pe.unsqueeze(0).to(torch.get_default_dtype()))

    def forward(self, x):
        # X: (B, T, d_model)
        # Add positional encodings up to the current sequence length T
        return x + self.pe[:, :x.size(1)]
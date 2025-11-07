import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannelAttention(nn.Module):
    def __init__(self, in_dim, num_heads=2):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.num_heads = num_heads

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        out = (x + attn_output) / 2
        return out, attn_weights

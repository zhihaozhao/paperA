import torch
import torch.nn as nn
from typing import List

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(1, channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, channels // reduction), channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x)
        return x * w

class LiteAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        a = torch.softmax(self.w(torch.tanh(x)).squeeze(-1), dim=-1)  # (B, T)
        ctx = torch.bmm(a.unsqueeze(1), x).squeeze(1)  # (B, D)
        return ctx

class MultiScaleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], scales: List[int], se: bool = True, attn: bool = True):
        super().__init__()
        self.scales = scales
        self.proj = nn.Conv1d(input_dim, hidden_dims[0], 1)
        self.se = SqueezeExcite(hidden_dims[0]) if se else nn.Identity()
        self.lstms = nn.ModuleList([nn.LSTM(hidden_dims[0], hidden_dims[1]//2, batch_first=True, bidirectional=True) for _ in scales])
        self.attn = LiteAttention(hidden_dims[1]) if attn else None
        self.head = nn.Linear(hidden_dims[1]* (1 if attn else len(scales)),  num_classes:=6)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.permute(0,2,1)  # (B, F, T)
        x = self.proj(x)
        x = self.se(x)
        x = x.permute(0,2,1)  # (B, T, C)
        outs = []
        for s, lstm in zip(self.scales, self.lstms):
            xs = x[:, ::s, :]
            h,_ = lstm(xs)
            if self.attn:
                outs.append(self.attn(h))
            else:
                outs.append(h.mean(dim=1))
        feat = torch.cat(outs, dim=-1)
        return self.head(feat)

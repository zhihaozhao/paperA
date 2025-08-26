import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None

class LiteAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.w(torch.tanh(x)).squeeze(-1), dim=-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)

class MambaEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2, se: bool = True, attn: bool = True, num_classes: int = 6):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            (Mamba(d_model=hidden_dim) if Mamba is not None else nn.GRU(hidden_dim, hidden_dim, batch_first=True))
            for _ in range(layers)
        ])
        self.se = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//8), nn.ReLU(), nn.Linear(hidden_dim//8, hidden_dim), nn.Sigmoid()) if se else None
        self.attn = LiteAttention(hidden_dim) if attn else None
        self.head = nn.Linear(hidden_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.proj(x)
        for blk in self.blocks:
            if isinstance(blk, nn.GRU):
                h,_ = blk(h)
            else:
                h = blk(h)
        if self.se is not None:
            g = self.se(h.mean(dim=1)).unsqueeze(1)
            h = h * g
        if self.attn is not None:
            h = self.attn(h)
        else:
            h = h.mean(dim=1)
        return self.head(h)

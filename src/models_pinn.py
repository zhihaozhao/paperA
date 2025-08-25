import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PINNLSTMMultiScale(nn.Module):
    """
    Physics-informed LSTM with multi-scale temporal windows.
    - Expects input x: [batch, T, F]
    - Creates pooled sequences at multiple window sizes, passes each through a shared BiLSTM,
      concatenates the last hidden states, and classifies.
    - Designed to be drop-in similar to other models returning logits.
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 window_sizes: List[int] = None,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 bidirectional: bool = True):
        super().__init__()
        if window_sizes is None:
            window_sizes = [32, 64, 128]
        self.window_sizes = list(sorted(set(window_sizes)))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * len(self.window_sizes), max(128, feat_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(max(128, feat_dim), num_classes)
        )

    @staticmethod
    def _pool_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Down/upsample temporal length to target_len via average pooling/interpolation.
        x: [B, T, F]
        """
        b, t, f = x.shape
        if t == target_len:
            return x
        # Use average pooling when downsampling, linear interpolation when upsampling
        if target_len < t:
            # pool along time: reshape to [B, F, T]
            xt = x.transpose(1, 2)  # [B, F, T]
            stride = t // target_len
            kernel = stride
            if kernel < 1:
                kernel = 1
            pooled = F.avg_pool1d(xt, kernel_size=kernel, stride=stride, ceil_mode=True)
            return pooled.transpose(1, 2)  # [B, T', F]
        else:
            xt = x.transpose(1, 2)  # [B, F, T]
            interp = F.interpolate(xt, size=target_len, mode='linear', align_corners=False)
            return interp.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for ws in self.window_sizes:
            xs = self._pool_to_length(x, ws)
            out, _ = self.lstm(xs)  # [B, ws, hidden*dir]
            h_last = out[:, -1, :]  # [B, feat]
            features.append(h_last)
        h = torch.cat(features, dim=-1)
        logits = self.classifier(h)
        return logits


class MambaBlock1D(nn.Module):
    """
    Lightweight Mamba-like temporal block (no external deps):
    - 1x1 projection -> depthwise conv -> gated linear unit -> 1x1 projection
    - LayerNorm and residual
    """
    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.pw_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h = self.ln(x)
        h = h.transpose(1, 2)  # [B, C, T]
        h = self.pw_in(h)
        h_a, h_b = torch.chunk(h, 2, dim=1)
        h = torch.sigmoid(h_a) * h_b  # GLU
        h = self.dw(h)
        h = self.pw_out(h)
        h = h.transpose(1, 2)  # [B, T, C]
        return x + self.drop(h)


class PINNMamba(nn.Module):
    """
    Physics-informed Mamba-like temporal model (scaffold):
    - Input: [B, T, F]
    - Project features -> stack MambaBlock1D -> pooling -> classifier
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 d_model: int = 192,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([MambaBlock1D(d_model=d_model, kernel_size=7, dropout=dropout)
                                     for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        h = self.proj(x)  # [B, T, C]
        for blk in self.blocks:
            h = blk(h)
        # pool over time
        h_t = h.transpose(1, 2)  # [B, C, T]
        logits = self.head(h_t)
        return logits
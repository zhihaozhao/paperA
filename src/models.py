# Complete src/models.py
# This is a standalone, complete version assuming a typical CSI classification setup.
# It includes definitions for 'enhanced' (as TinyNet example), 'bilstm', and 'cnn'.
# Replace your existing src/models.py with this, or merge it (keep your custom parts).
# Assumes input shape: [batch, T, F] for LSTM (1D seq), [batch, 1, T, F] for CNN (2D).
# Adjust hyperparameters (e.g., hidden_dims) based on your paperA needs.
# Requires: import torch and torch.nn as nn (already included).

import torch
import torch.nn as nn
from .models_pinn import PINNLSTMMultiScale, PINNMamba  # NEW: PINN models


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention along time axis.
    Input expects [B, C, T, F]. We pool over F to 1, then attend over T with embed=C.
    """

    def __init__(self, channels: int, num_heads: int = 4, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = channels
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=attn_dropout, batch_first=False)
        self.ln = nn.LayerNorm(channels)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        # Pool over F
        x = torch.mean(x, dim=3, keepdim=False)  # [B, C, T]
        x = x.permute(2, 0, 1)  # [T, B, C]
        x = self.ln(x)
        out, _ = self.attn(x, x, x, need_weights=False)
        out = self.proj_drop(out)
        out = out.permute(1, 2, 0)  # [B, C, T]
        return out


class EnhancedNet(nn.Module):
    """
    CNN + SE + light temporal self-attention.
    Input: x [B, T, F]
    """

    def __init__(self, T: int, F: int, num_classes: int, base_channels: int = 160, attn_heads: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv2d(base_channels, base_channels, k=3, s=1, p=1),
            SqueezeExcite(base_channels, reduction=12),
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv2d(base_channels, base_channels * 2, k=3, s=(2, 1), p=1),
            SqueezeExcite(base_channels * 2, reduction=12),
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv2d(base_channels * 2, base_channels * 2, k=3, s=1, p=1),
            SqueezeExcite(base_channels * 2, reduction=12),
        )
        self.attn = TemporalSelfAttention(channels=base_channels * 2, num_heads=attn_heads, attn_dropout=0.0, proj_dropout=0.0)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # over T after attention
            nn.Flatten(),
            nn.Linear(base_channels * 2, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, F]
        b, t, f = x.shape
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Attention over time
        x_att = self.attn(x)  # [B, C, T']
        logits = self.head(x_att)
        return logits

class TinyNet(nn.Module):  # Example for 'enhanced' - replace with your actual enhanced model if different
    def __init__(self, input_features, num_classes):
        super(TinyNet, self).__init__()
        self.fc1 = nn.Linear(input_features * 128, 256)  # Flatten T=128, F=input_features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten [batch, T, F] -> [batch, T*F]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=8):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Bidirectional: *2

    def forward(self, x):
        # x: [batch, T, F] (seq_len=T, features=F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last timestep
        return self.fc(out)

class SimpleCNN(nn.Module):
    def __init__(self, T, F, num_classes=8, input_channels=1, c1: int = 16, c2: int = 32, fc_hidden: int = 48):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Calculate flattened size: after two pools, height=T//4, width=F//4
        self.fc1 = nn.Linear(c2 * (T // 4) * (F // 4), fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        # x: [batch, T, F] -> Add channel: [batch, 1, T, F]
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def build_model(name, F, num_classes, T=128):  # T added for CNN
    if name == "enhanced":
        # CNN + SE + light attention (recommended enhanced)
        return EnhancedNet(T=T, F=F, num_classes=num_classes)

    elif name == "bilstm":
        return BiLSTM(input_dim=F, num_classes=num_classes)

    elif name == "cnn":
        return SimpleCNN(T=T, F=F, num_classes=num_classes)

    elif name == "conformer_lite":
        return ConformerLite(input_dim=F, d_model=192, num_layers=2, num_heads=4, num_classes=num_classes)

    # NEW: PINN models
    elif name == "pinn_lstm_ms":
        return PINNLSTMMultiScale(input_dim=F, num_classes=num_classes)
    elif name == "pinn_mamba":
        return PINNMamba(input_dim=F, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model {name}")

# Optional: Function to count parameters (add to your train_eval.py if needed)
# print(f"Model: {name} with {sum(p.numel() for p in model.parameters())} params")


# -------- Conformer-lite (minimal) --------

class ConvModule1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU(inplace=True)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.pw1(x)
        x = self.glu(x)
        x = self.dw(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pw2(x)
        x = self.drop(x)
        x = x.transpose(1, 2)  # [B, T, C]
        return x


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.ln(x), self.ln(x), self.ln(x), need_weights=False)
        return self.drop(y)


class ConformerBlockLite(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, conv_kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, expansion=4, dropout=dropout)
        self.mhsa = MHSA(d_model, num_heads=num_heads, dropout=dropout)
        self.conv = ConvModule1D(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, expansion=4, dropout=dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)


class ConformerLite(nn.Module):
    """
    Minimal Conformer-style encoder for time sequence on CSI.
    Pipeline: project F->d_model per timestep -> N blocks -> GAP -> Linear.
    Input: [B, T, F]
    """

    def __init__(self, input_dim: int, d_model: int, num_layers: int, num_heads: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
        )
        self.blocks = nn.ModuleList([
            ConformerBlockLite(d_model=d_model, num_heads=num_heads, conv_kernel=7, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = self.in_proj(x)  # [B, T, d_model]
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)  # [B, d_model, T]
        x = self.pool(x).squeeze(-1)  # [B, d_model]
        return self.head(x)

def get_model(name, input_dim, num_classes=4, logit_l2=0.05, T=128):
    """
    Alias for build_model for backward compatibility
    Maps input_dim to F parameter for consistency
    """
    return build_model(name=name, F=input_dim, num_classes=num_classes, T=T)


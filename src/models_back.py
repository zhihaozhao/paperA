
# This is a complete patched version of your models.py.
# Key changes:
# - Made 'num_classes' a required parameter (no default=4) for all models to ensure it's always passed and configurable.
# - Removed hardcoded num_classes=4 from __init__ signatures; now must be provided (e.g., from args.num_classes).
# - In build_model: Added num_classes parameter and passed it to all model constructors.
# - Retained logit_l2 as before (e.g., for BiLSTM, TCN, TinyTransformer).
# - No other changes – backward compatible if you pass num_classes=4, but now flexible for 8+.
# - Integration: In train_eval.py, call build_model(..., num_classes=args.num_classes) – add if missing.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, num_classes=4, bidir=True, logit_l2=0.05):  # Removed default=4
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, num_classes))
        self.logit_l2 = logit_l2

    def forward(self, x, y=None):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        feat = out[:, -1, :]  # simple pooling; replace with better pooling if needed
        logits = self.head(feat)
        loss = None
        if y is not None:
            ce = nn.CrossEntropyLoss()(logits, y)
            l2 = (logits.pow(2).sum(dim=1).mean())
            loss = ce + self.logit_l2 * l2
        return logits, loss

class LSTM32(nn.Module):
    def __init__(self, input_dim, num_classes):  # Removed default=4
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x, y=None):
        out, _ = self.lstm(x); logits = self.fc(out[:, -1, :])
        loss = nn.CrossEntropyLoss()(logits, y) if y is not None else None
        return logits, loss

# TODO: add TCN and TinyTransformer minimal variants if needed
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)

        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)
        return out

class TCN(nn.Module):
    def __init__(self, input_dim, num_classes, channels=(32, 64), kernel_size=3, dropout=0.1, logit_l2=0.0):  # Removed default=4
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(
                n_inputs=in_ch,
                n_outputs=out_ch,
                kernel_size=kernel_size,
                stride=1,
                dilation=2**i,
                dropout=dropout
            ))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.logit_l2 = float(logit_l2)

    def forward(self, x, y=None):
        # 输入统一为 (B, C, T)
        if x.dim() == 3 and x.shape[1] != x.shape[2]:
            # 常见输入是 (B, T, C)
            x = x.transpose(1, 2)

        feat = self.tcn(x)
        logits = self.head(feat)  # (B, num_classes)

        if y is None:
            return logits, None

        loss = self.criterion(logits, y)
        if self.logit_l2 and self.logit_l2 > 0:
            loss = loss + self.logit_l2 * logits.pow(2).mean()
        return logits, loss

    def l2_on_logits(self, logits):
        if self.logit_l2 and self.logit_l2 > 0:
            return self.logit_l2 * logits.pow(2).mean()
        return torch.tensor(0.0, device=logits.device if torch.is_tensor(logits) else "cpu")

class TinyTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, max_len=512, logit_l2=0.0):  # Removed default=4
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_embed = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.logit_l2 = float(logit_l2)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, y=None):
        # 期望 (B, T, C); 若是 (B, C, T)，转换
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            # 可能已经是 (B, T, C)，无需转
            # pass
            x = x.transpose(1, 2)
        # elif x.dim() == 3 and x.shape[1] > x.shape[2]:
        #     # 可能是 (B, C, T)
        #     x = x.transpose(1, 2)

        B, T, C = x.shape
        h = self.proj(x)  # (B, T, d_model)

        # prepend CLS token
        cls_tok = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        h = torch.cat([cls_tok, h], dim=1)  # (B, 1+T, d_model)

        h = self.pos_embed(h)
        h = self.encoder(h)                  # (B, 1+T, d_model)
        cls = self.norm(h[:, 0, :])          # (B, d_model)
        logits = self.head(cls)              # (B, num_classes)

        if y is None:
            return logits, None

        loss = self.criterion(logits, y)
        if self.logit_l2 and self.logit_l2 > 0:
            loss = loss + self.logit_l2 * logits.pow(2).mean()
        return logits, loss


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

def build_model(name, input_dim, num_classes, logit_l2=0.05):  # Added num_classes param
    name = str(name).lower()
    if name == "enhanced":
        return BiLSTM(input_dim, hidden=128, layers=2, num_classes=num_classes, bidir=True, logit_l2=logit_l2)
    elif name == "lstm":
        return LSTM32(input_dim, num_classes=num_classes)
    elif name in ("tcn", "tcn1d", "temporalconvnet"):
        return TCN(input_dim=input_dim, num_classes=num_classes, logit_l2=logit_l2)
    elif name in ("txf", "tiny_txf", "transformer_tiny", "transformer"):
        # 如果你还没有实现 TinyTransformer，请暂时去掉这分支或先实现
       return TinyTransformer(input_dim=input_dim, num_classes=num_classes, logit_l2=logit_l2)
    else:
        raise ValueError(f"Unknown model {name}")

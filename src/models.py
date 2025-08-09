import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, num_classes=4, bidir=True, logit_l2=0.05):
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
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x, y=None):
        out, _ = self.lstm(x); logits = self.fc(out[:, -1, :])
        loss = nn.CrossEntropyLoss()(logits, y) if y is not None else None
        return logits, loss

# TODO: add TCN and TinyTransformer minimal variants if needed

def build_model(name, input_dim, num_classes, logit_l2=0.05):
    name = name.lower()
    if name == "enhanced":
        return BiLSTM(input_dim, hidden=128, layers=2, num_classes=num_classes, bidir=True, logit_l2=logit_l2)
    if name == "lstm":
        return LSTM32(input_dim, num_classes)
    raise ValueError(f"Unknown model {name}")
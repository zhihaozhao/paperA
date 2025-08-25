import torch
import torch.nn as nn


def temporal_smoothness_penalty(x: torch.Tensor) -> torch.Tensor:
    """
    Penalize second temporal derivative to encourage smoothness in representations/logits.
    x: [B, T, C] or [B, T, F]
    returns scalar tensor
    """
    # Second finite difference along T
    d1 = x[:, 1:, :] - x[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return (d2.pow(2).mean())


def energy_penalty(x: torch.Tensor) -> torch.Tensor:
    """L2 energy regularization on sequence features/logits"""
    return (x.pow(2).mean())


class PINNWrapperLoss(nn.Module):
    """
    Combine CE with physics-informed penalties.
    - logits: [B, C]
    - feats: optional sequence features [B, T, C] (if model can return them)
    """
    def __init__(self, lambda_smooth: float = 0.0, lambda_energy: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_energy = float(lambda_energy)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, seq_feats: torch.Tensor = None) -> torch.Tensor:
        loss = self.ce(logits, targets)
        if seq_feats is not None:
            if self.lambda_smooth > 0:
                loss = loss + self.lambda_smooth * temporal_smoothness_penalty(seq_feats)
            if self.lambda_energy > 0:
                loss = loss + self.lambda_energy * energy_penalty(seq_feats)
        return loss
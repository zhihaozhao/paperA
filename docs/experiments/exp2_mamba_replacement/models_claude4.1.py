"""
Mamba State-Space Model for CSI-based HAR
Complete implementation for Exp2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math
from einops import rearrange, repeat


class SelectiveSSM(nn.Module):
    """Selective State-Space Model (Mamba) block"""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=bias
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_std)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # State matrices
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, length, d_model]
        Returns:
            output: Output tensor [batch, length, d_model]
        """
        batch, length, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :length]
        x = rearrange(x, 'b d l -> b l d')
        
        # SSM
        x = F.silu(x)
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan algorithm"""
        batch, length, d_inner = x.shape
        
        # Compute SSM parameters
        deltaBC = self.x_proj(x)  # [batch, length, dt_rank + 2*d_state]
        
        # Split into components
        delta = deltaBC[..., :self.dt_rank]
        B = deltaBC[..., self.dt_rank:self.dt_rank + self.d_state]
        C = deltaBC[..., self.dt_rank + self.d_state:]
        
        # Compute dt
        delta = F.softplus(self.dt_proj(delta))  # [batch, length, d_inner]
        
        # Get A
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        
        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [batch, length, d_inner, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [batch, length, d_inner, d_state]
        
        # Selective scan (simplified for clarity - actual implementation would use parallel scan)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        ys = []
        
        for t in range(length):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t:t+1].transpose(1, 2)
            y = (h @ C[:, t:t+1, :].transpose(1, 2)).transpose(1, 2)
            ys.append(y)
        
        y = torch.cat(ys, dim=1)
        
        # Add skip connection
        y = y + x * self.D
        
        return y


class MambaBlock(nn.Module):
    """Complete Mamba block with normalization and residual"""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state, expand=expand)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class CSIEmbedding(nn.Module):
    """Embed CSI data for Mamba processing"""
    
    def __init__(self, num_subcarriers: int, num_antennas: int, d_model: int):
        super().__init__()
        input_dim = num_subcarriers * num_antennas
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CSI input [batch, time, subcarriers, antennas]
        Returns:
            embedded: [batch, time, d_model]
        """
        batch, time, subcarriers, antennas = x.shape
        x = x.reshape(batch, time, -1)
        return self.projection(x)


class MultiResolutionMamba(nn.Module):
    """Multi-resolution Mamba processing"""
    
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        
        # Three resolution branches
        self.mamba_fine = MambaBlock(d_model, d_state)
        self.mamba_med = MambaBlock(d_model, d_state)
        self.mamba_coarse = MambaBlock(d_model, d_state)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch, time, d_model]
        Returns:
            fused: Multi-resolution features [batch, time, d_model]
        """
        batch, time, d_model = x.shape
        
        # Fine resolution
        y_fine = self.mamba_fine(x)
        
        # Medium resolution (downsample by 4)
        if time >= 4:
            x_med = F.avg_pool1d(x.transpose(1, 2), kernel_size=4, stride=4).transpose(1, 2)
            y_med = self.mamba_med(x_med)
            y_med = F.interpolate(y_med.transpose(1, 2), size=time, mode='linear').transpose(1, 2)
        else:
            y_med = y_fine
        
        # Coarse resolution (downsample by 16)
        if time >= 16:
            x_coarse = F.avg_pool1d(x.transpose(1, 2), kernel_size=16, stride=16).transpose(1, 2)
            y_coarse = self.mamba_coarse(x_coarse)
            y_coarse = F.interpolate(y_coarse.transpose(1, 2), size=time, mode='linear').transpose(1, 2)
        else:
            y_coarse = y_fine
        
        # Concatenate and fuse
        concat = torch.cat([y_fine, y_med, y_coarse], dim=-1)
        fused = self.fusion(concat)
        
        return fused


class MambaCSI(nn.Module):
    """Complete Mamba model for CSI-based HAR"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.num_subcarriers = config.get('num_subcarriers', 30)
        self.num_antennas = config.get('num_antennas', 3)
        self.num_classes = config.get('num_classes', 6)
        self.d_model = config.get('d_model', 256)
        self.d_state = config.get('d_state', 16)
        self.n_layers = config.get('n_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # CSI embedding
        self.embedding = CSIEmbedding(
            self.num_subcarriers,
            self.num_antennas,
            self.d_model
        )
        
        # Multi-resolution Mamba
        self.multi_res_mamba = MultiResolutionMamba(self.d_model, self.d_state)
        
        # Stacked Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(self.d_model, self.d_state)
            for _ in range(self.n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass
        Args:
            x: CSI input [batch, time, subcarriers, antennas]
            labels: Optional labels for training
        Returns:
            Dictionary containing predictions and losses
        """
        # Embed CSI
        x = self.embedding(x)
        
        # Multi-resolution processing
        x = self.multi_res_mamba(x)
        
        # Stack of Mamba blocks
        for block in self.mamba_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        # Prepare output
        output = {
            'logits': logits,
            'predictions': torch.argmax(logits, dim=1),
            'features': x
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output['loss'] = loss
        
        return output


def create_mamba_model(config: Dict) -> MambaCSI:
    """Factory function to create Mamba model"""
    return MambaCSI(config)


# Simplified Mamba for testing (without custom CUDA kernels)
class SimplifiedMamba(nn.Module):
    """Simplified Mamba using standard PyTorch operations"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.num_subcarriers = config.get('num_subcarriers', 30)
        self.num_antennas = config.get('num_antennas', 3)
        self.num_classes = config.get('num_classes', 6)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Simple architecture using GRU as SSM replacement
        input_dim = self.num_subcarriers * self.num_antennas
        
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # Use GRU as simplified state-space model
        self.gru_layers = nn.ModuleList([
            nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True)
            for _ in range(3)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        batch, time, subcarriers, antennas = x.shape
        
        # Flatten and project
        x = x.reshape(batch, time, -1)
        x = self.input_proj(x)
        
        # GRU layers (simplified SSM)
        for gru in self.gru_layers:
            x_gru, _ = gru(x)
            x = self.output_proj(x_gru) + x  # Residual connection
        
        # Pool and classify
        x = x.mean(dim=1)
        logits = self.classifier(x)
        
        output = {
            'logits': logits,
            'predictions': torch.argmax(logits, dim=1)
        }
        
        if labels is not None:
            output['loss'] = F.cross_entropy(logits, labels)
        
        return output


if __name__ == "__main__":
    # Test model creation
    config = {
        'num_subcarriers': 30,
        'num_antennas': 3,
        'num_classes': 6,
        'd_model': 256,
        'd_state': 16,
        'n_layers': 4,
        'dropout': 0.1
    }
    
    # Use simplified version for testing
    model = SimplifiedMamba(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    time_steps = 100
    x = torch.randn(batch_size, time_steps, 30, 3)
    labels = torch.randint(0, 6, (batch_size,))
    
    output = model(x, labels)
    print(f"Output keys: {output.keys()}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
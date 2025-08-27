"""
Physics-Informed Multi-Scale LSTM with Lightweight Attention
Complete implementation for Exp1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math

class MultiScaleLSTM(nn.Module):
    """Multi-scale LSTM processing at different temporal resolutions"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        
        # Create LSTM for each scale
        self.lstm_fine = nn.LSTM(input_dim, hidden_dim, 2, 
                                  batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_med = nn.LSTM(input_dim, hidden_dim, 2,
                                 batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_coarse = nn.LSTM(input_dim, hidden_dim, 2,
                                    batch_first=True, bidirectional=True, dropout=0.2)
        
        # Adaptive fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Input tensor [batch, time, features]
        Returns:
            fused_features: Multi-scale features [batch, time, hidden_dim]
            scale_outputs: Dict containing individual scale outputs
        """
        batch, time, features = x.shape
        
        # Fine scale (full resolution)
        h_fine, _ = self.lstm_fine(x)
        
        # Medium scale (downsample by 5)
        if time >= 5:
            x_med = F.avg_pool1d(x.transpose(1, 2), kernel_size=5, stride=5).transpose(1, 2)
            h_med, _ = self.lstm_med(x_med)
            # Upsample back to original resolution
            h_med = F.interpolate(h_med.transpose(1, 2), size=time, mode='linear').transpose(1, 2)
        else:
            h_med = h_fine
        
        # Coarse scale (downsample by 10)
        if time >= 10:
            x_coarse = F.avg_pool1d(x.transpose(1, 2), kernel_size=10, stride=10).transpose(1, 2)
            h_coarse, _ = self.lstm_coarse(x_coarse)
            # Upsample back to original resolution
            h_coarse = F.interpolate(h_coarse.transpose(1, 2), size=time, mode='linear').transpose(1, 2)
        else:
            h_coarse = h_fine
        
        # Adaptive fusion with normalized weights
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * h_fine + weights[1] * h_med + weights[2] * h_coarse
        
        # Project to output dimension
        output = self.output_proj(fused)
        
        scale_outputs = {
            'fine': h_fine,
            'medium': h_med,
            'coarse': h_coarse,
            'weights': weights
        }
        
        return output, scale_outputs


class LightweightAttention(nn.Module):
    """Linear-complexity attention mechanism for CSI sequences"""
    
    def __init__(self, dim: int, num_heads: int = 8, kernel_fn: str = 'elu'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Kernel function for linear attention
        if kernel_fn == 'elu':
            self.kernel = lambda x: F.elu(x) + 1
        else:
            self.kernel = lambda x: F.relu(x)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, time, dim]
            mask: Optional attention mask
        Returns:
            attended: Output tensor [batch, time, dim]
        """
        batch, time, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).reshape(batch, time, self.num_heads, self.head_dim)
        K = self.k_proj(x).reshape(batch, time, self.num_heads, self.head_dim)
        V = self.v_proj(x).reshape(batch, time, self.num_heads, self.head_dim)
        
        # Apply kernel function for linear attention
        Q = self.kernel(Q)
        K = self.kernel(K)
        
        # Compute KV (this is the key optimization - compute KV first)
        # [batch, num_heads, head_dim, head_dim]
        KV = torch.einsum('bthd,bthe->bhde', K, V)
        
        # Compute normalizer
        Z = 1 / (torch.einsum('bthd,bhd->bth', Q, K.sum(dim=1)) + 1e-6)
        
        # Compute attention output
        # [batch, time, num_heads, head_dim]
        out = torch.einsum('bthd,bhde->bthe', Q, KV) * Z.unsqueeze(-1)
        
        # Reshape and project
        out = out.reshape(batch, time, self.dim)
        out = self.out_proj(out)
        
        return out


class PhysicsLoss(nn.Module):
    """Physics-informed loss functions for WiFi propagation"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.lambda_fresnel = config.get('lambda_fresnel', 0.1)
        self.lambda_multipath = config.get('lambda_multipath', 0.05)
        self.lambda_doppler = config.get('lambda_doppler', 0.05)
        
        # WiFi parameters
        self.frequency = 2.4e9  # 2.4 GHz
        self.wavelength = 3e8 / self.frequency  # ~0.125m
        
    def fresnel_loss(self, csi: torch.Tensor, distance: float = 5.0) -> torch.Tensor:
        """
        Compute Fresnel zone consistency loss
        Args:
            csi: CSI amplitude [batch, time, subcarriers]
            distance: Transmitter-receiver distance (meters)
        """
        # First Fresnel zone radius at midpoint
        r_fresnel = np.sqrt(self.wavelength * distance / 4)
        
        # Expected attenuation pattern based on Fresnel zones
        # Simplified model: attenuation increases with frequency
        batch, time, subcarriers = csi.shape
        freq_factor = torch.linspace(0.8, 1.2, subcarriers, device=csi.device)
        expected_pattern = torch.exp(-freq_factor / r_fresnel)
        
        # Compute mean CSI pattern
        csi_pattern = csi.mean(dim=1)  # Average over time
        
        # Normalize patterns
        csi_pattern = F.normalize(csi_pattern, p=2, dim=1)
        expected_pattern = F.normalize(expected_pattern.unsqueeze(0), p=2, dim=1)
        
        # MSE loss between patterns
        loss = F.mse_loss(csi_pattern, expected_pattern.expand_as(csi_pattern))
        
        return loss
    
    def multipath_loss(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Multipath propagation consistency loss
        Args:
            csi: Complex CSI [batch, time, subcarriers]
        """
        # Compute frequency correlation (multipath creates correlation)
        batch, time, subcarriers = csi.shape
        
        # Frequency domain correlation
        csi_freq = torch.fft.fft(csi, dim=2)
        
        # Multipath should create smooth frequency response
        freq_diff = torch.diff(csi_freq.abs(), dim=2)
        smoothness_loss = freq_diff.pow(2).mean()
        
        # Sparsity in delay domain (few multipath components)
        csi_delay = torch.fft.ifft(csi_freq, dim=2)
        sparsity_loss = -torch.abs(csi_delay).sum(dim=2).mean()  # L1 sparsity
        
        loss = smoothness_loss + 0.01 * sparsity_loss
        
        return loss
    
    def doppler_loss(self, csi: torch.Tensor, activity_label: torch.Tensor) -> torch.Tensor:
        """
        Activity-specific Doppler constraint loss
        Args:
            csi: CSI amplitude [batch, time, subcarriers]
            activity_label: Activity labels [batch]
        """
        # Compute Doppler spectrum via time-frequency analysis
        batch, time, subcarriers = csi.shape
        
        # Short-time Fourier transform along time dimension
        window_size = min(64, time // 4)
        if window_size < 16:
            return torch.tensor(0.0, device=csi.device)
        
        # Simple Doppler: compute temporal frequency content
        csi_time = torch.fft.fft(csi, dim=1)
        doppler_spectrum = csi_time.abs().mean(dim=2)  # [batch, time_freq]
        
        # Expected Doppler patterns for different activities
        # 0: static, 1: walking, 2: running, etc.
        num_classes = activity_label.max().item() + 1
        
        # Create expected Doppler masks
        freq_bins = doppler_spectrum.shape[1]
        doppler_masks = torch.zeros(num_classes, freq_bins, device=csi.device)
        
        # Static activities: energy concentrated at low frequencies
        doppler_masks[0, :freq_bins//8] = 1.0
        
        # Walking: medium frequency components
        if num_classes > 1:
            doppler_masks[1, freq_bins//8:freq_bins//4] = 1.0
        
        # Running: higher frequency components
        if num_classes > 2:
            doppler_masks[2, freq_bins//4:freq_bins//2] = 1.0
        
        # Select masks for batch
        batch_masks = doppler_masks[activity_label]
        
        # Normalize spectra
        doppler_spectrum = F.normalize(doppler_spectrum, p=1, dim=1)
        batch_masks = F.normalize(batch_masks, p=1, dim=1)
        
        # KL divergence between actual and expected Doppler
        loss = F.kl_div(
            (doppler_spectrum + 1e-8).log(),
            batch_masks,
            reduction='batchmean'
        )
        
        return loss
    
    def forward(self, csi: torch.Tensor, predictions: torch.Tensor, 
                labels: torch.Tensor, features: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute combined physics loss
        Args:
            csi: Input CSI data
            predictions: Model predictions
            labels: Ground truth labels
            features: Additional features from model
        """
        losses = {}
        
        # Classification loss
        losses['ce'] = F.cross_entropy(predictions, labels)
        
        # Physics losses
        if self.lambda_fresnel > 0:
            losses['fresnel'] = self.lambda_fresnel * self.fresnel_loss(csi)
        
        if self.lambda_multipath > 0:
            losses['multipath'] = self.lambda_multipath * self.multipath_loss(csi)
        
        if self.lambda_doppler > 0:
            losses['doppler'] = self.lambda_doppler * self.doppler_loss(csi, labels)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class PhysicsInformedCSIModel(nn.Module):
    """Complete Physics-Informed Multi-Scale LSTM model for CSI-based HAR"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model configuration
        self.num_subcarriers = config.get('num_subcarriers', 30)
        self.num_antennas = config.get('num_antennas', 3)
        self.num_classes = config.get('num_classes', 6)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # Input projection
        input_dim = self.num_subcarriers * self.num_antennas
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.hidden_dim)
        )
        
        # Multi-scale LSTM
        self.multi_scale_lstm = MultiScaleLSTM(self.hidden_dim, self.hidden_dim)
        
        # Lightweight attention
        self.attention = LightweightAttention(self.hidden_dim, num_heads=8)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )
        
        # Physics loss
        self.physics_loss = PhysicsLoss(config.get('physics_config', {}))
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from raw CSI"""
        batch, time, subcarriers, antennas = x.shape
        
        # Flatten subcarriers and antennas
        x = x.reshape(batch, time, -1)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        return x
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass
        Args:
            x: CSI input [batch, time, subcarriers, antennas]
            labels: Optional labels for training
        Returns:
            Dictionary containing predictions and losses
        """
        # Store original input for physics loss
        csi_input = x.clone()
        
        # Extract features
        features = self.extract_features(x)
        
        # Multi-scale LSTM processing
        lstm_out, scale_outputs = self.multi_scale_lstm(features)
        
        # Apply attention
        attended = self.attention(lstm_out)
        
        # Global average pooling
        pooled = attended.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Prepare output
        output = {
            'logits': logits,
            'predictions': torch.argmax(logits, dim=1),
            'features': attended,
            'scale_outputs': scale_outputs
        }
        
        # Compute losses if labels provided
        if labels is not None:
            # Reshape CSI for physics loss
            batch, time, subcarriers, antennas = csi_input.shape
            csi_reshaped = csi_input.mean(dim=3)  # Average over antennas
            
            losses = self.physics_loss(csi_reshaped, logits, labels, scale_outputs)
            output['losses'] = losses
        
        return output


def create_model(config: Dict) -> PhysicsInformedCSIModel:
    """Factory function to create model"""
    return PhysicsInformedCSIModel(config)


if __name__ == "__main__":
    # Test model creation
    config = {
        'num_subcarriers': 30,
        'num_antennas': 3,
        'num_classes': 6,
        'hidden_dim': 128,
        'physics_config': {
            'lambda_fresnel': 0.1,
            'lambda_multipath': 0.05,
            'lambda_doppler': 0.05
        }
    }
    
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    time_steps = 100
    x = torch.randn(batch_size, time_steps, 30, 3)
    labels = torch.randint(0, 6, (batch_size,))
    
    output = model(x, labels)
    print(f"Output keys: {output.keys()}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Total loss: {output['losses']['total'].item():.4f}")
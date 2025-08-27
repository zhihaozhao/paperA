"""
Exp3: Physics-Informed LSTM with Causal Attention
物理引导的多尺度LSTM + 因果注意力机制
Author: Claude 4.1
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import math

class PhysicsFeatureExtractor(nn.Module):
    """
    物理特征提取器
    从原始CSI数据中提取物理意义明确的特征
    """
    def __init__(self, 
                 freq_dim: int = 114,
                 time_dim: int = 500,
                 carrier_freq: float = 5.8e9):
        super().__init__()
        self.freq_dim = freq_dim
        self.time_dim = time_dim
        self.carrier_freq = carrier_freq
        self.c = 3e8
        self.wavelength = self.c / carrier_freq
        
        # Learnable physics feature projections
        self.fresnel_proj = nn.Linear(freq_dim, 64)
        self.multipath_proj = nn.Linear(freq_dim, 64)
        self.doppler_proj = nn.Linear(time_dim, 64)
        
    def extract_fresnel_features(self, csi_amp: torch.Tensor) -> torch.Tensor:
        """Extract Fresnel zone related features"""
        batch_size, freq_dim, time_dim = csi_amp.shape
        
        # Compute spatial coherence across frequency
        freq_corr = torch.matmul(csi_amp, csi_amp.transpose(-1, -2)) / time_dim
        
        # Extract dominant eigenvectors (Fresnel zones)
        eigenvalues, eigenvectors = torch.linalg.eigh(freq_corr)
        
        # Take top-k components
        top_k = 3  # First 3 Fresnel zones
        fresnel_features = eigenvectors[:, :, -top_k:]  # [batch, freq, 3]
        
        # Project to feature space
        fresnel_features = fresnel_features.reshape(batch_size, -1)
        fresnel_features = F.relu(self.fresnel_proj(fresnel_features[:, :freq_dim]))
        
        return fresnel_features
    
    def extract_multipath_features(self, csi_complex: torch.Tensor) -> torch.Tensor:
        """Extract multipath propagation features"""
        batch_size = csi_complex.shape[0]
        
        # Frequency domain analysis for path delays
        freq_response = torch.fft.fft(csi_complex, dim=1)
        
        # Power delay profile
        power_delay = torch.abs(freq_response) ** 2
        
        # Extract delay statistics
        mean_delay = torch.mean(power_delay, dim=-1)
        rms_delay = torch.sqrt(torch.var(power_delay, dim=-1))
        
        # Combine features
        multipath_features = torch.cat([mean_delay, rms_delay], dim=-1)
        multipath_features = multipath_features.reshape(batch_size, -1)
        multipath_features = F.relu(self.multipath_proj(multipath_features[:, :self.freq_dim]))
        
        return multipath_features
    
    def extract_doppler_features(self, csi_time: torch.Tensor) -> torch.Tensor:
        """Extract Doppler shift features"""
        batch_size = csi_time.shape[0]
        
        # Time domain FFT for Doppler analysis
        doppler_spectrum = torch.fft.fft(csi_time, dim=-1)
        doppler_mag = torch.abs(doppler_spectrum)
        
        # Doppler statistics
        doppler_centroid = self._compute_spectral_centroid(doppler_mag)
        doppler_spread = self._compute_spectral_spread(doppler_mag, doppler_centroid)
        
        # Combine and project
        doppler_features = torch.cat([doppler_centroid, doppler_spread], dim=-1)
        doppler_features = doppler_features.reshape(batch_size, -1)
        doppler_features = F.relu(self.doppler_proj(doppler_features[:, :self.time_dim]))
        
        return doppler_features
    
    def _compute_spectral_centroid(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid (center of mass)"""
        freqs = torch.arange(spectrum.shape[-1], device=spectrum.device).float()
        centroid = torch.sum(freqs * spectrum, dim=-1) / (torch.sum(spectrum, dim=-1) + 1e-8)
        return centroid
    
    def _compute_spectral_spread(self, spectrum: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        """Compute spectral spread (bandwidth)"""
        freqs = torch.arange(spectrum.shape[-1], device=spectrum.device).float()
        spread = torch.sqrt(
            torch.sum(((freqs - centroid.unsqueeze(-1)) ** 2) * spectrum, dim=-1) / 
            (torch.sum(spectrum, dim=-1) + 1e-8)
        )
        return spread
    
    def forward(self, csi_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract all physics features
        
        Args:
            csi_data: Dictionary with 'amplitude', 'phase', etc.
            
        Returns:
            Physics features [batch, feature_dim]
        """
        features = []
        
        # Fresnel features
        if 'amplitude' in csi_data:
            fresnel_feat = self.extract_fresnel_features(csi_data['amplitude'])
            features.append(fresnel_feat)
        
        # Multipath features
        if 'amplitude' in csi_data and 'phase' in csi_data:
            csi_complex = csi_data['amplitude'] * torch.exp(1j * csi_data['phase'])
            multipath_feat = self.extract_multipath_features(csi_complex)
            features.append(multipath_feat)
        
        # Doppler features
        if 'amplitude' in csi_data:
            doppler_feat = self.extract_doppler_features(csi_data['amplitude'])
            features.append(doppler_feat)
        
        # Concatenate all features
        physics_features = torch.cat(features, dim=-1)
        
        return physics_features


class MultiScaleLSTM(nn.Module):
    """
    多尺度LSTM处理器
    在不同时间尺度上处理CSI序列
    """
    def __init__(self, 
                 input_dim: int = 192,  # Physics features dimension
                 hidden_dims: List[int] = [128, 256, 512],
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.scales = len(hidden_dims)
        
        # Create LSTM for each scale
        self.lstm_layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_dim if i == 0 else hidden_dims[i-1],
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True
                )
            )
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(sum(hidden_dims) * 2, 512),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        
        # Temporal pooling strategies
        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1),
            nn.AdaptiveAvgPool1d(1)
        ])
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Process input through multi-scale LSTMs
        
        Args:
            x: Input features [batch, feature_dim] or [batch, seq_len, feature_dim]
            seq_len: Sequence length for reshaping
            
        Returns:
            Multi-scale features [batch, feature_dim]
        """
        batch_size = x.shape[0]
        
        # Ensure input is 3D
        if len(x.shape) == 2:
            if seq_len is None:
                seq_len = 10  # Default sequence length
            x = x.unsqueeze(1).repeat(1, seq_len, 1)
        
        scale_outputs = []
        current_input = x
        
        for i, (lstm, pooling) in enumerate(zip(self.lstm_layers, self.pooling_layers)):
            # Process through LSTM
            lstm_out, (h_n, c_n) = lstm(current_input)
            
            # Pool over time dimension
            lstm_out_pooled = pooling(lstm_out.transpose(1, 2)).squeeze(-1)
            scale_outputs.append(lstm_out_pooled)
            
            # Downsample for next scale
            if i < len(self.lstm_layers) - 1:
                current_input = lstm_out[:, ::2, :]  # Downsample by 2
        
        # Concatenate all scales
        multi_scale_features = torch.cat(scale_outputs, dim=-1)
        
        # Fuse scales
        fused_features = self.scale_fusion(multi_scale_features)
        
        return fused_features


class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制
    确保时序因果性，只关注过去和当前的信息
    """
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 500):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )
        
        # Relative position encoding
        self.rel_pos_embedding = nn.Parameter(
            torch.randn(1, num_heads, max_seq_len, max_seq_len) * 0.02
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply causal self-attention
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Attended features [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        rel_pos_bias = self.rel_pos_embedding[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores + rel_pos_bias
        
        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class PINNLSTMCausalModel(nn.Module):
    """
    Exp3: Complete Physics-Informed LSTM with Causal Attention Model
    物理特征提取 + 多尺度LSTM + 因果注意力
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 114, 500),
                 num_classes: int = 6,
                 dropout_rate: float = 0.1,
                 use_physics_loss: bool = True):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_physics_loss = use_physics_loss
        
        # Physics feature extraction
        self.physics_extractor = PhysicsFeatureExtractor(
            freq_dim=input_shape[1],
            time_dim=input_shape[2]
        )
        
        # Raw feature extraction (complementary to physics)
        self.raw_feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 192)
        )
        
        # Multi-scale LSTM processing
        self.multi_scale_lstm = MultiScaleLSTM(
            input_dim=192 + 192,  # Physics + raw features
            hidden_dims=[128, 256, 512],
            num_layers=2,
            dropout=dropout_rate
        )
        
        # Causal self-attention
        self.causal_attention = CausalSelfAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout_rate,
            max_seq_len=input_shape[2]
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Physics loss (if enabled)
        if use_physics_loss:
            self.physics_loss_fn = PhysicsConstraintLoss()
        
    def prepare_csi_data(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepare CSI data dictionary from input tensor
        
        Args:
            x: Input tensor [batch, 3, freq, time]
            
        Returns:
            CSI data dictionary
        """
        # Assume channel 0 is amplitude, channel 1 is phase
        csi_data = {
            'amplitude': x[:, 0, :, :],
            'phase': x[:, 1, :, :]
        }
        return csi_data
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input CSI data [batch, channels, freq, time]
            
        Returns:
            predictions: Class predictions
            features: Dictionary of intermediate features for analysis
        """
        batch_size = x.shape[0]
        
        # Prepare CSI data
        csi_data = self.prepare_csi_data(x)
        
        # Extract physics features
        physics_features = self.physics_extractor(csi_data)
        
        # Extract raw features
        raw_features = self.raw_feature_extractor(x)
        
        # Combine features
        combined_features = torch.cat([physics_features, raw_features], dim=-1)
        
        # Process through multi-scale LSTM
        lstm_features = self.multi_scale_lstm(combined_features, seq_len=10)
        
        # Apply causal attention
        # Reshape for attention (need sequence dimension)
        lstm_features_seq = lstm_features.unsqueeze(1).repeat(1, 10, 1)
        attended_features = self.causal_attention(lstm_features_seq)
        
        # Pool over sequence
        final_features = torch.mean(attended_features, dim=1)
        
        # Classification
        predictions = self.classifier(final_features)
        
        # Collect features for analysis
        features_dict = {
            'physics_features': physics_features,
            'raw_features': raw_features,
            'lstm_features': lstm_features,
            'attended_features': attended_features,
            'final_features': final_features
        }
        
        return predictions, features_dict
    
    def compute_loss(self,
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    features: Dict[str, torch.Tensor],
                    alpha: float = 0.3) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined task and physics loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            features: Feature dictionary from forward pass
            alpha: Weight for physics loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        # Task loss
        task_loss = F.cross_entropy(predictions, targets)
        
        loss_dict = {'task_loss': task_loss.item()}
        total_loss = task_loss
        
        # Physics constraint loss
        if self.use_physics_loss and 'physics_features' in features:
            physics_loss = self.physics_loss_fn(features['physics_features'])
            total_loss = task_loss + alpha * physics_loss
            loss_dict['physics_loss'] = physics_loss.item()
            loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class PhysicsConstraintLoss(nn.Module):
    """
    Physics constraint loss for regularization
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, physics_features: torch.Tensor) -> torch.Tensor:
        """
        Compute physics constraint loss
        
        Args:
            physics_features: Extracted physics features
            
        Returns:
            Physics constraint loss
        """
        # Encourage sparsity in physics features
        sparsity_loss = torch.mean(torch.abs(physics_features))
        
        # Encourage smoothness
        if len(physics_features.shape) == 3:
            diff = physics_features[:, 1:, :] - physics_features[:, :-1, :]
            smoothness_loss = torch.mean(torch.abs(diff))
        else:
            smoothness_loss = 0
        
        # Encourage bounded values
        bound_loss = torch.mean(F.relu(torch.abs(physics_features) - 10))
        
        total_loss = 0.3 * sparsity_loss + 0.5 * smoothness_loss + 0.2 * bound_loss
        
        return total_loss


if __name__ == "__main__":
    # Test the model
    print("Testing Exp3: PINN LSTM with Causal Attention...")
    
    # Create model
    model = PINNLSTMCausalModel(
        input_shape=(3, 114, 500),
        num_classes=6,
        use_physics_loss=True
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 114, 500)
    
    # Forward pass
    predictions, features = model(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Feature keys: {features.keys()}")
    
    # Test loss computation
    targets = torch.randint(0, 6, (batch_size,))
    total_loss, loss_dict = model.compute_loss(predictions, targets, features)
    
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Feature dimensions
    print(f"\nFeature dimensions:")
    for key, feat in features.items():
        print(f"  {key}: {feat.shape}")
    
    print("\nExp3 PINN LSTM Causal Model test completed successfully!")
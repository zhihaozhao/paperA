"""
Exp2: Enhanced Model with Physics-Informed Neural Network Loss
Enhanced模型架构 + 自适应物理约束损失函数
Author: Claude 4.1  
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import math

class PhysicsInformedLoss(nn.Module):
    """
    物理信息神经网络损失函数
    包含Fresnel区理论、多径传播模型和Doppler效应约束
    """
    def __init__(self,
                 freq_dim: int = 114,
                 time_dim: int = 500,
                 subcarrier_spacing: float = 312.5e3,  # Hz
                 carrier_freq: float = 5.8e9,  # Hz
                 environment: str = 'indoor'):
        super().__init__()
        self.freq_dim = freq_dim
        self.time_dim = time_dim
        self.subcarrier_spacing = subcarrier_spacing
        self.carrier_freq = carrier_freq
        self.environment = environment
        
        # Physical constants
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / carrier_freq
        
        # Environment-specific parameters
        self.env_params = self._get_environment_params(environment)
        
    def _get_environment_params(self, environment: str) -> Dict:
        """Get environment-specific physical parameters"""
        params = {
            'indoor': {
                'typical_distance': 3.0,  # meters
                'max_velocity': 2.0,  # m/s (walking)
                'multipath_components': 5,
                'path_loss_exponent': 2.5,
                'shadowing_std': 4.0  # dB
            },
            'outdoor': {
                'typical_distance': 10.0,
                'max_velocity': 5.0,  # m/s (running)
                'multipath_components': 3,
                'path_loss_exponent': 3.5,
                'shadowing_std': 8.0
            },
            'corridor': {
                'typical_distance': 5.0,
                'max_velocity': 3.0,
                'multipath_components': 7,
                'path_loss_exponent': 2.0,
                'shadowing_std': 3.0
            }
        }
        return params.get(environment, params['indoor'])
    
    def compute_fresnel_zone_loss(self, csi_amp: torch.Tensor, distance_est: Optional[float] = None) -> torch.Tensor:
        """
        Fresnel区一致性损失
        确保CSI振幅变化符合Fresnel区理论
        
        The nth Fresnel zone radius: r_n = sqrt(n * λ * d1 * d2 / (d1 + d2))
        """
        batch_size = csi_amp.shape[0]
        
        # Estimate distance if not provided
        if distance_est is None:
            distance_est = self.env_params['typical_distance']
        
        # Calculate first three Fresnel zones
        fresnel_radii = []
        for n in range(1, 4):
            # Simplified for line-of-sight: d1 = d2 = distance/2
            radius = math.sqrt(n * self.wavelength * distance_est / 2)
            fresnel_radii.append(radius)
        
        # Extract spatial variation features
        spatial_var = torch.var(csi_amp, dim=(1, 2))  # [batch]
        spatial_mean = torch.mean(csi_amp, dim=(1, 2))  # [batch]
        
        # Fresnel zone constraint: variations should follow zone boundaries
        # First zone should have strong signal
        expected_var = torch.tensor(fresnel_radii[0] / 10, device=csi_amp.device)
        
        # Compute loss as deviation from expected Fresnel behavior
        fresnel_loss = F.mse_loss(spatial_var, expected_var.expand_as(spatial_var))
        
        # Add constraint for signal strength decay
        distance_decay = torch.exp(-distance_est / 10)
        strength_loss = F.mse_loss(spatial_mean, torch.full_like(spatial_mean, distance_decay))
        
        return fresnel_loss + 0.1 * strength_loss
    
    def compute_multipath_propagation_loss(self, csi_complex: torch.Tensor) -> torch.Tensor:
        """
        多径传播一致性损失
        确保CSI相位和振幅变化符合多径传播模型
        
        Multipath model: H(f) = Σ_i α_i * exp(-j*2π*f*τ_i)
        """
        # Compute frequency domain representation
        freq_response = torch.fft.fft(csi_complex, dim=-1)
        
        # Extract multipath components using MUSIC-like approach
        # Compute autocorrelation matrix
        batch_size, freq_bins, time_steps = csi_complex.shape
        
        # Reshape for correlation computation
        csi_reshaped = csi_complex.view(batch_size, -1)
        
        # Compute correlation (simplified)
        correlation = torch.matmul(csi_reshaped.unsqueeze(2), csi_reshaped.unsqueeze(1))
        correlation = correlation / (freq_bins * time_steps)
        
        # Eigenvalue decomposition for multipath detection
        try:
            eigenvalues, _ = torch.linalg.eigh(correlation)
        except:
            # Fallback if eigenvalue computation fails
            eigenvalues = torch.ones(batch_size, 1, device=csi_complex.device)
        
        # Number of significant paths should match environment
        expected_paths = self.env_params['multipath_components']
        
        # Count significant eigenvalues (paths)
        threshold = 0.1 * torch.max(eigenvalues, dim=-1, keepdim=True)[0]
        num_paths = torch.sum(eigenvalues > threshold, dim=-1).float()
        
        # Loss: encourage correct number of multipath components
        path_count_loss = F.mse_loss(num_paths, torch.full_like(num_paths, expected_paths))
        
        # Phase linearity constraint
        phase = torch.angle(freq_response)
        phase_unwrapped = self._unwrap_phase(phase)
        
        # Compute phase slope (related to delay)
        phase_diff = phase_unwrapped[:, :, 1:] - phase_unwrapped[:, :, :-1]
        
        # Second-order difference to check linearity
        second_diff = phase_diff[:, :, 1:] - phase_diff[:, :, :-1]
        linearity_loss = torch.mean(torch.abs(second_diff))
        
        return path_count_loss + 0.5 * linearity_loss
    
    def compute_doppler_effect_loss(self, csi_time_series: torch.Tensor, velocity_est: Optional[float] = None) -> torch.Tensor:
        """
        Doppler效应一致性损失
        确保时域CSI变化符合Doppler频移理论
        
        Doppler shift: f_d = v * f_c * cos(θ) / c
        """
        # Estimate velocity if not provided
        if velocity_est is None:
            velocity_est = self.env_params['max_velocity'] / 2  # Average velocity
        
        # Compute Doppler spectrum via FFT along time dimension
        doppler_spectrum = torch.fft.fft(csi_time_series, dim=-1)
        doppler_magnitude = torch.abs(doppler_spectrum)
        
        # Expected maximum Doppler shift
        max_doppler = velocity_est * self.carrier_freq / self.c
        
        # Normalize frequency bins to actual frequencies
        freq_bins = torch.fft.fftfreq(csi_time_series.shape[-1], device=csi_time_series.device)
        freq_hz = freq_bins * self.subcarrier_spacing
        
        # Doppler spectrum should be concentrated within [-max_doppler, max_doppler]
        doppler_mask = torch.abs(freq_hz) <= max_doppler
        
        # Compute energy ratio inside vs outside Doppler band
        energy_inside = torch.sum(doppler_magnitude * doppler_mask.unsqueeze(0).unsqueeze(0), dim=-1)
        energy_total = torch.sum(doppler_magnitude, dim=-1) + 1e-8
        doppler_ratio = energy_inside / energy_total
        
        # Loss: maximize energy concentration in Doppler band
        doppler_concentration_loss = 1.0 - torch.mean(doppler_ratio)
        
        # Add spectrum smoothness constraint
        spectrum_diff = doppler_magnitude[:, :, 1:] - doppler_magnitude[:, :, :-1]
        smoothness_loss = torch.mean(torch.abs(spectrum_diff))
        
        return doppler_concentration_loss + 0.1 * smoothness_loss
    
    def compute_channel_reciprocity_loss(self, csi_forward: torch.Tensor, csi_backward: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        信道互易性损失
        WiFi信道应满足上下行互易性
        """
        if csi_backward is None:
            # If no backward channel, use time-reversed forward as approximation
            csi_backward = torch.flip(csi_forward, dims=[-1])
        
        # Channel reciprocity: forward and backward channels should be similar
        reciprocity_loss = F.mse_loss(csi_forward, csi_backward)
        
        # Add phase coherence constraint
        phase_forward = torch.angle(csi_forward) if torch.is_complex(csi_forward) else csi_forward
        phase_backward = torch.angle(csi_backward) if torch.is_complex(csi_backward) else csi_backward
        
        phase_diff = torch.abs(phase_forward - phase_backward)
        phase_coherence_loss = torch.mean(torch.sin(phase_diff / 2) ** 2)
        
        return reciprocity_loss + 0.5 * phase_coherence_loss
    
    def _unwrap_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Unwrap phase to remove 2π discontinuities"""
        diff = phase[:, :, 1:] - phase[:, :, :-1]
        diff_unwrapped = diff - 2 * math.pi * torch.round(diff / (2 * math.pi))
        phase_unwrapped = torch.cat([phase[:, :, :1], phase[:, :, :1] + torch.cumsum(diff_unwrapped, dim=-1)], dim=-1)
        return phase_unwrapped
    
    def forward(self, 
                csi_data: Dict[str, torch.Tensor],
                weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute all physics-informed losses
        
        Args:
            csi_data: Dictionary containing CSI measurements
            weights: Optional weights for each loss component
            
        Returns:
            total_loss: Weighted sum of all physics losses
            loss_dict: Dictionary of individual loss components
        """
        if weights is None:
            weights = {
                'fresnel': 1.0,
                'multipath': 1.0,
                'doppler': 1.0,
                'reciprocity': 0.5
            }
        
        loss_dict = {}
        
        # Fresnel zone loss
        if 'amplitude' in csi_data:
            loss_dict['fresnel'] = self.compute_fresnel_zone_loss(csi_data['amplitude'])
        
        # Multipath propagation loss
        if 'complex' in csi_data:
            loss_dict['multipath'] = self.compute_multipath_propagation_loss(csi_data['complex'])
        elif 'amplitude' in csi_data and 'phase' in csi_data:
            # Construct complex from amplitude and phase
            csi_complex = csi_data['amplitude'] * torch.exp(1j * csi_data['phase'])
            loss_dict['multipath'] = self.compute_multipath_propagation_loss(csi_complex)
        
        # Doppler effect loss
        if 'amplitude' in csi_data:
            loss_dict['doppler'] = self.compute_doppler_effect_loss(csi_data['amplitude'])
        
        # Channel reciprocity loss
        if 'forward' in csi_data:
            backward = csi_data.get('backward', None)
            loss_dict['reciprocity'] = self.compute_channel_reciprocity_loss(csi_data['forward'], backward)
        
        # Compute weighted total loss
        total_loss = torch.tensor(0.0, device=next(iter(csi_data.values())).device)
        for key, loss in loss_dict.items():
            total_loss += weights.get(key, 1.0) * loss
        
        return total_loss, loss_dict


class AdaptivePhysicsWeightScheduler(nn.Module):
    """
    自适应物理损失权重调度器
    根据训练动态自动调整物理约束的权重
    """
    def __init__(self,
                 initial_weight: float = 0.1,
                 target_weight: float = 0.5,
                 warmup_epochs: int = 10,
                 adaptation_window: int = 5):
        super().__init__()
        
        self.initial_weight = initial_weight
        self.target_weight = target_weight
        self.warmup_epochs = warmup_epochs
        self.adaptation_window = adaptation_window
        
        # Learnable weight parameters for each physics component
        self.weight_params = nn.ParameterDict({
            'fresnel': nn.Parameter(torch.tensor(initial_weight)),
            'multipath': nn.Parameter(torch.tensor(initial_weight)),
            'doppler': nn.Parameter(torch.tensor(initial_weight)),
            'reciprocity': nn.Parameter(torch.tensor(initial_weight * 0.5))
        })
        
        # History tracking
        self.loss_history = {
            'task': [],
            'physics': [],
            'validation': []
        }
        
    def update_weights(self, 
                      epoch: int,
                      task_loss: float,
                      physics_losses: Dict[str, float],
                      val_performance: Optional[float] = None):
        """
        Update weights based on training dynamics
        
        Args:
            epoch: Current training epoch
            task_loss: Current task loss value
            physics_losses: Dictionary of physics loss values
            val_performance: Optional validation performance
        """
        # Record history
        self.loss_history['task'].append(task_loss)
        self.loss_history['physics'].append(sum(physics_losses.values()))
        if val_performance is not None:
            self.loss_history['validation'].append(val_performance)
        
        # Warmup phase
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            for key in self.weight_params:
                target = self.target_weight if key != 'reciprocity' else self.target_weight * 0.5
                self.weight_params[key].data = self.initial_weight + (target - self.initial_weight) * progress
            return
        
        # Adaptive phase
        if len(self.loss_history['task']) >= self.adaptation_window:
            # Analyze recent trends
            recent_task = self.loss_history['task'][-self.adaptation_window:]
            recent_physics = self.loss_history['physics'][-self.adaptation_window:]
            
            # Compute trends
            task_trend = np.polyfit(range(self.adaptation_window), recent_task, 1)[0]
            physics_trend = np.polyfit(range(self.adaptation_window), recent_physics, 1)[0]
            
            # Adjust weights based on trends
            for key, param in self.weight_params.items():
                current_val = param.data.item()
                
                # If task loss not improving but physics loss is, increase physics weight
                if task_trend > -0.001 and physics_trend < -0.001:
                    param.data = torch.clamp(param.data * 1.1, 0.01, 1.0)
                # If task loss improving but physics loss stagnant, decrease physics weight
                elif task_trend < -0.001 and physics_trend > -0.001:
                    param.data = torch.clamp(param.data * 0.9, 0.01, 1.0)
                
                # Component-specific adjustments
                if key in physics_losses:
                    component_loss = physics_losses[key]
                    # If component loss is too high, increase its weight
                    if component_loss > 1.0:
                        param.data = torch.clamp(param.data * 1.05, 0.01, 1.0)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weight values"""
        return {key: param.data.item() for key, param in self.weight_params.items()}


class EnhancedModelWithPINNLoss(nn.Module):
    """
    Exp2: Enhanced Model with Physics-Informed Loss
    保持Enhanced模型架构，添加自适应物理约束损失
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 114, 500),
                 num_classes: int = 6,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # ========== Enhanced Model Architecture (unchanged) ==========
        # Multi-scale CNN branches
        self.conv_branches = nn.ModuleList([
            self._make_conv_branch(kernel_size=3),
            self._make_conv_branch(kernel_size=5),
            self._make_conv_branch(kernel_size=7)
        ])
        
        # Squeeze-and-Excitation fusion
        self.se_fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(384, 96, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(96, 384, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=384,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # ========== Physics-Informed Components ==========
        self.physics_loss = PhysicsInformedLoss(
            freq_dim=input_shape[1],
            time_dim=input_shape[2]
        )
        self.weight_scheduler = AdaptivePhysicsWeightScheduler()
        
        # Training state
        self.current_epoch = 0
    
    def _make_conv_branch(self, kernel_size: int) -> nn.Module:
        """Create a convolutional branch"""
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(self.input_shape[0], 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (same as Enhanced model)
        
        Args:
            x: Input CSI data [batch, channels, freq, time]
            
        Returns:
            Class predictions [batch, num_classes]
        """
        # Multi-scale feature extraction
        branch_outputs = []
        for branch in self.conv_branches:
            branch_outputs.append(branch(x))
        
        # Concatenate multi-scale features
        features = torch.cat(branch_outputs, dim=1)  # [batch, 384, H, W]
        
        # SE attention
        se_weights = self.se_fusion(features)
        features = features * se_weights
        
        # Global pooling
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)  # [batch, 384]
        
        # Temporal attention
        features_seq = features.unsqueeze(1)  # [batch, 1, 384]
        attended, _ = self.temporal_attention(features_seq, features_seq, features_seq)
        attended = attended.squeeze(1)  # [batch, 384]
        
        # Classification
        logits = self.classifier(attended)
        
        return logits
    
    def compute_loss(self,
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    csi_data: Dict[str, torch.Tensor],
                    epoch: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with physics constraints
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            csi_data: Dictionary of CSI data
            epoch: Current epoch (for weight scheduling)
            
        Returns:
            total_loss: Combined task and physics loss
            loss_dict: Dictionary of loss components
        """
        # Task loss
        task_loss = F.cross_entropy(predictions, targets)
        
        # Physics losses
        physics_weights = self.weight_scheduler.get_current_weights()
        physics_total, physics_losses = self.physics_loss(csi_data, physics_weights)
        
        # Update weight scheduler
        if epoch is not None:
            self.current_epoch = epoch
            with torch.no_grad():
                val_acc = (predictions.argmax(dim=1) == targets).float().mean()
                self.weight_scheduler.update_weights(
                    epoch, 
                    task_loss.item(),
                    {k: v.item() for k, v in physics_losses.items()},
                    val_acc.item()
                )
        
        # Combine losses
        total_loss = task_loss + physics_total
        
        # Create loss dictionary
        loss_dict = {
            'task_loss': task_loss.item(),
            'physics_total': physics_total.item(),
            'total_loss': total_loss.item()
        }
        
        # Add individual physics losses
        for key, loss in physics_losses.items():
            loss_dict[f'physics_{key}'] = loss.item()
            loss_dict[f'weight_{key}'] = physics_weights[key]
        
        return total_loss, loss_dict
    
    def get_physics_consistency_score(self, csi_data: Dict[str, torch.Tensor]) -> float:
        """
        Compute physics consistency score (0-1, higher is better)
        
        Args:
            csi_data: CSI data dictionary
            
        Returns:
            Physics consistency score
        """
        with torch.no_grad():
            _, physics_losses = self.physics_loss(csi_data)
            
            # Convert losses to scores (lower loss = higher score)
            scores = {}
            for key, loss in physics_losses.items():
                scores[key] = 1.0 / (1.0 + loss.item())
            
            # Average score
            avg_score = sum(scores.values()) / len(scores) if scores else 0.0
            
            return avg_score


if __name__ == "__main__":
    # Test the model
    print("Testing Exp2: Enhanced Model with PINN Loss...")
    
    # Create model
    model = EnhancedModelWithPINNLoss(
        input_shape=(3, 114, 500),
        num_classes=6
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 114, 500)
    dummy_csi_data = {
        'amplitude': torch.randn(batch_size, 114, 500),
        'phase': torch.randn(batch_size, 114, 500),
        'complex': torch.randn(batch_size, 114, 500, dtype=torch.complex64)
    }
    
    # Test inference
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Test physics consistency score
        physics_score = model.get_physics_consistency_score(dummy_csi_data)
        print(f"Physics consistency score: {physics_score:.4f}")
    
    # Test training mode with loss computation
    model.train()
    predictions = model(dummy_input)
    targets = torch.randint(0, 6, (batch_size,))
    
    total_loss, loss_dict = model.compute_loss(
        predictions, targets, dummy_csi_data, epoch=5
    )
    
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Test weight scheduler
    print(f"\nCurrent physics weights:")
    weights = model.weight_scheduler.get_current_weights()
    for key, value in weights.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nExp2 Enhanced Model with PINN Loss test completed successfully!")
"""
Exp1: Enhanced Model for Sim2Real Transfer Learning
用于从仿真到真实数据的迁移学习的Enhanced模型
Author: Claude 4.1
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import math

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiScaleCNNExtractor(nn.Module):
    """多尺度CNN特征提取器"""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Branch 1: 3x3 kernels
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SqueezeExcitation(128),
            nn.MaxPool2d(2)
        )
        
        # Branch 2: 5x5 kernels
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SqueezeExcitation(128),
            nn.MaxPool2d(2)
        )
        
        # Branch 3: 7x7 kernels
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SqueezeExcitation(128),
            nn.MaxPool2d(2)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SqueezeExcitation(256)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract multi-scale features
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        
        # Concatenate features
        multi_scale_features = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Fuse features
        fused_features = self.fusion(multi_scale_features)
        
        return fused_features

class TemporalAttention(nn.Module):
    """时序注意力机制"""
    def __init__(self, feature_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, feature_dim, height, width]
        batch, channels, h, w = x.shape
        
        # Reshape to sequence format
        x_seq = x.view(batch, channels, h * w).permute(0, 2, 1)  # [batch, h*w, channels]
        
        # Apply attention
        attended, _ = self.attention(x_seq, x_seq, x_seq)
        attended = self.dropout(attended)
        attended = self.norm(attended + x_seq)
        
        # Reshape back
        attended = attended.permute(0, 2, 1).view(batch, channels, h, w)
        
        return attended

class DomainAdaptationModule(nn.Module):
    """领域自适应模块 - 用于Sim2Real迁移"""
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        # Domain discriminator
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # 2 domains: sim and real
        )
        
        # Feature alignment layers
        self.sim_adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.real_adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, features: torch.Tensor, domain: str = 'sim') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Input features [batch, feature_dim]
            domain: 'sim' or 'real'
        
        Returns:
            adapted_features: Domain-adapted features
            domain_logits: Domain classification logits
        """
        # Domain classification
        domain_logits = self.domain_classifier(features)
        
        # Domain-specific adaptation
        if domain == 'sim':
            adapted_features = self.sim_adapter(features)
        else:
            adapted_features = self.real_adapter(features)
            
        return adapted_features, domain_logits

class EnhancedSim2RealModel(nn.Module):
    """
    Exp1: Enhanced Model optimized for Sim2Real Transfer
    专门优化用于从仿真到真实数据迁移的Enhanced模型
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 114, 500),
                 num_classes: int = 6,
                 dropout_rate: float = 0.5,
                 use_domain_adaptation: bool = True):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_domain_adaptation = use_domain_adaptation
        
        # Multi-scale CNN feature extractor
        self.feature_extractor = MultiScaleCNNExtractor(in_channels=input_shape[0])
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(feature_dim=256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Domain adaptation module
        if use_domain_adaptation:
            self.domain_adapter = DomainAdaptationModule(feature_dim=256)
        
        # Task classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Sim2Real specific components
        self.sim_noise_augmentation = nn.Dropout2d(0.1)  # Simulate real-world noise
        self.feature_normalization = nn.BatchNorm1d(256)
        
    def extract_features(self, x: torch.Tensor, augment_sim: bool = False) -> torch.Tensor:
        """
        Extract features with optional simulation augmentation
        
        Args:
            x: Input CSI data [batch, channels, freq, time]
            augment_sim: Whether to apply simulation augmentation
        
        Returns:
            Extracted features [batch, feature_dim]
        """
        # Apply simulation noise augmentation if training on sim data
        if augment_sim and self.training:
            x = self.sim_noise_augmentation(x)
        
        # Multi-scale feature extraction
        features = self.feature_extractor(x)
        
        # Temporal attention
        features = self.temporal_attention(features)
        
        # Global pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # Feature normalization for better transfer
        features = self.feature_normalization(features)
        
        return features
    
    def forward(self, 
                x: torch.Tensor,
                domain: str = 'sim',
                return_domain_logits: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with domain adaptation
        
        Args:
            x: Input CSI data
            domain: Current domain ('sim' or 'real')
            return_domain_logits: Whether to return domain classification logits
        
        Returns:
            Class predictions, optionally domain logits
        """
        # Extract features with appropriate augmentation
        augment = (domain == 'sim')
        features = self.extract_features(x, augment_sim=augment)
        
        # Domain adaptation
        if self.use_domain_adaptation:
            adapted_features, domain_logits = self.domain_adapter(features, domain)
        else:
            adapted_features = features
            domain_logits = None
        
        # Classification
        class_logits = self.classifier(adapted_features)
        
        if return_domain_logits and domain_logits is not None:
            return class_logits, domain_logits
        return class_logits
    
    def compute_sim2real_loss(self,
                              sim_data: torch.Tensor,
                              real_data: torch.Tensor,
                              sim_labels: torch.Tensor,
                              real_labels: Optional[torch.Tensor] = None,
                              alpha: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Compute Sim2Real training loss
        
        Args:
            sim_data: Simulation data
            real_data: Real data
            sim_labels: Simulation labels
            real_labels: Real labels (optional for semi-supervised)
            alpha: Domain adaptation weight
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Forward pass for simulation data
        sim_class_logits, sim_domain_logits = self.forward(
            sim_data, domain='sim', return_domain_logits=True
        )
        
        # Forward pass for real data
        real_class_logits, real_domain_logits = self.forward(
            real_data, domain='real', return_domain_logits=True
        )
        
        # Classification loss for simulation data
        losses['sim_class'] = F.cross_entropy(sim_class_logits, sim_labels)
        
        # Classification loss for real data (if labels available)
        if real_labels is not None:
            losses['real_class'] = F.cross_entropy(real_class_logits, real_labels)
        
        # Domain adaptation loss
        if self.use_domain_adaptation:
            # Create domain labels
            sim_domain_labels = torch.zeros(sim_data.size(0), dtype=torch.long, device=sim_data.device)
            real_domain_labels = torch.ones(real_data.size(0), dtype=torch.long, device=real_data.device)
            
            # Domain classification loss
            losses['domain'] = F.cross_entropy(sim_domain_logits, sim_domain_labels) + \
                               F.cross_entropy(real_domain_logits, real_domain_labels)
        
        # Total loss
        total_loss = losses['sim_class']
        if 'real_class' in losses:
            total_loss += losses['real_class']
        if 'domain' in losses:
            total_loss += alpha * losses['domain']
        
        losses['total'] = total_loss
        
        return losses
    
    def adapt_to_target_domain(self, 
                               target_loader,
                               num_epochs: int = 10,
                               lr: float = 0.0001):
        """
        Fine-tune model on target domain data
        
        Args:
            target_loader: DataLoader for target domain
            num_epochs: Number of adaptation epochs
            lr: Learning rate for adaptation
        """
        # Freeze feature extractor, only adapt domain-specific layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Create optimizer for adaptation layers
        adapt_params = list(self.classifier.parameters())
        if self.use_domain_adaptation:
            adapt_params += list(self.domain_adapter.parameters())
        
        optimizer = torch.optim.Adam(adapt_params, lr=lr)
        
        for epoch in range(num_epochs):
            for batch_idx, (data, labels) in enumerate(target_loader):
                self.train()
                
                # Forward pass
                predictions = self.forward(data, domain='real')
                loss = F.cross_entropy(predictions, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    print(f'Adaptation Epoch {epoch} [{batch_idx}/{len(target_loader)}] Loss: {loss.item():.4f}')
        
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True


class Sim2RealTrainer:
    """
    Sim2Real训练器
    专门用于仿真到真实数据的迁移学习
    """
    def __init__(self,
                 model: EnhancedSim2RealModel,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Separate optimizers for different training phases
        self.sim_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.adapt_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Training history
        self.history = {
            'sim_train': {'loss': [], 'acc': []},
            'sim_val': {'loss': [], 'acc': []},
            'real_train': {'loss': [], 'acc': []},
            'real_val': {'loss': [], 'acc': []}
        }
    
    def pretrain_on_simulation(self, sim_loader, val_loader, num_epochs: int = 50):
        """
        Phase 1: Pretrain on simulation data
        """
        print("Phase 1: Pretraining on simulation data...")
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, labels) in enumerate(sim_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                predictions = self.model(data, domain='sim')
                loss = F.cross_entropy(predictions, labels)
                
                # Backward pass
                self.sim_optimizer.zero_grad()
                loss.backward()
                self.sim_optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = predictions.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, domain='sim')
            
            # Record history
            train_acc = 100. * correct / total
            self.history['sim_train']['loss'].append(train_loss / len(sim_loader))
            self.history['sim_train']['acc'].append(train_acc)
            self.history['sim_val']['loss'].append(val_loss)
            self.history['sim_val']['acc'].append(val_acc)
            
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    def adapt_with_real_data(self, 
                             real_loader,
                             real_val_loader,
                             num_epochs: int = 20,
                             few_shot: bool = False):
        """
        Phase 2: Adapt with real data
        """
        print("Phase 2: Adapting with real data...")
        
        if few_shot:
            print("Using few-shot adaptation strategy...")
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, labels) in enumerate(real_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                predictions = self.model(data, domain='real')
                loss = F.cross_entropy(predictions, labels)
                
                # Add regularization for few-shot
                if few_shot:
                    # L2 regularization to prevent overfitting
                    l2_reg = sum(p.pow(2.0).sum() for p in self.model.classifier.parameters())
                    loss = loss + 0.001 * l2_reg
                
                # Backward pass
                self.adapt_optimizer.zero_grad()
                loss.backward()
                self.adapt_optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = predictions.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Validation
            val_loss, val_acc = self.validate(real_val_loader, domain='real')
            
            # Record history
            train_acc = 100. * correct / total
            self.history['real_train']['loss'].append(train_loss / len(real_loader))
            self.history['real_train']['acc'].append(train_acc)
            self.history['real_val']['loss'].append(val_loss)
            self.history['real_val']['acc'].append(val_acc)
            
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    def validate(self, val_loader, domain: str = 'real'):
        """Validate model performance"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                predictions = self.model(data, domain=domain)
                loss = F.cross_entropy(predictions, labels)
                
                val_loss += loss.item()
                _, predicted = predictions.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = val_loss / len(val_loader)
        
        return avg_loss, accuracy


if __name__ == "__main__":
    # Test the model
    print("Testing Exp1: Enhanced Sim2Real Model...")
    
    # Create model
    model = EnhancedSim2RealModel(
        input_shape=(3, 114, 500),
        num_classes=6,
        use_domain_adaptation=True
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    sim_data = torch.randn(batch_size, 3, 114, 500)
    real_data = torch.randn(batch_size, 3, 114, 500)
    
    # Test simulation forward
    sim_output = model(sim_data, domain='sim')
    print(f"Simulation output shape: {sim_output.shape}")
    
    # Test real forward
    real_output = model(real_data, domain='real')
    print(f"Real output shape: {real_output.shape}")
    
    # Test Sim2Real loss
    sim_labels = torch.randint(0, 6, (batch_size,))
    real_labels = torch.randint(0, 6, (batch_size,))
    
    losses = model.compute_sim2real_loss(
        sim_data, real_data, sim_labels, real_labels
    )
    
    print("\nSim2Real losses:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")
    
    print("\nExp1 Enhanced Sim2Real Model test completed successfully!")
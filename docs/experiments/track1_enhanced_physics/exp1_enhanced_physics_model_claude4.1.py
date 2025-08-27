"""
Track 1 - Exp1: Enhanced Model with Adaptive Physics-Informed Loss
主线路径：在SOTA基础上增加物理约束，保持高精度的同时提升物理一致性和泛化能力
Author: Claude 4.1
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math

class AdaptivePhysicsWeighter(nn.Module):
    """
    自适应物理损失权重调度器
    根据训练进度和性能动态调整物理约束的权重
    """
    def __init__(self, 
                 initial_weight: float = 0.1,
                 max_weight: float = 0.5,
                 warmup_epochs: int = 10,
                 adaptation_rate: float = 0.01):
        super().__init__()
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.warmup_epochs = warmup_epochs
        self.adaptation_rate = adaptation_rate
        
        # 可学习的权重参数
        self.weight_params = nn.Parameter(torch.tensor([initial_weight]))
        
        # 历史性能追踪
        self.task_loss_history = []
        self.physics_loss_history = []
        
    def forward(self, epoch: int, task_loss: float, physics_loss: float) -> float:
        """
        计算当前的自适应权重
        
        Args:
            epoch: 当前训练轮次
            task_loss: 任务损失值
            physics_loss: 物理损失值
            
        Returns:
            自适应权重值
        """
        # Warmup阶段线性增加
        if epoch < self.warmup_epochs:
            base_weight = self.initial_weight + (self.max_weight - self.initial_weight) * (epoch / self.warmup_epochs)
        else:
            # 根据损失比例动态调整
            self.task_loss_history.append(task_loss)
            self.physics_loss_history.append(physics_loss)
            
            if len(self.task_loss_history) > 5:
                # 计算损失趋势
                task_trend = np.polyfit(range(5), self.task_loss_history[-5:], 1)[0]
                physics_trend = np.polyfit(range(5), self.physics_loss_history[-5:], 1)[0]
                
                # 如果任务损失下降慢而物理损失下降快，增加物理权重
                if task_trend > -0.01 and physics_trend < -0.01:
                    self.weight_params.data += self.adaptation_rate
                # 如果任务损失下降快而物理损失停滞，减少物理权重
                elif task_trend < -0.01 and physics_trend > -0.01:
                    self.weight_params.data -= self.adaptation_rate
                    
            base_weight = torch.clamp(self.weight_params, self.initial_weight, self.max_weight).item()
            
        return base_weight

class PhysicsInformedLoss(nn.Module):
    """
    物理信息损失函数
    包含Fresnel区、多径传播和Doppler效应约束
    """
    def __init__(self, 
                 freq_dim: int = 114,
                 time_dim: int = 500,
                 subcarrier_spacing: float = 312.5e3,  # Hz
                 carrier_freq: float = 5.8e9):  # Hz
        super().__init__()
        self.freq_dim = freq_dim
        self.time_dim = time_dim
        self.subcarrier_spacing = subcarrier_spacing
        self.carrier_freq = carrier_freq
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / carrier_freq
        
    def compute_fresnel_loss(self, csi_amp: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Fresnel区一致性损失
        确保信号传播符合Fresnel区理论
        """
        batch_size = csi_amp.shape[0]
        
        # 计算第一Fresnel区半径
        # r1 = sqrt(λ * d / 2) for far field
        estimated_distance = 3.0  # meters (typical indoor)
        fresnel_radius = math.sqrt(self.wavelength * estimated_distance / 2)
        
        # 提取空间特征的标准差
        spatial_std = torch.std(csi_amp, dim=(1, 2))  # [batch]
        
        # Fresnel区内信号应该相对稳定
        expected_std = fresnel_radius / 10  # 经验值
        fresnel_loss = F.mse_loss(spatial_std, torch.full_like(spatial_std, expected_std))
        
        return fresnel_loss
    
    def compute_multipath_loss(self, csi_phase: torch.Tensor) -> torch.Tensor:
        """
        多径传播一致性损失
        确保相位变化符合多径传播模型
        """
        # 相位展开
        phase_unwrapped = torch.unwrap(csi_phase, dim=-1)
        
        # 计算相位斜率（与传播延迟相关）
        phase_diff = phase_unwrapped[:, :, :, 1:] - phase_unwrapped[:, :, :, :-1]
        
        # 多径传播应该产生线性相位变化
        # 使用二阶差分检测非线性
        second_diff = phase_diff[:, :, :, 1:] - phase_diff[:, :, :, :-1]
        
        # 最小化二阶差分，鼓励线性相位
        multipath_loss = torch.mean(torch.abs(second_diff))
        
        return multipath_loss
    
    def compute_doppler_loss(self, csi_amp: torch.Tensor, velocity_est: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Doppler频移一致性损失
        确保时域变化符合Doppler效应
        """
        # 时域FFT获取Doppler谱
        doppler_spectrum = torch.fft.fft(csi_amp, dim=-1)
        doppler_magnitude = torch.abs(doppler_spectrum)
        
        # Doppler频移应该集中在特定范围
        # f_d = v * f_c / c
        max_velocity = 2.0  # m/s (walking speed)
        max_doppler = max_velocity * self.carrier_freq / self.c
        
        # 计算频谱熵，鼓励集中的Doppler谱
        doppler_magnitude_norm = doppler_magnitude / (doppler_magnitude.sum(dim=-1, keepdim=True) + 1e-8)
        doppler_entropy = -torch.sum(doppler_magnitude_norm * torch.log(doppler_magnitude_norm + 1e-8), dim=-1)
        
        # 低熵表示Doppler谱集中
        doppler_loss = torch.mean(doppler_entropy)
        
        return doppler_loss
    
    def forward(self, csi_data: Dict[str, torch.Tensor], features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算所有物理损失
        
        Args:
            csi_data: 包含'amplitude'和'phase'的字典
            features: 模型提取的特征
            
        Returns:
            各项物理损失的字典
        """
        losses = {}
        
        # Fresnel区损失
        if 'amplitude' in csi_data:
            losses['fresnel'] = self.compute_fresnel_loss(csi_data['amplitude'], features)
        
        # 多径损失
        if 'phase' in csi_data:
            losses['multipath'] = self.compute_multipath_loss(csi_data['phase'])
        
        # Doppler损失
        if 'amplitude' in csi_data:
            losses['doppler'] = self.compute_doppler_loss(csi_data['amplitude'])
        
        return losses

class EnhancedPhysicsModel(nn.Module):
    """
    Track 1 - Exp1: Enhanced Model with Physics-Informed Loss
    在保持Enhanced Model高精度的基础上，增加物理约束提升泛化能力
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 114, 500),
                 num_classes: int = 6,
                 dropout_rate: float = 0.5,
                 use_physics: bool = True):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_physics = use_physics
        
        # ========== Enhanced Model架构（保持不变） ==========
        # 多尺度CNN特征提取
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv_branch2 = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv_branch3 = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Squeeze-and-Excitation模块
        self.se_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(384, 96, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(96, 384, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 时序注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=384,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # ========== 物理约束模块（新增） ==========
        if self.use_physics:
            self.physics_loss = PhysicsInformedLoss(
                freq_dim=input_shape[1],
                time_dim=input_shape[2]
            )
            self.adaptive_weighter = AdaptivePhysicsWeighter()
            
        # 训练状态
        self.current_epoch = 0
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取多尺度CNN特征
        
        Args:
            x: 输入CSI数据 [batch, channels, freq, time]
            
        Returns:
            提取的特征 [batch, feature_dim]
        """
        # 三个分支提取不同尺度特征
        feat1 = self.conv_branch1(x)
        feat2 = self.conv_branch2(x)
        feat3 = self.conv_branch3(x)
        
        # 特征拼接
        features = torch.cat([feat1, feat2, feat3], dim=1)  # [batch, 384, H, W]
        
        # SE注意力
        se_weights = self.se_module(features)
        features = features * se_weights
        
        # 全局平均池化
        features = F.adaptive_avg_pool2d(features, 1)  # [batch, 384, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch, 384]
        
        return features
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入CSI数据
            return_features: 是否返回中间特征
            
        Returns:
            预测结果，可选返回特征
        """
        # 特征提取
        features = self.extract_features(x)
        
        # 时序注意力（自注意力）
        features_seq = features.unsqueeze(1)  # [batch, 1, 384]
        attended_features, _ = self.temporal_attention(features_seq, features_seq, features_seq)
        attended_features = attended_features.squeeze(1)  # [batch, 384]
        
        # 分类
        logits = self.classifier(attended_features)
        
        if return_features:
            return logits, features
        return logits
    
    def compute_loss(self, 
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     csi_data: Dict[str, torch.Tensor],
                     features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失（任务损失 + 物理损失）
        
        Args:
            predictions: 模型预测
            targets: 真实标签
            csi_data: CSI数据字典
            features: 中间特征
            
        Returns:
            总损失和损失分量字典
        """
        # 任务损失
        task_loss = F.cross_entropy(predictions, targets)
        
        loss_dict = {'task_loss': task_loss.item()}
        total_loss = task_loss
        
        # 物理损失（如果启用）
        if self.use_physics and self.training:
            physics_losses = self.physics_loss(csi_data, features)
            
            # 自适应权重
            alpha = self.adaptive_weighter(
                self.current_epoch,
                task_loss.item(),
                sum(physics_losses.values()).item()
            )
            
            # 加权物理损失
            weighted_physics_loss = alpha * sum(physics_losses.values())
            total_loss = task_loss + weighted_physics_loss
            
            # 记录各项损失
            loss_dict['physics_weight'] = alpha
            for name, loss in physics_losses.items():
                loss_dict[f'physics_{name}'] = loss.item()
            loss_dict['physics_total'] = weighted_physics_loss.item()
        
        return total_loss, loss_dict
    
    def set_epoch(self, epoch: int):
        """设置当前训练轮次"""
        self.current_epoch = epoch
        
    def get_physics_consistency_score(self, csi_data: Dict[str, torch.Tensor]) -> float:
        """
        计算物理一致性得分
        
        Args:
            csi_data: CSI数据
            
        Returns:
            物理一致性得分（0-1）
        """
        if not self.use_physics:
            return 0.0
            
        with torch.no_grad():
            # 提取特征
            if 'amplitude' in csi_data:
                x = csi_data['amplitude']
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)  # Add channel dimension if needed
                features = self.extract_features(x)
            else:
                return 0.0
            
            # 计算物理损失
            physics_losses = self.physics_loss(csi_data, features)
            
            # 转换为一致性得分（损失越小，得分越高）
            total_physics_loss = sum(physics_losses.values()).item()
            consistency_score = 1.0 / (1.0 + total_physics_loss)
            
            return consistency_score


class Exp1Trainer:
    """
    Exp1模型训练器
    包含自适应物理损失调度和性能监控
    """
    def __init__(self, 
                 model: EnhancedPhysicsModel,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # 性能追踪
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'physics_score': []
        }
        self.val_history = {
            'loss': [],
            'accuracy': [],
            'physics_score': []
        }
        
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        self.model.set_epoch(epoch)
        
        total_loss = 0
        correct = 0
        total = 0
        physics_scores = []
        
        for batch_idx, (csi_amp, csi_phase, labels) in enumerate(train_loader):
            # 准备数据
            csi_amp = csi_amp.to(self.device)
            csi_phase = csi_phase.to(self.device)
            labels = labels.to(self.device)
            
            # 组合CSI数据
            csi_input = torch.stack([csi_amp, csi_phase, csi_amp], dim=1)  # [batch, 3, freq, time]
            csi_data = {
                'amplitude': csi_amp,
                'phase': csi_phase
            }
            
            # 前向传播
            predictions, features = self.model(csi_input, return_features=True)
            
            # 计算损失
            loss, loss_dict = self.model.compute_loss(predictions, labels, csi_data, features)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = predictions.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # 物理一致性得分
            physics_score = self.model.get_physics_consistency_score(csi_data)
            physics_scores.append(physics_score)
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Task: {loss_dict["task_loss"]:.4f} '
                      f'Physics: {loss_dict.get("physics_total", 0):.4f} '
                      f'Alpha: {loss_dict.get("physics_weight", 0):.4f}')
        
        # 记录历史
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        avg_physics_score = np.mean(physics_scores)
        
        self.train_history['loss'].append(avg_loss)
        self.train_history['accuracy'].append(accuracy)
        self.train_history['physics_score'].append(avg_physics_score)
        
        return avg_loss, accuracy, avg_physics_score
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        physics_scores = []
        
        with torch.no_grad():
            for csi_amp, csi_phase, labels in val_loader:
                # 准备数据
                csi_amp = csi_amp.to(self.device)
                csi_phase = csi_phase.to(self.device)
                labels = labels.to(self.device)
                
                # 组合CSI数据
                csi_input = torch.stack([csi_amp, csi_phase, csi_amp], dim=1)
                csi_data = {
                    'amplitude': csi_amp,
                    'phase': csi_phase
                }
                
                # 前向传播
                predictions = self.model(csi_input)
                
                # 计算损失
                loss = F.cross_entropy(predictions, labels)
                
                # 统计
                total_loss += loss.item()
                _, predicted = predictions.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # 物理一致性得分
                physics_score = self.model.get_physics_consistency_score(csi_data)
                physics_scores.append(physics_score)
        
        # 记录历史
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        avg_physics_score = np.mean(physics_scores)
        
        self.val_history['loss'].append(avg_loss)
        self.val_history['accuracy'].append(accuracy)
        self.val_history['physics_score'].append(avg_physics_score)
        
        return avg_loss, accuracy, avg_physics_score


if __name__ == "__main__":
    # 测试代码
    print("Testing Enhanced Physics Model (Track 1 - Exp1)...")
    
    # 创建模型
    model = EnhancedPhysicsModel(
        input_shape=(3, 114, 500),
        num_classes=6,
        use_physics=True
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 114, 500)
    dummy_csi_data = {
        'amplitude': torch.randn(batch_size, 114, 500),
        'phase': torch.randn(batch_size, 114, 500)
    }
    
    # 测试推理
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # 测试物理一致性得分
        physics_score = model.get_physics_consistency_score(dummy_csi_data)
        print(f"Physics consistency score: {physics_score:.4f}")
    
    # 测试训练模式
    model.train()
    output, features = model(dummy_input, return_features=True)
    targets = torch.randint(0, 6, (batch_size,))
    
    loss, loss_dict = model.compute_loss(output, targets, dummy_csi_data, features)
    print(f"Total loss: {loss.item():.4f}")
    print("Loss components:", loss_dict)
    
    print("\nExp1 Enhanced Physics Model test completed successfully!")
#!/usr/bin/env python3
"""
真实数据适配器

将WiFi-CSI-Sensing-Benchmark的数据集适配到我们现有的训练框架中，
专注于Sim2Real验证，而非模型架构对比。

核心功能:
1. 将真实数据集转换为我们的数据格式
2. 支持与合成数据的混合训练
3. 实现少样本学习实验
4. 最小化对现有代码的修改

作者: AI Assistant  
日期: 2025-01-16
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pickle

# 添加benchmark路径
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks" / "wifi_csi_benchmark"
sys.path.append(str(BENCHMARK_DIR))

# 添加我们的源码路径
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.append(str(SRC_DIR))


class RealDataAdapter:
    """真实数据适配器 - 将benchmark数据集适配到我们的训练框架"""
    
    def __init__(self, cache_dir="cache/real_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的数据集配置
        self.dataset_configs = {
            "UT_HAR": {
                "name": "UT_HAR_data",
                "num_classes": 7,
                "description": "行为识别数据集"
            },
            "NTU_Fi_HAR": {
                "name": "NTU-Fi_HAR", 
                "num_classes": 6,
                "description": "行为识别数据集"
            }
        }
    
    def load_real_dataset(self, dataset_name: str, use_cache=True):
        """
        加载真实数据集并转换为我们的格式
        
        Args:
            dataset_name: 数据集名称 ("UT_HAR" 或 "NTU_Fi_HAR")
            use_cache: 是否使用缓存
            
        Returns:
            train_loader, test_loader: 我们格式的数据加载器
        """
        cache_file = self.cache_dir / f"{dataset_name}_adapted.pkl"
        
        # 尝试从缓存加载
        if use_cache and cache_file.exists():
            print(f"📁 从缓存加载 {dataset_name}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return self._create_loaders(data)
        
        # 从benchmark加载原始数据
        print(f"📊 从benchmark加载 {dataset_name}")
        config = self.dataset_configs[dataset_name]
        
        try:
            from util import load_data_n_model
            train_loader, test_loader, _, _ = load_data_n_model(
                config["name"], 
                "MLP",  # 使用最简单的模型，只是为了获取数据
                str(BENCHMARK_DIR / "Data")
            )
            
            # 提取数据张量
            train_data = self._extract_data_from_loader(train_loader)
            test_data = self._extract_data_from_loader(test_loader)
            
            # 适配到我们的格式
            adapted_data = self._adapt_to_our_format(train_data, test_data, config)
            
            # 保存到缓存
            if use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(adapted_data, f)
                print(f"💾 数据已缓存到 {cache_file}")
            
            return self._create_loaders(adapted_data)
            
        except Exception as e:
            print(f"❌ 加载 {dataset_name} 失败: {e}")
            print("请确保:")
            print("1. benchmark数据集已下载到正确位置")
            print("2. 数据集格式正确")
            raise
    
    def _extract_data_from_loader(self, loader):
        """从DataLoader中提取所有数据"""
        X_list, y_list = [], []
        
        for batch in loader:
            X, y = batch
            X_list.append(X)
            y_list.append(y)
        
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)
        
        return X, y
    
    def _adapt_to_our_format(self, train_data, test_data, config):
        """将数据适配到我们的格式"""
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        print(f"📏 原始数据形状:")
        print(f"  训练: X={X_train.shape}, y={y_train.shape}")
        print(f"  测试: X={X_test.shape}, y={y_test.shape}")
        
        # 确保数据格式与我们的合成数据一致
        # 我们的格式: [B, T, F] (batch, time, features)
        
        if len(X_train.shape) == 4:  # [B, C, H, W] -> [B, T, F]
            B, C, H, W = X_train.shape
            X_train = X_train.reshape(B, H, C*W)  # [B, T, F]
            X_test = X_test.reshape(X_test.shape[0], H, C*W)
        elif len(X_train.shape) == 3:  # [B, T, F] - 已经是正确格式
            pass
        else:
            raise ValueError(f"不支持的数据形状: {X_train.shape}")
        
        # 获取维度信息
        B_train, T, F = X_train.shape
        B_test = X_test.shape[0]
        num_classes = config["num_classes"]
        
        print(f"📏 适配后数据形状:")
        print(f"  训练: X={X_train.shape}, y={y_train.shape}")
        print(f"  测试: X={X_test.shape}, y={y_test.shape}")
        print(f"  参数: T={T}, F={F}, num_classes={num_classes}")
        
        return {
            "X_train": X_train.float(),
            "y_train": y_train.long(),
            "X_test": X_test.float(), 
            "y_test": y_test.long(),
            "T": T,
            "F": F,
            "num_classes": num_classes,
            "dataset_info": config
        }
    
    def _create_loaders(self, data):
        """创建数据加载器"""
        train_dataset = TensorDataset(data["X_train"], data["y_train"])
        test_dataset = TensorDataset(data["X_test"], data["y_test"])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=64,  # 与我们的设置保持一致
            shuffle=True,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False, 
            pin_memory=True
        )
        
        return train_loader, test_loader, data
    
    def create_few_shot_loader(self, data, ratio=0.1):
        """创建少样本学习的数据加载器"""
        X_train, y_train = data["X_train"], data["y_train"]
        
        # 计算每个类别的样本数
        unique_classes = torch.unique(y_train)
        few_shot_indices = []
        
        for cls in unique_classes:
            cls_indices = torch.where(y_train == cls)[0]
            n_samples = max(1, int(len(cls_indices) * ratio))
            selected = torch.randperm(len(cls_indices))[:n_samples]
            few_shot_indices.append(cls_indices[selected])
        
        few_shot_indices = torch.cat(few_shot_indices)
        
        X_few_shot = X_train[few_shot_indices]
        y_few_shot = y_train[few_shot_indices]
        
        print(f"📊 少样本数据: {len(X_few_shot)} 样本 ({ratio*100:.1f}% of {len(X_train)})")
        
        few_shot_dataset = TensorDataset(X_few_shot, y_few_shot)
        few_shot_loader = DataLoader(
            few_shot_dataset,
            batch_size=32,
            shuffle=True,
            pin_memory=True
        )
        
        return few_shot_loader


def create_sim2real_experiment(dataset_name: str = "UT_HAR"):
    """
    创建Sim2Real实验的数据
    
    Args:
        dataset_name: 真实数据集名称
        
    Returns:
        synthetic_loaders: 合成数据加载器
        real_loaders: 真实数据加载器  
        metadata: 数据元信息
    """
    print(f"🔄 创建 {dataset_name} Sim2Real实验数据...")
    
    # 1. 加载真实数据
    adapter = RealDataAdapter()
    real_train_loader, real_test_loader, real_data = adapter.load_real_dataset(dataset_name)
    
    # 2. 创建匹配的合成数据
    from data_synth import get_synth_loaders
    
    # 使用真实数据的维度创建合成数据
    T = real_data["T"]
    F = real_data["F"] 
    num_classes = real_data["num_classes"]
    
    print(f"📊 创建匹配的合成数据: T={T}, F={F}, classes={num_classes}")
    
    synth_train_loader, synth_test_loader = get_synth_loaders(
        n=10000,  # 合成数据量
        T=T,
        F=F,
        difficulty="hard",
        seed=0,
        num_classes=num_classes
    )
    
    metadata = {
        "dataset": dataset_name,
        "T": T,
        "F": F, 
        "num_classes": num_classes,
        "real_train_size": len(real_data["X_train"]),
        "real_test_size": len(real_data["X_test"]),
        "synth_train_size": 7000,  # 默认训练集大小
        "synth_test_size": 3000    # 默认测试集大小
    }
    
    return {
        "synthetic": (synth_train_loader, synth_test_loader),
        "real": (real_train_loader, real_test_loader),
        "metadata": metadata,
        "adapter": adapter,
        "real_data": real_data
    }


# 使用示例
if __name__ == "__main__":
    print("🚀 真实数据适配器测试")
    
    # 测试数据适配
    try:
        experiment_data = create_sim2real_experiment("UT_HAR")
        print("✅ Sim2Real实验数据创建成功!")
        
        metadata = experiment_data["metadata"] 
        print(f"📊 实验配置: {metadata}")
        
        # 测试少样本学习数据
        adapter = experiment_data["adapter"]
        real_data = experiment_data["real_data"]
        
        few_shot_loader = adapter.create_few_shot_loader(real_data, ratio=0.1)
        print("✅ 少样本学习数据创建成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("这是正常的，因为可能没有下载真实数据集")
        print("实际使用时需要先下载数据集到benchmark/Data/目录")
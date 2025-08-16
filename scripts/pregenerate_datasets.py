#!/usr/bin/env python3
"""
数据集预生成脚本
用于提前生成常用的数据集配置，避免训练时重复生成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_synth import SynthCSIDataset
import argparse
import itertools
from pathlib import Path
import time

def pregenerate_datasets(
    n_values=[2000, 10000, 20000],
    T_values=[64, 128, 256], 
    F_values=[30, 64],
    difficulties=["easy", "mid", "hard"],
    seeds=[0, 1, 2, 3, 4, 5, 6, 7],
    sc_corr_rho_values=[None, 0.3, 0.5, 0.7],
    env_burst_rate_values=[0.0, 0.5, 1.0],
    gain_drift_std_values=[0.0, 0.1, 0.2],
    cache_dir="cache/synth_data"
):
    """
    预生成多种配置的数据集
    """
    
    # 创建配置组合
    configs = []
    
    # 基础配置（最常用的）
    base_configs = list(itertools.product(
        n_values, T_values[:2], F_values[:1], difficulties, seeds[:4]  # 减少组合数
    ))
    
    for n, T, F, difficulty, seed in base_configs:
        configs.append({
            'n': n, 'T': T, 'F': F, 'difficulty': difficulty, 'seed': seed,
            'sc_corr_rho': None, 'env_burst_rate': 0.0, 'gain_drift_std': 0.0
        })
    
    # 扩展配置（带特殊参数的）
    extended_configs = []
    for n, difficulty, seed in itertools.product([20000], ["mid"], [0, 1, 2]):
        # 子载波相关
        for rho in sc_corr_rho_values[1:]:  # 跳过None
            extended_configs.append({
                'n': n, 'T': 128, 'F': 30, 'difficulty': difficulty, 'seed': seed,
                'sc_corr_rho': rho, 'env_burst_rate': 0.0, 'gain_drift_std': 0.0
            })
        
        # 环境突发
        for burst in env_burst_rate_values[1:]:  # 跳过0.0
            extended_configs.append({
                'n': n, 'T': 128, 'F': 30, 'difficulty': difficulty, 'seed': seed,
                'sc_corr_rho': None, 'env_burst_rate': burst, 'gain_drift_std': 0.0
            })
        
        # 增益漂移
        for drift in gain_drift_std_values[1:]:  # 跳过0.0
            extended_configs.append({
                'n': n, 'T': 128, 'F': 30, 'difficulty': difficulty, 'seed': seed,
                'sc_corr_rho': None, 'env_burst_rate': 0.0, 'gain_drift_std': drift
            })
    
    configs.extend(extended_configs)
    
    print(f"[INFO] Total configurations to pregenerate: {len(configs)}")
    print(f"[INFO] Cache directory: {cache_dir}")
    
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    failed_configs = []
    total_time = 0
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Generating: {config}")
        start_time = time.time()
        
        try:
            # 这会自动检查缓存并生成（如果需要）
            dataset = SynthCSIDataset(
                cache_dir=cache_dir,
                **config
            )
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"[OK] Generated in {elapsed:.2f}s (total: {total_time:.1f}s)")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate: {e}")
            failed_configs.append((config, str(e)))
    
    print(f"\n[SUMMARY]")
    print(f"Successfully generated: {len(configs) - len(failed_configs)}/{len(configs)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per config: {total_time/len(configs):.2f}s")
    
    if failed_configs:
        print(f"\nFailed configurations:")
        for config, error in failed_configs:
            print(f"  {config}: {error}")
    
    return len(failed_configs) == 0

def main():
    parser = argparse.ArgumentParser(description="预生成数据集缓存")
    parser.add_argument("--cache-dir", type=str, default="cache/synth_data",
                       help="缓存目录路径")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式：只生成最基础的配置")
    parser.add_argument("--n-samples", type=int, nargs="+", 
                       default=[2000, 10000, 20000],
                       help="样本数列表")
    parser.add_argument("--seeds", type=int, nargs="+",
                       default=[0, 1, 2, 3, 4, 5, 6, 7],
                       help="随机种子列表")
    parser.add_argument("--difficulties", type=str, nargs="+",
                       default=["easy", "mid", "hard"],
                       help="难度级别")
    
    args = parser.parse_args()
    
    if args.quick:
        # 快速模式：只生成最常用的配置
        n_values = [20000]  # 主要用这个
        T_values = [128]
        F_values = [30]
        difficulties = ["mid"]
        seeds = [0, 1, 2]
        sc_corr_rho_values = [None, 0.5]
        env_burst_rate_values = [0.0, 1.0]
        gain_drift_std_values = [0.0, 0.1]
    else:
        # 完整模式
        n_values = args.n_samples
        T_values = [64, 128, 256]
        F_values = [30, 64]
        difficulties = args.difficulties
        seeds = args.seeds
        sc_corr_rho_values = [None, 0.3, 0.5, 0.7]
        env_burst_rate_values = [0.0, 0.5, 1.0]
        gain_drift_std_values = [0.0, 0.1, 0.2]
    
    success = pregenerate_datasets(
        n_values=n_values,
        T_values=T_values,
        F_values=F_values,
        difficulties=difficulties,
        seeds=seeds,
        sc_corr_rho_values=sc_corr_rho_values,
        env_burst_rate_values=env_burst_rate_values,
        gain_drift_std_values=gain_drift_std_values,
        cache_dir=args.cache_dir
    )
    
    if success:
        print(f"\n✅ All datasets pregenerated successfully!")
        print(f"   Cache location: {args.cache_dir}")
        print(f"   Now your training will load data instantly from cache.")
    else:
        print(f"\n❌ Some datasets failed to generate. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
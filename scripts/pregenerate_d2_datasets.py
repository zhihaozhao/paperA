#!/usr/bin/env python3
"""
D2实验专用数据集预生成脚本
根据实际sweep配置预生成所有需要的数据集
"""

import json
import sys
import os
import itertools
import time
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_synth import SynthCSIDataset


def load_sweep_spec(spec_file):
    """加载sweep配置文件"""
    try:
        with open(spec_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {spec_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return None


def generate_param_combinations(spec):
    """生成所有参数组合"""
    fixed = spec.get("fixed", {})
    grid = spec.get("grid", {})
    seeds = spec.get("seeds", [0])
    
    # 生成网格搜索组合
    if not grid:
        # 没有网格参数，只有fixed参数
        grid_combinations = [{}]
    else:
        keys = sorted(grid.keys())
        values = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]
        grid_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    # 每个网格组合 × 每个种子
    all_combinations = []
    for seed in seeds:
        for grid_combo in grid_combinations:
            # 合并fixed参数和grid参数
            params = {**fixed}
            params.update(grid_combo)
            params['seed'] = seed
            all_combinations.append(params)
    
    return all_combinations


def pregenerate_datasets_from_spec(spec_file, cache_dir="cache/synth_data", dry_run=False):
    """根据配置文件预生成所有数据集"""
    
    print(f"🔧 D2实验数据集预生成工具")
    print(f"配置文件: {spec_file}")
    print(f"缓存目录: {cache_dir}")
    print("=" * 60)
    
    # 加载配置
    spec = load_sweep_spec(spec_file)
    if not spec:
        return False
    
    # 生成所有参数组合
    combinations = generate_param_combinations(spec)
    
    print(f"📊 统计信息:")
    print(f"  种子数: {len(spec.get('seeds', [0]))}")
    print(f"  网格参数: {len(spec.get('grid', {}))}")
    print(f"  总组合数: {len(combinations)}")
    print(f"  估计每个耗时: ~2分钟")
    print(f"  预计总耗时: ~{len(combinations) * 2:.0f}分钟 ({len(combinations) * 2 / 60:.1f}小时)")
    print()
    
    if dry_run:
        print("🔍 DRY RUN - 预览前10个组合:")
        for i, params in enumerate(combinations[:10]):
            print(f"  {i+1:3d}: {params}")
        if len(combinations) > 10:
            print(f"  ... 还有 {len(combinations) - 10} 个组合")
        return True
    
    # 确认继续
    response = input(f"确认预生成 {len(combinations)} 个数据集? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 取消预生成")
        return False
    
    # 开始预生成
    start_time = time.time()
    success_count = 0
    failed_combinations = []
    
    for i, params in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] 预生成数据集...")
        
        # 提取SynthCSIDataset需要的参数
        dataset_params = {
            'n': params.get('n_samples', 20000),
            'T': params.get('T', 128),
            'F': params.get('F', 30),
            'difficulty': params.get('difficulty', 'mid'),
            'seed': params.get('seed', 0),
            'sc_corr_rho': params.get('sc_corr_rho', None),
            'env_burst_rate': params.get('env_burst_rate', 0.0),
            'gain_drift_std': params.get('gain_drift_std', 0.0),
            'class_overlap': params.get('class_overlap', 0.0),
            'label_noise_prob': params.get('label_noise_prob', 0.0),
            'num_classes': params.get('num_classes', 8),
            'cache_dir': cache_dir
        }
        
        print(f"  参数: n={dataset_params['n']}, seed={dataset_params['seed']}, "
              f"T={dataset_params['T']}, F={dataset_params['F']}, "
              f"difficulty={dataset_params['difficulty']}")
        
        iter_start = time.time()
        try:
            # 生成数据集（会自动处理缓存）
            dataset = SynthCSIDataset(**dataset_params)
            iter_time = time.time() - iter_start
            success_count += 1
            
            status = "生成+缓存" if iter_time > 5 else "缓存命中"
            print(f"  ✅ 完成 ({iter_time:.1f}秒, {status})")
            
        except Exception as e:
            iter_time = time.time() - iter_start
            print(f"  ❌ 失败: {e} ({iter_time:.1f}秒)")
            failed_combinations.append((i, params, str(e)))
    
    # 统计结果
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎯 预生成完成!")
    print(f"  成功: {success_count}/{len(combinations)}")
    print(f"  失败: {len(failed_combinations)}")
    print(f"  总耗时: {total_time/60:.1f}分钟")
    print(f"  平均每个: {total_time/len(combinations):.1f}秒")
    
    if failed_combinations:
        print(f"\n❌ 失败的组合:")
        for idx, params, error in failed_combinations[:5]:
            print(f"  {idx}: {params}")
            print(f"      错误: {error}")
        if len(failed_combinations) > 5:
            print(f"  ... 还有 {len(failed_combinations) - 5} 个失败")
    
    # 检查缓存目录
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
        total_size_mb = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files) / (1024*1024)
        print(f"\n📁 缓存统计:")
        print(f"  缓存文件数: {len(cache_files)}")
        print(f"  总大小: {total_size_mb:.1f}MB")
        print(f"  位置: {cache_dir}")
    
    return len(failed_combinations) == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="D2实验数据集预生成工具")
    parser.add_argument("--spec", type=str, default="scripts/d2_spec.json",
                       help="Sweep配置文件路径")
    parser.add_argument("--cache-dir", type=str, default="cache/synth_data",
                       help="缓存目录")
    parser.add_argument("--dry-run", action="store_true",
                       help="只预览，不实际生成")
    
    args = parser.parse_args()
    
    success = pregenerate_datasets_from_spec(
        spec_file=args.spec,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run
    )
    
    if not success:
        sys.exit(1)
    
    print(f"\n✅ 现在运行您的sweep实验将享受秒级数据加载！")
    print(f"命令: python scripts/run_sweep_from_json.py --spec {args.spec} --resume")


if __name__ == "__main__":
    main()
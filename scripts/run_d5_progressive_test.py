#!/usr/bin/env python3
"""
D5渐进式难度测试脚本
目标：找到Enhanced模型优势最明显的难度设置
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse

def run_experiment(model, config, seed=0, epochs=20):
    """运行单个实验"""
    
    cmd = [
        "python", "-m", "src.train_eval",
        "--model", model,
        "--difficulty", "hard",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--class_overlap", str(config["class_overlap"]),
        "--label_noise_prob", str(config["label_noise_prob"]),
        "--env_burst_rate", str(config["env_burst_rate"]),
        "--gain_drift_std", str(config["gain_drift_std"]),
        "--sc_corr_rho", str(config["sc_corr_rho"]),
        "--out", f"results_gpu/d5_progressive/{model}_level{config['level']}_s{seed}.json"
    ]
    
    print(f"[RUN] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        if result.returncode == 0:
            print(f"[OK] {model} level{config['level']} completed")
            return True
        else:
            print(f"[FAIL] {model} level{config['level']}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {model} level{config['level']}")
        return False
    except Exception as e:
        print(f"[ERROR] {model} level{config['level']}: {e}")
        return False

def load_results(configs):
    """加载实验结果"""
    results = []
    
    for config in configs:
        level = config['level']
        
        # 尝试加载Enhanced结果
        enhanced_file = f"results_gpu/d5_progressive/enhanced_level{level}_s0.json"
        cnn_file = f"results_gpu/d5_progressive/cnn_level{level}_s0.json"
        
        if Path(enhanced_file).exists() and Path(cnn_file).exists():
            try:
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f)
                with open(cnn_file, 'r') as f:
                    cnn_data = json.load(f)
                
                enhanced_f1 = enhanced_data['metrics']['macro_f1']
                cnn_f1 = cnn_data['metrics']['macro_f1']
                gap = enhanced_f1 - cnn_f1
                
                enhanced_ece = enhanced_data['metrics']['ece_cal']
                cnn_ece = cnn_data['metrics']['ece_cal']
                ece_improvement = cnn_ece - enhanced_ece
                
                results.append({
                    'level': level,
                    'config': config,
                    'enhanced_f1': enhanced_f1,
                    'cnn_f1': cnn_f1,
                    'performance_gap': gap,
                    'enhanced_ece': enhanced_ece,
                    'cnn_ece': cnn_ece,
                    'ece_improvement': ece_improvement,
                    'enhanced_temp': enhanced_data['metrics']['temperature'],
                    'cnn_temp': cnn_data['metrics']['temperature']
                })
                
            except Exception as e:
                print(f"[ERROR] Loading level {level}: {e}")
        else:
            print(f"[MISSING] Level {level} results not found")
    
    return results

def analyze_results(results):
    """分析结果并找到最佳配置"""
    if not results:
        print("[ERROR] No results to analyze")
        return None
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("D5渐进式难度测试结果")
    print("="*80)
    
    # 显示详细结果
    for _, row in df.iterrows():
        print(f"\nLevel {row['level']}:")
        print(f"  配置: overlap={row['config']['class_overlap']}, noise={row['config']['label_noise_prob']}, burst={row['config']['env_burst_rate']}, drift={row['config']['gain_drift_std']}, corr={row['config']['sc_corr_rho']}")
        print(f"  Enhanced F1: {row['enhanced_f1']:.3f}")
        print(f"  CNN F1: {row['cnn_f1']:.3f}")
        print(f"  性能差异: {row['performance_gap']:.3f}")
        print(f"  Enhanced ECE: {row['enhanced_ece']:.3f}")
        print(f"  CNN ECE: {row['cnn_ece']:.3f}")
        print(f"  ECE改善: {row['ece_improvement']:.3f}")
    
    # 找到最佳配置
    best_gap_idx = df['performance_gap'].idxmax()
    best_level = df.loc[best_gap_idx]
    
    print("\n" + "="*80)
    print("最佳配置分析")
    print("="*80)
    print(f"最佳难度级别: Level {best_level['level']}")
    print(f"最大性能差异: {best_level['performance_gap']:.3f}")
    print(f"Enhanced F1: {best_level['enhanced_f1']:.3f}")
    print(f"CNN F1: {best_level['cnn_f1']:.3f}")
    print(f"ECE改善: {best_level['ece_improvement']:.3f}")
    
    # 保存结果
    output_file = "results_gpu/d5_progressive/progressive_test_summary.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_config": {
            "level": int(best_level['level']),
            "config": best_level['config'],
            "performance_gap": float(best_level['performance_gap']),
            "enhanced_f1": float(best_level['enhanced_f1']),
            "cnn_f1": float(best_level['cnn_f1']),
            "ece_improvement": float(best_level['ece_improvement'])
        },
        "all_results": df.to_dict('records')
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return best_level, df

def main():
    parser = argparse.ArgumentParser(description="D5渐进式难度测试")
    parser.add_argument("--run", action="store_true", help="运行实验")
    parser.add_argument("--analyze", action="store_true", help="分析结果")
    parser.add_argument("--quick", action="store_true", help="快速测试(只运行level 1和2)")
    args = parser.parse_args()
    
    # 定义难度配置
    difficulty_configs = [
        {
            "level": 1,
            "class_overlap": 0.8,
            "label_noise_prob": 0.1,
            "env_burst_rate": 0.2,
            "gain_drift_std": 0.6,
            "sc_corr_rho": 0.5
        },
        {
            "level": 2,
            "class_overlap": 0.85,
            "label_noise_prob": 0.15,
            "env_burst_rate": 0.3,
            "gain_drift_std": 0.8,
            "sc_corr_rho": 0.7
        },
        {
            "level": 3,
            "class_overlap": 0.9,
            "label_noise_prob": 0.2,
            "env_burst_rate": 0.4,
            "gain_drift_std": 1.0,
            "sc_corr_rho": 0.8
        },
        {
            "level": 4,
            "class_overlap": 0.95,
            "label_noise_prob": 0.25,
            "env_burst_rate": 0.5,
            "gain_drift_std": 1.2,
            "sc_corr_rho": 0.9
        }
    ]
    
    if args.quick:
        difficulty_configs = difficulty_configs[:2]
        print("[QUICK] 快速测试模式：只运行Level 1和2")
    
    if args.run:
        print("开始D5渐进式难度测试...")
        
        # 创建输出目录
        Path("results_gpu/d5_progressive").mkdir(parents=True, exist_ok=True)
        
        # 运行实验
        for config in difficulty_configs:
            print(f"\n{'='*60}")
            print(f"运行 Level {config['level']}")
            print(f"{'='*60}")
            
            # 运行Enhanced
            enhanced_success = run_experiment("enhanced", config, seed=0, epochs=20)
            if enhanced_success:
                time.sleep(5)  # 等待GPU冷却
            
            # 运行CNN
            cnn_success = run_experiment("cnn", config, seed=0, epochs=20)
            if cnn_success:
                time.sleep(5)  # 等待GPU冷却
            
            if not (enhanced_success and cnn_success):
                print(f"[WARNING] Level {config['level']} 部分失败")
    
    if args.analyze or not args.run:
        print("分析实验结果...")
        results = load_results(difficulty_configs)
        if results:
            best_config, df = analyze_results(results)
        else:
            print("[ERROR] 没有找到实验结果，请先运行实验")

if __name__ == "__main__":
    main()


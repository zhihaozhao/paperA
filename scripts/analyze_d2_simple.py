#!/usr/bin/env python3
"""
D2 Experiment Results Analysis Script (Basic Version)
Analyzes 540 D2 experiment results using only standard libraries
"""

import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_d2_results():
    """Analyze all D2 experiment results"""
    results_dir = Path("results_gpu/d2")
    result_files = list(results_dir.glob("paperA_*.json"))
    
    print(f"[INFO] Found {len(result_files)} D2 result files")
    
    # Load all results
    results = {}
    failed_files = []
    
    for file_path in result_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Extract experiment identifiers
            filename = file_path.stem
            # paperA_enhanced_hard_s0_cla0p0_env0p0_lab0p0
            parts = filename.split('_')
            model = parts[1]
            difficulty = parts[2]
            seed = int(parts[3][1:])  # s0 -> 0
            
            # Extract parameter values
            cla_overlap = float(parts[4][3:].replace('p', '.'))  # cla0p4 -> 0.4
            env_burst = float(parts[5][3:].replace('p', '.'))   # env0p1 -> 0.1
            lab_noise = float(parts[6][3:].replace('p', '.'))   # lab0p05 -> 0.05
            
            key = f"{model}_{difficulty}_s{seed}_cla{cla_overlap}_env{env_burst}_lab{lab_noise}"
            
            results[key] = {
                'filename': filename,
                'model': model,
                'difficulty': difficulty,
                'seed': seed,
                'class_overlap': cla_overlap,
                'env_burst_rate': env_burst,
                'label_noise_prob': lab_noise,
                'metrics': data.get('metrics', {}),
                'meta': data.get('meta', {}),
                'args': data.get('args', {})
            }
            
        except Exception as e:
            failed_files.append((file_path, str(e)))
    
    if failed_files:
        print(f"[WARNING] Failed to load {len(failed_files)} files")
    
    print(f"[INFO] Successfully loaded {len(results)} experiments")
    
    # Group results by model
    by_model = defaultdict(list)
    all_metrics = []
    
    for key, result in results.items():
        model = result['model']
        metrics = result['metrics']
        by_model[model].append(metrics)
        all_metrics.append(metrics)
    
    # Compute model statistics
    print("\n" + "="*60)
    print("D2 EXPERIMENT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total experiments: {len(results)}")
    print(f"Models: {list(by_model.keys())}")
    print(f"Experiments per model: {len(all_metrics) // len(by_model) if by_model else 0}")
    
    print("\nModel Performance Comparison:")
    print(f"{'Model':<15} {'Macro F1':<15} {'Falling F1':<15} {'ECE Raw':<15} {'ECE Cal':<15}")
    print("-" * 75)
    
    model_stats = {}
    for model, metrics_list in by_model.items():
        # Extract metric arrays
        macro_f1s = [m.get('macro_f1', 0) for m in metrics_list if m.get('macro_f1') is not None]
        falling_f1s = [m.get('falling_f1', 0) for m in metrics_list if m.get('falling_f1') is not None]
        ece_raws = [m.get('ece_raw', 0) for m in metrics_list if m.get('ece_raw') is not None]
        ece_cals = [m.get('ece_cal', 0) for m in metrics_list if m.get('ece_cal') is not None]
        
        # Compute statistics
        macro_f1_mean = np.mean(macro_f1s) if macro_f1s else 0
        macro_f1_std = np.std(macro_f1s) if macro_f1s else 0
        
        falling_f1_mean = np.mean(falling_f1s) if falling_f1s else 0
        falling_f1_std = np.std(falling_f1s) if falling_f1s else 0
        
        ece_raw_mean = np.mean(ece_raws) if ece_raws else 0
        ece_raw_std = np.std(ece_raws) if ece_raws else 0
        
        ece_cal_mean = np.mean(ece_cals) if ece_cals else 0
        ece_cal_std = np.std(ece_cals) if ece_cals else 0
        
        model_stats[model] = {
            'macro_f1': {'mean': macro_f1_mean, 'std': macro_f1_std, 'count': len(macro_f1s)},
            'falling_f1': {'mean': falling_f1_mean, 'std': falling_f1_std, 'count': len(falling_f1s)},
            'ece_raw': {'mean': ece_raw_mean, 'std': ece_raw_std, 'count': len(ece_raws)},
            'ece_cal': {'mean': ece_cal_mean, 'std': ece_cal_std, 'count': len(ece_cals)}
        }
        
        print(f"{model.capitalize():<15} {macro_f1_mean:.3f}±{macro_f1_std:.3f}    {falling_f1_mean:.3f}±{falling_f1_std:.3f}     {ece_raw_mean:.3f}±{ece_raw_std:.3f}     {ece_cal_mean:.3f}±{ece_cal_std:.3f}")
    
    # Overall statistics
    all_macro_f1s = [m.get('macro_f1', 0) for m in all_metrics if m.get('macro_f1') is not None]
    all_falling_f1s = [m.get('falling_f1', 0) for m in all_metrics if m.get('falling_f1') is not None]
    all_ece_raws = [m.get('ece_raw', 0) for m in all_metrics if m.get('ece_raw') is not None]
    all_ece_cals = [m.get('ece_cal', 0) for m in all_metrics if m.get('ece_cal') is not None]
    
    print("\nOverall Statistics:")
    print(f"  Macro F1: {np.mean(all_macro_f1s):.3f}±{np.std(all_macro_f1s):.3f} (n={len(all_macro_f1s)})")
    print(f"  Falling F1: {np.mean(all_falling_f1s):.3f}±{np.std(all_falling_f1s):.3f} (n={len(all_falling_f1s)})")
    print(f"  ECE Raw: {np.mean(all_ece_raws):.3f}±{np.std(all_ece_raws):.3f} (n={len(all_ece_raws)})")
    print(f"  ECE Calibrated: {np.mean(all_ece_cals):.3f}±{np.std(all_ece_cals):.3f} (n={len(all_ece_cals)})")
    
    # Calibration improvement
    improvements = []
    for metrics in all_metrics:
        ece_raw = metrics.get('ece_raw', 0)
        ece_cal = metrics.get('ece_cal', 0)
        if ece_raw > 0:
            improvement = (ece_raw - ece_cal) / ece_raw
            improvements.append(improvement)
    
    if improvements:
        print(f"\nCalibration Improvement:")
        print(f"  Mean ECE reduction: {np.mean(improvements):.1%}")
        print(f"  Median ECE reduction: {np.median(improvements):.1%}")
        print(f"  Max ECE reduction: {np.max(improvements):.1%}")
    
    # Save summary data for paper
    paper_data = {
        'total_experiments': len(results),
        'models_evaluated': len(by_model),
        'parameter_combinations': len(set(f"{r['class_overlap']}_{r['env_burst_rate']}_{r['label_noise_prob']}" for r in results.values())),
        'overall_performance': {
            'macro_f1_mean': float(np.mean(all_macro_f1s)),
            'macro_f1_std': float(np.std(all_macro_f1s)),
            'falling_f1_mean': float(np.mean(all_falling_f1s)),
            'falling_f1_std': float(np.std(all_falling_f1s)),
            'ece_raw_mean': float(np.mean(all_ece_raws)),
            'ece_cal_mean': float(np.mean(all_ece_cals)),
            'calibration_improvement_mean': float(np.mean(improvements)) if improvements else 0
        },
        'model_comparison': model_stats
    }
    
    # Save to JSON for paper integration
    with open('results/d2_paper_stats.json', 'w') as f:
        json.dump(paper_data, f, indent=2)
    
    print(f"\n[SUCCESS] Analysis completed! Paper statistics saved to results/d2_paper_stats.json")
    
    return paper_data, model_stats

if __name__ == "__main__":
    analyze_d2_results()
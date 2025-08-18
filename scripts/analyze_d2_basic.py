#!/usr/bin/env python3
"""
D2 Experiment Results Analysis Script (Standard Library Only)
Analyzes 540 D2 experiment results using only Python standard library
"""

import json
import glob
import math
from pathlib import Path
from collections import defaultdict

def mean(values):
    """Calculate mean of a list"""
    return sum(values) / len(values) if values else 0

def std(values):
    """Calculate standard deviation of a list"""
    if not values or len(values) < 2:
        return 0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def median(values):
    """Calculate median of a list"""
    if not values:
        return 0
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]

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
            if len(parts) >= 7:
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
        for file_path, error in failed_files[:3]:  # Show first 3 errors
            print(f"  {file_path.name}: {error}")
    
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
    
    print("\nModel Performance Comparison:")
    print(f"{'Model':<15} {'Macro F1':<15} {'Falling F1':<15} {'ECE Raw':<15} {'ECE Cal':<15}")
    print("-" * 75)
    
    model_stats = {}
    for model in ['enhanced', 'cnn', 'bilstm', 'conformer_lite']:
        if model in by_model:
            metrics_list = by_model[model]
            
            # Extract metric arrays
            macro_f1s = [m.get('macro_f1', 0) for m in metrics_list if m.get('macro_f1') is not None]
            falling_f1s = [m.get('falling_f1', 0) for m in metrics_list if m.get('falling_f1') is not None]
            ece_raws = [m.get('ece_raw', 0) for m in metrics_list if m.get('ece_raw') is not None]
            ece_cals = [m.get('ece_cal', 0) for m in metrics_list if m.get('ece_cal') is not None]
            
            # Compute statistics
            macro_f1_mean = mean(macro_f1s)
            macro_f1_std = std(macro_f1s)
            
            falling_f1_mean = mean(falling_f1s)
            falling_f1_std = std(falling_f1s)
            
            ece_raw_mean = mean(ece_raws)
            ece_raw_std = std(ece_raws)
            
            ece_cal_mean = mean(ece_cals)
            ece_cal_std = std(ece_cals)
            
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
    print(f"  Macro F1: {mean(all_macro_f1s):.3f}±{std(all_macro_f1s):.3f} (n={len(all_macro_f1s)})")
    print(f"  Falling F1: {mean(all_falling_f1s):.3f}±{std(all_falling_f1s):.3f} (n={len(all_falling_f1s)})")
    print(f"  ECE Raw: {mean(all_ece_raws):.3f}±{std(all_ece_raws):.3f} (n={len(all_ece_raws)})")
    print(f"  ECE Calibrated: {mean(all_ece_cals):.3f}±{std(all_ece_cals):.3f} (n={len(all_ece_cals)})")
    
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
        print(f"  Mean ECE reduction: {mean(improvements):.1%}")
        print(f"  Median ECE reduction: {median(improvements):.1%}")
        print(f"  Max ECE reduction: {max(improvements):.1%}")
    
    # Enhanced vs baselines comparison
    if 'enhanced' in model_stats:
        enhanced_f1 = model_stats['enhanced']['falling_f1']['mean']
        baseline_f1s = []
        for baseline in ['cnn', 'bilstm', 'conformer_lite']:
            if baseline in model_stats:
                baseline_f1s.append(model_stats[baseline]['falling_f1']['mean'])
        
        if baseline_f1s:
            avg_baseline_f1 = mean(baseline_f1s)
            advantage = enhanced_f1 - avg_baseline_f1
            print(f"\nEnhanced Model Advantage:")
            print(f"  Enhanced Falling F1: {enhanced_f1:.3f}")
            print(f"  Average Baseline F1: {avg_baseline_f1:.3f}")
            print(f"  Advantage: +{advantage:.3f} ({advantage/avg_baseline_f1:.1%})")
    
    # Parameter sensitivity analysis
    param_combinations = set()
    for result in results.values():
        combo = f"{result['class_overlap']}_{result['env_burst_rate']}_{result['label_noise_prob']}"
        param_combinations.add(combo)
    
    print(f"\nParameter Sensitivity:")
    print(f"  Unique parameter combinations: {len(param_combinations)}")
    print(f"  Expected total: 3×3×3 = 27 combinations")
    
    # Save summary data for paper
    paper_data = {
        'total_experiments': len(results),
        'models_evaluated': len(by_model),
        'parameter_combinations': len(param_combinations),
        'overall_performance': {
            'macro_f1_mean': mean(all_macro_f1s),
            'macro_f1_std': std(all_macro_f1s),
            'falling_f1_mean': mean(all_falling_f1s),
            'falling_f1_std': std(all_falling_f1s),
            'ece_raw_mean': mean(all_ece_raws),
            'ece_cal_mean': mean(all_ece_cals),
            'calibration_improvement_mean': mean(improvements) if improvements else 0
        },
        'model_comparison': model_stats
    }
    
    # Create results directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)
    
    # Save to JSON for paper integration
    with open('results/d2_paper_stats.json', 'w') as f:
        json.dump(paper_data, f, indent=2)
    
    print(f"\n[SUCCESS] Analysis completed! Paper statistics saved to results/d2_paper_stats.json")
    
    return paper_data, model_stats

if __name__ == "__main__":
    analyze_d2_results()
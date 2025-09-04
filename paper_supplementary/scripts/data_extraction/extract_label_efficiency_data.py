#!/usr/bin/env python3
"""
Extract Label Efficiency (Sim2Real) data for Figure 5.
Data source: /workspace/results_gpu/d4/sim2real/*.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_label_efficiency():
    """
    Extract label efficiency performance from Sim2Real experiments.
    Shows how performance improves with different amounts of labeled data.
    """
    
    d4_dir = Path("/workspace/results_gpu/d4/sim2real")
    
    if not d4_dir.exists():
        raise FileNotFoundError(f"Sim2Real data directory not found: {d4_dir}")
    
    label_efficiency_results = defaultdict(lambda: defaultdict(dict))
    
    # Process Sim2Real files
    json_files = list(d4_dir.glob("*.json"))
    print(f"  Processing {len(json_files)} Sim2Real experiment files...")
    
    for json_file in json_files:
        if 'summary' in json_file.name:
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            model = data.get('model', 'unknown')
            label_ratio = data.get('label_ratio', 0)
            transfer_method = data.get('transfer_method', 'unknown')
            
            # Convert label ratio to percentage
            label_percent = label_ratio * 100
            
            # Extract performance metrics
            zero_shot_f1 = data.get('zero_shot_metrics', {}).get('macro_f1', 0)
            target_f1 = data.get('target_metrics', {}).get('macro_f1', 0)
            
            # Store results
            if model not in label_efficiency_results:
                label_efficiency_results[model] = {}
            
            if label_percent not in label_efficiency_results[model]:
                label_efficiency_results[model][label_percent] = {
                    'zero_shot': [],
                    'fine_tuned': [],
                    'transfer_methods': []
                }
            
            label_efficiency_results[model][label_percent]['zero_shot'].append(zero_shot_f1)
            label_efficiency_results[model][label_percent]['fine_tuned'].append(target_f1)
            label_efficiency_results[model][label_percent]['transfer_methods'].append(transfer_method)
            
            # Also extract fall-specific metrics if available
            if 'zero_shot_metrics' in data:
                for key in ['epileptic_fall_f1', 'elderly_fall_f1', 'falling_f1']:
                    if key in data['zero_shot_metrics']:
                        if f'zero_shot_{key}' not in label_efficiency_results[model][label_percent]:
                            label_efficiency_results[model][label_percent][f'zero_shot_{key}'] = []
                        label_efficiency_results[model][label_percent][f'zero_shot_{key}'].append(
                            data['zero_shot_metrics'][key])
            
            if 'target_metrics' in data:
                for key in ['epileptic_fall_f1', 'elderly_fall_f1', 'falling_f1']:
                    if key in data['target_metrics']:
                        if f'target_{key}' not in label_efficiency_results[model][label_percent]:
                            label_efficiency_results[model][label_percent][f'target_{key}'] = []
                        label_efficiency_results[model][label_percent][f'target_{key}'].append(
                            data['target_metrics'][key])
                        
        except Exception as e:
            print(f"  Warning: Failed to process {json_file.name}: {e}")
    
    # Calculate summary statistics
    summary = {}
    for model, label_data in label_efficiency_results.items():
        summary[model] = {}
        for label_percent, metrics in label_data.items():
            summary[model][label_percent] = {}
            
            # Calculate means for zero-shot and fine-tuned
            if metrics['zero_shot']:
                summary[model][label_percent]['zero_shot'] = {
                    'mean': np.mean(metrics['zero_shot']),
                    'std': np.std(metrics['zero_shot']),
                    'n': len(metrics['zero_shot'])
                }
            
            if metrics['fine_tuned']:
                summary[model][label_percent]['fine_tuned'] = {
                    'mean': np.mean(metrics['fine_tuned']),
                    'std': np.std(metrics['fine_tuned']),
                    'n': len(metrics['fine_tuned'])
                }
            
            # Transfer methods used
            summary[model][label_percent]['transfer_methods'] = list(set(metrics['transfer_methods']))
    
    # Format for Figure 5
    formatted_results = {
        'figure_data': {},
        'table_data': {}
    }
    
    # Focus on 'enhanced' model (PASE-Net)
    if 'enhanced' in summary:
        label_ratios = sorted(summary['enhanced'].keys())
        
        formatted_results['figure_data'] = {
            'label_percentages': label_ratios,
            'zero_shot': [],
            'fine_tuned': [],
            'zero_shot_std': [],
            'fine_tuned_std': []
        }
        
        for ratio in label_ratios:
            if ratio in summary['enhanced']:
                zero_shot_data = summary['enhanced'][ratio].get('zero_shot', {})
                fine_tuned_data = summary['enhanced'][ratio].get('fine_tuned', {})
                
                formatted_results['figure_data']['zero_shot'].append(
                    zero_shot_data.get('mean', 0) * 100)
                formatted_results['figure_data']['fine_tuned'].append(
                    fine_tuned_data.get('mean', 0) * 100)
                formatted_results['figure_data']['zero_shot_std'].append(
                    zero_shot_data.get('std', 0) * 100)
                formatted_results['figure_data']['fine_tuned_std'].append(
                    fine_tuned_data.get('std', 0) * 100)
        
        # Create table data
        formatted_results['table_data'] = {
            'Label Ratio': label_ratios,
            'Zero-Shot F1 (%)': formatted_results['figure_data']['zero_shot'],
            'Fine-Tuned F1 (%)': formatted_results['figure_data']['fine_tuned']
        }
    
    # Print summary
    print("\n  Label Efficiency Summary (PASE-Net):")
    print("  " + "-"*50)
    print("  Label%   Zero-Shot   Fine-Tuned   Improvement")
    print("  " + "-"*50)
    if 'enhanced' in summary:
        for ratio in sorted(summary['enhanced'].keys()):
            zs = summary['enhanced'][ratio].get('zero_shot', {}).get('mean', 0) * 100
            ft = summary['enhanced'][ratio].get('fine_tuned', {}).get('mean', 0) * 100
            imp = ft - zs
            print(f"  {ratio:5.1f}%   {zs:8.1f}%   {ft:9.1f}%   {imp:+8.1f}%")
    
    return {
        'raw_results': dict(label_efficiency_results),
        'summary': summary,
        'formatted': formatted_results,
        'source': str(d4_dir),
        'num_files_processed': len(json_files)
    }

if __name__ == "__main__":
    # Test extraction
    result = extract_label_efficiency()
    print(json.dumps(result['formatted'], indent=2))
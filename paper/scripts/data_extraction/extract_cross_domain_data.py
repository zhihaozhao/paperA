#!/usr/bin/env python3
"""
Extract Cross-Domain (LOSO/LORO) performance data for Figure 3 and Table 1.
Data source: /workspace/results_gpu/d3/loso/*.json and /workspace/results_gpu/d3/loro/*.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_cross_domain_performance():
    """
    Extract LOSO and LORO performance from experimental results.
    This data is from REAL WiFi-CSI-Sensing-Benchmark dataset.
    """
    
    loso_dir = Path("/workspace/results_gpu/d3/loso")
    loro_dir = Path("/workspace/results_gpu/d3/loro")
    
    if not loso_dir.exists() or not loro_dir.exists():
        raise FileNotFoundError(f"Cross-domain data directories not found")
    
    results = {
        'LOSO': defaultdict(list),
        'LORO': defaultdict(list),
        'detailed_metrics': {},
        'summary': {}
    }
    
    # Process LOSO results
    print(f"  Processing LOSO experiments from {loso_dir}")
    loso_files = list(loso_dir.glob("loso_*.json"))
    
    for json_file in loso_files:
        if 'summary' in json_file.name:
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            model = json_file.stem.split('_')[1]  # e.g., loso_enhanced_seed0.json
            
            # Extract performance metrics
            if 'aggregate_stats' in data and 'macro_f1' in data['aggregate_stats']:
                f1_score = data['aggregate_stats']['macro_f1'].get('mean', 0)
                results['LOSO'][model].append(f1_score)
                
                # Also extract fall-specific metrics
                if model not in results['detailed_metrics']:
                    results['detailed_metrics'][model] = {}
                
                if 'epileptic_fall_f1' in data['aggregate_stats']:
                    if 'fall_metrics' not in results['detailed_metrics'][model]:
                        results['detailed_metrics'][model]['fall_metrics'] = {}
                    
                    results['detailed_metrics'][model]['fall_metrics']['epileptic'] = \
                        data['aggregate_stats']['epileptic_fall_f1'].get('mean', 0)
                    results['detailed_metrics'][model]['fall_metrics']['elderly'] = \
                        data['aggregate_stats'].get('elderly_fall_f1', {}).get('mean', 0)
                    results['detailed_metrics'][model]['fall_metrics']['cantgetup'] = \
                        data['aggregate_stats'].get('fall_cantgetup_f1', {}).get('mean', 0)
                    results['detailed_metrics'][model]['fall_metrics']['overall'] = \
                        data['aggregate_stats'].get('falling_f1', {}).get('mean', 0)
                        
        except Exception as e:
            print(f"  Warning: Failed to process {json_file.name}: {e}")
    
    # Process LORO results
    print(f"  Processing LORO experiments from {loro_dir}")
    loro_files = list(loro_dir.glob("loro_*.json"))
    
    for json_file in loro_files:
        if 'summary' in json_file.name:
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model = json_file.stem.split('_')[1]
            
            if 'aggregate_stats' in data and 'macro_f1' in data['aggregate_stats']:
                f1_score = data['aggregate_stats']['macro_f1'].get('mean', 0)
                results['LORO'][model].append(f1_score)
                
        except Exception as e:
            print(f"  Warning: Failed to process {json_file.name}: {e}")
    
    # Calculate summary statistics
    for protocol in ['LOSO', 'LORO']:
        results['summary'][protocol] = {}
        for model, scores in results[protocol].items():
            if scores:
                results['summary'][protocol][model] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'n': len(scores),
                    'mean_percent': np.mean(scores) * 100,
                    'std_percent': np.std(scores) * 100
                }
    
    # Create formatted output for paper
    formatted_results = {
        'table_data': {},
        'figure_data': {}
    }
    
    # Format for Table 1
    model_mapping = {
        'enhanced': 'PASE-Net',
        'cnn': 'CNN',
        'bilstm': 'BiLSTM',
        'conformer': 'Conformer'
    }
    
    for model_key, model_name in model_mapping.items():
        formatted_results['table_data'][model_name] = {
            'LOSO': results['summary']['LOSO'].get(model_key, {}).get('mean_percent', 'N/A'),
            'LOSO_std': results['summary']['LOSO'].get(model_key, {}).get('std_percent', 'N/A'),
            'LORO': results['summary']['LORO'].get(model_key, {}).get('mean_percent', 'N/A'),
            'LORO_std': results['summary']['LORO'].get(model_key, {}).get('std_percent', 'N/A')
        }
    
    # Format for Figure 3
    for protocol in ['LOSO', 'LORO']:
        formatted_results['figure_data'][protocol] = {}
        for model_key, model_name in model_mapping.items():
            if model_key in results['summary'][protocol]:
                formatted_results['figure_data'][protocol][model_name] = \
                    results['summary'][protocol][model_key]['mean_percent']
    
    # Print summary
    print("\n  Cross-Domain Performance Summary:")
    print("  " + "-"*50)
    print("  Model       LOSO (%)    LORO (%)")
    print("  " + "-"*50)
    for model_name in ['PASE-Net', 'CNN', 'BiLSTM', 'Conformer']:
        loso = formatted_results['table_data'][model_name]['LOSO']
        loro = formatted_results['table_data'][model_name]['LORO']
        if isinstance(loso, float) and isinstance(loro, float):
            print(f"  {model_name:10s}  {loso:5.1f}      {loro:5.1f}")
    
    return {
        'raw_results': dict(results),
        'summary': results['summary'],
        'formatted': formatted_results,
        'detailed_metrics': results['detailed_metrics'],
        'source': {
            'loso': str(loso_dir),
            'loro': str(loro_dir)
        },
        'num_files_processed': {
            'loso': len(loso_files),
            'loro': len(loro_files)
        }
    }

if __name__ == "__main__":
    # Test extraction
    result = extract_cross_domain_performance()
    print(json.dumps(result['formatted'], indent=2))
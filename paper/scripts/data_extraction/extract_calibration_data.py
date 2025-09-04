#!/usr/bin/env python3
"""
Extract Calibration performance data for Figure 4 and Table 1.
Data source: /workspace/results_gpu/d6/*.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_calibration_metrics():
    """
    Extract calibration metrics (ECE, temperature, etc.) from experimental results.
    """
    
    d6_dir = Path("/workspace/results_gpu/d6")
    
    if not d6_dir.exists():
        # Try alternative location
        d6_dir = Path("/workspace/results/")
        if not d6_dir.exists():
            raise FileNotFoundError(f"Calibration data directory not found")
    
    calibration_results = defaultdict(dict)
    
    # Process calibration files
    json_files = list(d6_dir.glob("paperA_*_hard_*.json"))
    print(f"  Processing {len(json_files)} calibration experiment files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            parts = json_file.stem.split('_')
            if len(parts) >= 2:
                model = parts[1]  # e.g., paperA_enhanced_hard_s0.json
                
                # Extract calibration metrics
                if 'metrics' in data:
                    metrics = data['metrics']
                    
                    if model not in calibration_results:
                        calibration_results[model] = {
                            'ece_raw': [],
                            'ece_cal': [],
                            'temperature': [],
                            'brier': [],
                            'nll_raw': [],
                            'nll_cal': []
                        }
                    
                    # Collect metrics
                    calibration_results[model]['ece_raw'].append(
                        metrics.get('ece_raw', 0))
                    calibration_results[model]['ece_cal'].append(
                        metrics.get('ece_cal', 0))
                    calibration_results[model]['temperature'].append(
                        metrics.get('temperature', 1.0))
                    calibration_results[model]['brier'].append(
                        metrics.get('brier', 0))
                    calibration_results[model]['nll_raw'].append(
                        metrics.get('nll_raw', 0))
                    calibration_results[model]['nll_cal'].append(
                        metrics.get('nll_cal', 0))
                        
        except Exception as e:
            print(f"  Warning: Failed to process {json_file.name}: {e}")
    
    # Calculate summary statistics
    summary = {}
    model_mapping = {
        'enhanced': 'PASE-Net',
        'cnn': 'CNN',
        'bilstm': 'BiLSTM',
        'conformer_lite': 'Conformer',
        'conformer': 'Conformer'
    }
    
    for model_key, metrics_lists in calibration_results.items():
        model_name = model_mapping.get(model_key, model_key.upper())
        summary[model_name] = {}
        
        for metric_name, values in metrics_lists.items():
            if values:
                summary[model_name][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n': len(values),
                    'values': values
                }
    
    # Format for paper
    formatted_results = {
        'table_data': {},
        'figure_data': {
            'before_calibration': {},
            'after_calibration': {},
            'temperature': {}
        }
    }
    
    for model_name, metrics in summary.items():
        if metrics:
            formatted_results['table_data'][model_name] = {
                'ECE_raw': metrics.get('ece_raw', {}).get('mean', 'N/A'),
                'ECE_cal': metrics.get('ece_cal', {}).get('mean', 'N/A'),
                'Temperature': metrics.get('temperature', {}).get('mean', 'N/A'),
                'Brier': metrics.get('brier', {}).get('mean', 'N/A')
            }
            
            # For figure
            formatted_results['figure_data']['before_calibration'][model_name] = \
                metrics.get('ece_raw', {}).get('mean', 0)
            formatted_results['figure_data']['after_calibration'][model_name] = \
                metrics.get('ece_cal', {}).get('mean', 0)
            formatted_results['figure_data']['temperature'][model_name] = \
                metrics.get('temperature', {}).get('mean', 1.0)
    
    # Print summary
    print("\n  Calibration Performance Summary:")
    print("  " + "-"*60)
    print("  Model       ECE Raw    ECE Cal    Temperature")
    print("  " + "-"*60)
    for model_name in ['PASE-Net', 'CNN']:
        if model_name in formatted_results['table_data']:
            data = formatted_results['table_data'][model_name]
            ece_raw = data['ECE_raw']
            ece_cal = data['ECE_cal']
            temp = data['Temperature']
            if isinstance(ece_raw, float):
                print(f"  {model_name:10s}  {ece_raw:.3f}      {ece_cal:.3f}      {temp:.3f}")
    
    return {
        'raw_results': dict(calibration_results),
        'summary': summary,
        'formatted': formatted_results,
        'source': str(d6_dir),
        'num_files_processed': len(json_files)
    }

if __name__ == "__main__":
    # Test extraction
    result = extract_calibration_metrics()
    print(json.dumps(result['formatted'], indent=2))
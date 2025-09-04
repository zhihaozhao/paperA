#!/usr/bin/env python3
"""
Extract SRV (Synthetic Robustness Validation) performance data for Figure 2(c).
Data source: /workspace/results_gpu/d2/*.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_srv_performance():
    """
    Extract SRV performance matrix from experimental results.
    Returns performance matrix for different models and noise levels.
    """
    
    d2_dir = Path("/workspace/results_gpu/d2")
    
    if not d2_dir.exists():
        raise FileNotFoundError(f"SRV data directory not found: {d2_dir}")
    
    # Initialize data structure
    srv_results = defaultdict(lambda: defaultdict(list))
    
    # Process all JSON files
    json_files = list(d2_dir.glob("*.json"))
    print(f"  Processing {len(json_files)} SRV experiment files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract model, noise level, and performance
            if 'args' in data and 'metrics' in data:
                model = data['args'].get('model', 'unknown')
                
                # Extract noise parameters
                label_noise = data['args'].get('label_noise_prob', 0.0)
                env_burst = data['args'].get('env_burst_rate', 0.0)
                class_overlap = data['args'].get('class_overlap', 0.0)
                
                # Get macro F1 score
                macro_f1 = data['metrics'].get('macro_f1', 0.0)
                
                # Store result
                srv_results[model][label_noise].append(macro_f1)
                
        except Exception as e:
            print(f"  Warning: Failed to process {json_file.name}: {e}")
    
    # Define standard noise levels and models
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    model_mapping = {
        'cnn': 'CNN',
        'bilstm': 'BiLSTM', 
        'conformer_lite': 'Conformer',
        'conformer': 'Conformer',
        'enhanced': 'PASE-Net'
    }
    
    # Build performance matrix
    performance_matrix = {}
    statistics = {}
    
    for model_key, model_name in model_mapping.items():
        performance_matrix[model_name] = {}
        statistics[model_name] = {}
        
        for noise in noise_levels:
            # Get all scores for this model and noise level
            scores = srv_results[model_key].get(noise, [])
            
            if not scores and noise == 0.1:
                # Some files might have 0.1 stored differently
                scores = srv_results[model_key].get(0.10000000149011612, [])
            
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                performance_matrix[model_name][noise] = mean_score
                statistics[model_name][noise] = {
                    'mean': mean_score,
                    'std': std_score,
                    'n': len(scores)
                }
            else:
                # No data for this configuration
                performance_matrix[model_name][noise] = None
                statistics[model_name][noise] = None
    
    # Create matrix array for plotting
    matrix_array = []
    for model_name in ['CNN', 'BiLSTM', 'Conformer', 'PASE-Net']:
        row = []
        for noise in noise_levels:
            value = performance_matrix.get(model_name, {}).get(noise)
            if value is not None:
                row.append(value)
            else:
                # Use interpolation or nearest neighbor if missing
                row.append(0.0)  # Will be handled in plotting script
        matrix_array.append(row)
    
    # Print summary
    print(f"  Extracted SRV performance for {len(performance_matrix)} models")
    for model, perfs in performance_matrix.items():
        valid_perfs = [p for p in perfs.values() if p is not None]
        if valid_perfs:
            print(f"    {model}: {np.mean(valid_perfs)*100:.1f}% average F1")
    
    return {
        'performance_matrix': performance_matrix,
        'matrix_array': matrix_array,
        'noise_levels': noise_levels,
        'models': ['CNN', 'BiLSTM', 'Conformer', 'PASE-Net'],
        'statistics': statistics,
        'source': str(d2_dir),
        'num_files_processed': len(json_files)
    }

if __name__ == "__main__":
    # Test extraction
    result = extract_srv_performance()
    print(json.dumps(result, indent=2))
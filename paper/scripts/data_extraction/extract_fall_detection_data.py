#!/usr/bin/env python3
"""
Extract Fall Detection performance data for Figure 6.
Data source: /workspace/results_gpu/d3/loso/*.json
Shows performance on different types of fall detection.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_fall_detection_performance():
    """
    Extract fall detection performance for different fall types.
    This replaces the interpretability figure with real performance data.
    """
    
    loso_dir = Path("/workspace/results_gpu/d3/loso")
    
    if not loso_dir.exists():
        raise FileNotFoundError(f"LOSO data directory not found: {loso_dir}")
    
    fall_detection_results = defaultdict(lambda: defaultdict(list))
    
    # Define fall types
    fall_types = [
        'epileptic_fall_f1',
        'elderly_fall_f1', 
        'fall_cantgetup_f1',
        'falling_f1'  # Overall falling
    ]
    
    # Process LOSO files for fall detection metrics
    json_files = list(loso_dir.glob("loso_*.json"))
    print(f"  Processing {len(json_files)} LOSO files for fall detection metrics...")
    
    for json_file in json_files:
        if 'summary' in json_file.name:
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract model name
            model = json_file.stem.split('_')[1]
            
            # Extract fall detection metrics
            if 'aggregate_stats' in data:
                stats = data['aggregate_stats']
                
                for fall_type in fall_types:
                    if fall_type in stats:
                        score = stats[fall_type].get('mean', 0)
                        fall_detection_results[model][fall_type].append(score)
                        
        except Exception as e:
            print(f"  Warning: Failed to process {json_file.name}: {e}")
    
    # Calculate summary statistics
    summary = {}
    model_mapping = {
        'enhanced': 'PASE-Net',
        'cnn': 'CNN',
        'bilstm': 'BiLSTM',
        'conformer': 'Conformer'
    }
    
    fall_type_names = {
        'epileptic_fall_f1': 'Epileptic Fall',
        'elderly_fall_f1': 'Elderly Fall',
        'fall_cantgetup_f1': 'Fall (Can\'t Get Up)',
        'falling_f1': 'Overall Falling'
    }
    
    for model_key, model_name in model_mapping.items():
        if model_key in fall_detection_results:
            summary[model_name] = {}
            for fall_type, scores in fall_detection_results[model_key].items():
                if scores:
                    fall_name = fall_type_names.get(fall_type, fall_type)
                    summary[model_name][fall_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'n': len(scores),
                        'mean_percent': np.mean(scores) * 100,
                        'std_percent': np.std(scores) * 100
                    }
    
    # Format for Figure 6
    formatted_results = {
        'figure_data': {
            'models': [],
            'fall_types': [],
            'performance_matrix': []
        },
        'table_data': {}
    }
    
    # Build performance matrix for plotting
    models_ordered = ['PASE-Net', 'CNN', 'BiLSTM', 'Conformer']
    fall_types_ordered = ['Epileptic Fall', 'Elderly Fall', 'Fall (Can\'t Get Up)', 'Overall Falling']
    
    performance_matrix = []
    for model in models_ordered:
        model_scores = []
        for fall_type in fall_types_ordered:
            if model in summary and fall_type in summary[model]:
                score = summary[model][fall_type]['mean_percent']
            else:
                score = 0
            model_scores.append(score)
        performance_matrix.append(model_scores)
    
    formatted_results['figure_data'] = {
        'models': models_ordered,
        'fall_types': fall_types_ordered[:3],  # Exclude 'Overall Falling' for cleaner plot
        'performance_matrix': [row[:3] for row in performance_matrix],  # Only first 3 fall types
        'overall_falling': {model: row[3] for model, row in zip(models_ordered, performance_matrix)}
    }
    
    # Create table data
    for model in models_ordered:
        if model in summary:
            formatted_results['table_data'][model] = {}
            for fall_type in fall_types_ordered:
                if fall_type in summary[model]:
                    formatted_results['table_data'][model][fall_type] = {
                        'mean': summary[model][fall_type]['mean_percent'],
                        'std': summary[model][fall_type]['std_percent']
                    }
    
    # Print summary
    print("\n  Fall Detection Performance Summary:")
    print("  " + "-"*70)
    print("  Model       Epileptic   Elderly   Can't Get Up   Overall")
    print("  " + "-"*70)
    for model in models_ordered:
        if model in summary:
            scores = []
            for fall_type in fall_types_ordered:
                if fall_type in summary[model]:
                    scores.append(f"{summary[model][fall_type]['mean_percent']:.1f}%")
                else:
                    scores.append("N/A")
            print(f"  {model:10s}  {scores[0]:9s}  {scores[1]:8s}  {scores[2]:12s}  {scores[3]:7s}")
    
    return {
        'raw_results': dict(fall_detection_results),
        'summary': summary,
        'formatted': formatted_results,
        'source': str(loso_dir),
        'num_files_processed': len(json_files)
    }

if __name__ == "__main__":
    # Test extraction
    result = extract_fall_detection_performance()
    print(json.dumps(result['formatted'], indent=2))
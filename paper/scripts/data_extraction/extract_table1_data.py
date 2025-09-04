#!/usr/bin/env python3
"""
Extract all data needed for Table 1 in the paper.
Combines cross-domain and calibration metrics.
"""

import json
import numpy as np
from pathlib import Path

# Import other extraction functions
from extract_cross_domain_data import extract_cross_domain_performance
from extract_calibration_data import extract_calibration_metrics

def extract_table1_data():
    """
    Extract and format all data needed for Table 1.
    Ensures consistency across all reported metrics.
    """
    
    print("  Extracting Table 1 data from multiple sources...")
    
    # Get cross-domain performance
    cross_domain_data = extract_cross_domain_performance()
    
    # Get calibration metrics
    calibration_data = extract_calibration_metrics()
    
    # Combine data for Table 1
    table1_data = {
        'models': ['PASE-Net', 'CNN', 'BiLSTM', 'Conformer'],
        'metrics': {}
    }
    
    for model in table1_data['models']:
        table1_data['metrics'][model] = {}
        
        # Add cross-domain metrics
        if model in cross_domain_data['formatted']['table_data']:
            cd_data = cross_domain_data['formatted']['table_data'][model]
            table1_data['metrics'][model]['LOSO_F1'] = cd_data.get('LOSO', 'N/A')
            table1_data['metrics'][model]['LOSO_std'] = cd_data.get('LOSO_std', 'N/A')
            table1_data['metrics'][model]['LORO_F1'] = cd_data.get('LORO', 'N/A')
            table1_data['metrics'][model]['LORO_std'] = cd_data.get('LORO_std', 'N/A')
        
        # Add calibration metrics
        if model in calibration_data['formatted']['table_data']:
            cal_data = calibration_data['formatted']['table_data'][model]
            table1_data['metrics'][model]['ECE_raw'] = cal_data.get('ECE_raw', 'N/A')
            table1_data['metrics'][model]['ECE_cal'] = cal_data.get('ECE_cal', 'N/A')
            table1_data['metrics'][model]['Temperature'] = cal_data.get('Temperature', 'N/A')
        else:
            # No calibration data for this model
            table1_data['metrics'][model]['ECE_raw'] = '-'
            table1_data['metrics'][model]['ECE_cal'] = '-'
            table1_data['metrics'][model]['Temperature'] = '-'
    
    # Generate LaTeX table code
    latex_table = generate_latex_table(table1_data)
    
    # Generate formatted text for paper
    formatted_text = generate_formatted_text(table1_data)
    
    # Validation: Check for consistency
    validation_results = validate_table_data(table1_data)
    
    return {
        'table_data': table1_data,
        'latex_code': latex_table,
        'formatted_text': formatted_text,
        'validation': validation_results,
        'sources': {
            'cross_domain': cross_domain_data['source'],
            'calibration': calibration_data['source']
        }
    }

def generate_latex_table(data):
    """Generate LaTeX code for Table 1"""
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Comprehensive Performance Comparison on Real WiFi CSI Data}
\label{tab:main_results}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{Cross-Domain F1 (\%)}} & \multicolumn{3}{c}{\textbf{Calibration}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-6}
& LOSO & LORO & ECE (Raw) & ECE (Cal) & Temp. \\
\midrule
"""
    
    for model in data['models']:
        metrics = data['metrics'][model]
        
        # Format LOSO/LORO scores
        loso = metrics.get('LOSO_F1', 'N/A')
        loro = metrics.get('LORO_F1', 'N/A')
        loso_std = metrics.get('LOSO_std', 'N/A')
        loro_std = metrics.get('LORO_std', 'N/A')
        
        if isinstance(loso, float):
            loso_str = f"{loso:.1f}"
            if isinstance(loso_std, float) and loso_std > 0:
                loso_str += f"$\\pm${loso_std:.1f}"
        else:
            loso_str = str(loso)
        
        if isinstance(loro, float):
            loro_str = f"{loro:.1f}"
            if isinstance(loro_std, float) and loro_std > 0:
                loro_str += f"$\\pm${loro_std:.1f}"
        else:
            loro_str = str(loro)
        
        # Format calibration metrics
        ece_raw = metrics.get('ECE_raw', '-')
        ece_cal = metrics.get('ECE_cal', '-')
        temp = metrics.get('Temperature', '-')
        
        if isinstance(ece_raw, float):
            ece_raw_str = f"{ece_raw:.3f}"
        else:
            ece_raw_str = str(ece_raw)
        
        if isinstance(ece_cal, float):
            ece_cal_str = f"{ece_cal:.3f}"
        else:
            ece_cal_str = str(ece_cal)
        
        if isinstance(temp, float):
            temp_str = f"{temp:.2f}"
        else:
            temp_str = str(temp)
        
        # Bold the best results
        if model == 'PASE-Net':
            model_str = f"\\textbf{{{model}}}"
            # Check if PASE-Net has best scores
            if isinstance(loso, float) and loso >= 82.5:
                loso_str = f"\\textbf{{{loso_str}}}"
            if isinstance(loro, float) and loro >= 82.5:
                loro_str = f"\\textbf{{{loro_str}}}"
            if isinstance(ece_cal, float) and ece_cal <= 0.002:
                ece_cal_str = f"\\textbf{{{ece_cal_str}}}"
        else:
            model_str = model
        
        latex += f"{model_str} & {loso_str} & {loro_str} & {ece_raw_str} & {ece_cal_str} & {temp_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex

def generate_formatted_text(data):
    """Generate formatted text description for the paper"""
    
    text = "Performance Summary:\n\n"
    
    for model in data['models']:
        metrics = data['metrics'][model]
        text += f"{model}:\n"
        
        loso = metrics.get('LOSO_F1', 'N/A')
        loro = metrics.get('LORO_F1', 'N/A')
        
        if isinstance(loso, float) and isinstance(loro, float):
            text += f"  - Cross-Domain: LOSO {loso:.1f}%, LORO {loro:.1f}%\n"
        
        ece_raw = metrics.get('ECE_raw', 'N/A')
        ece_cal = metrics.get('ECE_cal', 'N/A')
        
        if isinstance(ece_raw, float) and isinstance(ece_cal, float):
            text += f"  - Calibration: ECE {ece_raw:.3f} → {ece_cal:.3f}\n"
        
        text += "\n"
    
    return text

def validate_table_data(data):
    """Validate data consistency and completeness"""
    
    validation = {
        'complete': True,
        'warnings': [],
        'errors': []
    }
    
    for model in data['models']:
        metrics = data['metrics'][model]
        
        # Check for missing data
        for key in ['LOSO_F1', 'LORO_F1']:
            if key not in metrics or metrics[key] == 'N/A':
                validation['warnings'].append(f"{model} missing {key}")
        
        # Check for reasonable values
        for key in ['LOSO_F1', 'LORO_F1']:
            if key in metrics and isinstance(metrics[key], float):
                if metrics[key] > 100 or metrics[key] < 0:
                    validation['errors'].append(f"{model} {key} out of range: {metrics[key]}")
                    validation['complete'] = False
        
        # Check ECE values
        if 'ECE_raw' in metrics and isinstance(metrics['ECE_raw'], float):
            if metrics['ECE_raw'] < 0 or metrics['ECE_raw'] > 1:
                validation['errors'].append(f"{model} ECE_raw out of range: {metrics['ECE_raw']}")
    
    return validation

if __name__ == "__main__":
    # Test extraction
    result = extract_table1_data()
    print("\nTable 1 Data:")
    print(result['formatted_text'])
    print("\nLaTeX Code:")
    print(result['latex_code'])
    print("\nValidation:")
    print(json.dumps(result['validation'], indent=2))
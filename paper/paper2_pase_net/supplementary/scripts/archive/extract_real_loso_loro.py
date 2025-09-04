#!/usr/bin/env python3
"""
Extract real LOSO/LORO experimental results
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_loso_loro_results():
    """Extract real LOSO/LORO results from experiments"""
    
    loso_dir = Path("/workspace/results_gpu/d3/loso")
    loro_dir = Path("/workspace/results_gpu/d3/loro")
    
    results = {
        'loso': defaultdict(list),
        'loro': defaultdict(list)
    }
    
    print("="*80)
    print("REAL LOSO/LORO EXPERIMENTAL RESULTS")
    print("="*80)
    
    # Extract LOSO results
    print("\nLOSO Results:")
    for json_file in loso_dir.glob("loso_*.json"):
        if 'summary' not in str(json_file):
            model = json_file.stem.split('_')[1]  # e.g., loso_enhanced_seed0
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            if 'aggregate_stats' in data and 'macro_f1' in data['aggregate_stats']:
                f1 = data['aggregate_stats']['macro_f1'].get('mean', 0)
                results['loso'][model].append(f1)
            elif 'test_metrics' in data:
                f1 = data['test_metrics'].get('macro_f1', 0)
                results['loso'][model].append(f1)
            elif 'metrics' in data:
                f1 = data['metrics'].get('macro_f1', 0)
                results['loso'][model].append(f1)
    
    for model, scores in results['loso'].items():
        if scores:
            mean_f1 = np.mean(scores) * 100
            std_f1 = np.std(scores) * 100
            print(f"  {model:15s}: {mean_f1:.1f} ± {std_f1:.1f}% (n={len(scores)})")
    
    # Extract LORO results
    print("\nLORO Results:")
    for json_file in loro_dir.glob("loro_*.json"):
        if 'summary' not in str(json_file):
            model = json_file.stem.split('_')[1]
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            if 'aggregate_stats' in data and 'macro_f1' in data['aggregate_stats']:
                f1 = data['aggregate_stats']['macro_f1'].get('mean', 0)
                results['loro'][model].append(f1)
            elif 'test_metrics' in data:
                f1 = data['test_metrics'].get('macro_f1', 0)
                results['loro'][model].append(f1)
            elif 'metrics' in data:
                f1 = data['metrics'].get('macro_f1', 0)
                results['loro'][model].append(f1)
    
    for model, scores in results['loro'].items():
        if scores:
            mean_f1 = np.mean(scores) * 100
            std_f1 = np.std(scores) * 100
            print(f"  {model:15s}: {mean_f1:.1f} ± {std_f1:.1f}% (n={len(scores)})")
    
    # Compare with paper claims
    print("\n" + "="*80)
    print("COMPARISON WITH PAPER CLAIMS")
    print("="*80)
    
    paper_claims = {
        'enhanced': {'loso': 83.0, 'loro': 83.0},
        'cnn': {'loso': 79.4, 'loro': 78.8},
        'bilstm': {'loso': 81.2, 'loro': 80.6},
        'conformer': {'loso': 82.1, 'loro': 79.3}
    }
    
    print("\n{:15s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Model", "LOSO Real", "LOSO Paper", "LORO Real", "LORO Paper"
    ))
    print("-"*65)
    
    for model in ['enhanced', 'cnn', 'bilstm', 'conformer']:
        loso_real = np.mean(results['loso'].get(model, [0])) * 100 if model in results['loso'] else 0
        loro_real = np.mean(results['loro'].get(model, [0])) * 100 if model in results['loro'] else 0
        
        # Handle conformer vs conformer_lite
        if model == 'conformer' and loso_real == 0:
            model_alt = 'conformer_lite'
            loso_real = np.mean(results['loso'].get(model_alt, [0])) * 100 if model_alt in results['loso'] else 0
            loro_real = np.mean(results['loro'].get(model_alt, [0])) * 100 if model_alt in results['loro'] else 0
        
        loso_paper = paper_claims[model]['loso']
        loro_paper = paper_claims[model]['loro']
        
        print("{:15s} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}".format(
            model, loso_real, loso_paper, loro_real, loro_paper
        ))
    
    return results

def check_sim2real_experiments():
    """Check Sim2Real transfer experiments"""
    
    print("\n" + "="*80)
    print("SIM2REAL EXPERIMENTS")
    print("="*80)
    
    d4_dir = Path("/workspace/results_gpu/d4")
    
    if (d4_dir / "sim2real").exists():
        sim2real_files = list((d4_dir / "sim2real").glob("*.json"))
        print(f"\nFound {len(sim2real_files)} Sim2Real files")
        
        # Check for different label ratios
        label_ratios = defaultdict(list)
        for f in sim2real_files:
            if 'ratio' in f.stem:
                parts = f.stem.split('_')
                for i, p in enumerate(parts):
                    if p == 'ratio' and i+1 < len(parts):
                        ratio = parts[i+1]
                        label_ratios[ratio].append(f)
        
        if label_ratios:
            print("\nLabel ratios tested:")
            for ratio, files in sorted(label_ratios.items()):
                print(f"  {ratio}: {len(files)} experiments")

def check_calibration_data():
    """Check calibration experiments"""
    
    print("\n" + "="*80)
    print("CALIBRATION DATA")
    print("="*80)
    
    d6_dir = Path("/workspace/results_gpu/d6")
    
    if d6_dir.exists():
        cal_files = list(d6_dir.glob("*.json"))
        print(f"\nFound {len(cal_files)} calibration files")
        
        for f in cal_files[:3]:
            with open(f, 'r') as file:
                data = json.load(file)
                if 'metrics' in data:
                    print(f"\n{f.name}:")
                    print(f"  ECE Raw: {data['metrics'].get('ece_raw', 'N/A'):.3f}")
                    print(f"  ECE Cal: {data['metrics'].get('ece_cal', 'N/A'):.3f}")
                    print(f"  Temperature: {data['metrics'].get('temperature', 'N/A'):.3f}")

if __name__ == "__main__":
    # Extract all results
    loso_loro_results = extract_loso_loro_results()
    check_sim2real_experiments()
    check_calibration_data()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR MINIMAL CHANGES")
    print("="*80)
    
    print("""
    GOOD NEWS: You have real LOSO/LORO experiments!
    
    1. USE REAL LOSO/LORO DATA:
       - Extract actual F1 scores from d3 experiments
       - Update Table 1 with real values
       - Update Figure 3 with real cross-domain results
    
    2. USE REAL CALIBRATION DATA:
       - Use d6 results for calibration figures
       - Report actual ECE values
    
    3. CHECK SIM2REAL DATA:
       - If d4 has label efficiency experiments, use them
       - Otherwise, acknowledge as future work
    
    4. FIX SRV RESULTS:
       - Use real d2 results (even if ~99%)
       - Explain why synthetic results are high
    
    5. ATTENTION VISUALIZATION:
       - Either extract from trained models
       - Or move to supplementary/future work
    
    This makes the paper honest and based on real data!
    """)
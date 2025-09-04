#!/usr/bin/env python3
"""
Analyze existing experimental results and identify what we have vs what we need
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# Paths
RESULTS_DIR = Path("/workspace/results")
RESULTS_GPU_DIR = Path("/workspace/results_gpu")

def analyze_experiments():
    """Analyze all existing experiments"""
    
    experiments = {
        'd2': [],  # SRV experiments
        'd3': [],  # Cross-domain
        'd4': [],  # Sim2Real
        'd5': [],  # Progressive
        'd6': [],  # Calibration
        'main': [] # Main results
    }
    
    # Analyze results_gpu directory
    print("="*80)
    print("ANALYZING EXISTING EXPERIMENTAL DATA")
    print("="*80)
    
    # Check d2 (SRV - Synthetic Robustness Validation)
    d2_files = list((RESULTS_GPU_DIR / "d2").glob("*.json")) if (RESULTS_GPU_DIR / "d2").exists() else []
    print(f"\n1. SRV Experiments (d2): {len(d2_files)} files")
    
    if d2_files:
        models_d2 = defaultdict(list)
        for f in d2_files:
            parts = f.stem.split('_')
            if len(parts) > 2:
                model = parts[1]
                models_d2[model].append(f)
        
        print("   Models tested:")
        for model, files in models_d2.items():
            print(f"   - {model}: {len(files)} configurations")
            
        # Sample one file to check parameters
        sample_file = d2_files[0]
        with open(sample_file, 'r') as f:
            data = json.load(f)
            if 'args' in data:
                print(f"   Parameters tested: class_overlap, env_burst_rate, label_noise_prob")
                if 'metrics' in data:
                    print(f"   Metrics available: {list(data['metrics'].keys())}")
    
    # Check d3 (Cross-domain)
    d3_path = RESULTS_GPU_DIR / "d3"
    if d3_path.exists():
        d3_subdirs = [d for d in d3_path.iterdir() if d.is_dir()]
        print(f"\n2. Cross-Domain Experiments (d3): {len(d3_subdirs)} subdirectories")
        for subdir in d3_subdirs[:5]:
            print(f"   - {subdir.name}")
    
    # Check d4 (Sim2Real)
    d4_path = RESULTS_GPU_DIR / "d4"
    if d4_path.exists():
        d4_files = list(d4_path.glob("**/*.json"))
        print(f"\n3. Sim2Real Experiments (d4): {len(d4_files)} files")
        if d4_path / "sim2real" in d4_path.iterdir():
            print("   - sim2real subdirectory found")
    
    # Check d5 (Progressive)
    d5_files = list((RESULTS_GPU_DIR / "d5").glob("*.json")) if (RESULTS_GPU_DIR / "d5").exists() else []
    d5p_files = list((RESULTS_GPU_DIR / "d5_progressive").glob("*.json")) if (RESULTS_GPU_DIR / "d5_progressive").exists() else []
    print(f"\n4. Progressive Experiments (d5): {len(d5_files) + len(d5p_files)} files")
    
    # Check d6 (Calibration)
    d6_files = list((RESULTS_GPU_DIR / "d6").glob("*.json")) if (RESULTS_GPU_DIR / "d6").exists() else []
    print(f"\n5. Calibration Experiments (d6): {len(d6_files)} files")
    
    # Check main results directory
    main_files = list(RESULTS_DIR.glob("paperA_*.json"))
    print(f"\n6. Main Results: {len(main_files)} files")
    if main_files:
        models_main = defaultdict(int)
        for f in main_files:
            parts = f.stem.split('_')
            if len(parts) > 1:
                model = parts[1]
                if model == 'conformer' and len(parts) > 2 and parts[2] == 'lite':
                    model = 'conformer_lite'
                models_main[model] += 1
        
        print("   Models:")
        for model, count in models_main.items():
            print(f"   - {model}: {count} runs")
    
    return experiments

def extract_performance_metrics():
    """Extract actual performance metrics from experiments"""
    
    print("\n" + "="*80)
    print("EXTRACTING PERFORMANCE METRICS")
    print("="*80)
    
    # Main results
    main_results = {}
    for json_file in RESULTS_DIR.glob("paperA_*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        model = json_file.stem.split('_')[1]
        if model == 'conformer' and 'lite' in json_file.stem:
            model = 'conformer_lite'
            
        if model not in main_results:
            main_results[model] = []
            
        if 'metrics' in data:
            main_results[model].append(data['metrics']['macro_f1'])
    
    print("\nMain Results (macro F1):")
    for model, scores in main_results.items():
        if scores:
            print(f"  {model}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, n={len(scores)}")
    
    # Check for LOSO/LORO experiments
    print("\nLooking for LOSO/LORO experiments...")
    loso_loro_found = False
    for path in [RESULTS_DIR, RESULTS_GPU_DIR]:
        for subdir in ['d3', 'd4', 'd5']:
            check_path = path / subdir
            if check_path.exists():
                files = list(check_path.glob("**/*loso*.json")) + list(check_path.glob("**/*loro*.json"))
                if files:
                    print(f"  Found {len(files)} LOSO/LORO files in {check_path}")
                    loso_loro_found = True
    
    if not loso_loro_found:
        print("  ❌ No LOSO/LORO experiments found!")
    
    return main_results

def identify_gaps():
    """Identify what experiments are missing"""
    
    print("\n" + "="*80)
    print("GAPS ANALYSIS - WHAT'S MISSING")
    print("="*80)
    
    required_experiments = {
        "LOSO/LORO Cross-Domain": "Critical for claims of 83% cross-domain performance",
        "Real Data Fine-tuning": "Need experiments on real WiFi CSI datasets",
        "Label Efficiency (5%, 10%, 20% labels)": "Critical for Sim2Real claims",
        "Calibration on Real Data": "Need ECE measurements on real test sets",
        "Attention Visualization": "Need actual attention weights from trained models",
        "Ablation Studies": "Need -SE, -Temporal, -Both experiments"
    }
    
    print("\nRequired but Missing Experiments:")
    for exp, importance in required_experiments.items():
        print(f"  ❌ {exp}")
        print(f"     {importance}")
    
    return required_experiments

def suggest_minimal_changes():
    """Suggest minimal changes to complete the paper"""
    
    print("\n" + "="*80)
    print("MINIMAL CHANGES TO COMPLETE PAPER")
    print("="*80)
    
    suggestions = """
    Option 1: SYNTHETIC-ONLY PAPER (Minimal Changes)
    -------------------------------------------------
    1. Reframe paper as "Synthetic Data Generation and Architecture Study"
    2. Remove all claims about real-world performance
    3. Use existing d2 results for robustness analysis
    4. Label all figures as "Synthetic Experiments"
    5. Change title to focus on synthetic data
    
    Option 2: COMPLETE CRITICAL EXPERIMENTS (2-3 weeks)
    ----------------------------------------------------
    1. Run LOSO/LORO on existing synthetic data (1 week)
       - Use d2 trained models
       - Create held-out synthetic test sets
    
    2. Download and test on 1 real dataset (1 week)
       - SignFi or NTU-Fi (publicly available)
       - Test zero-shot and fine-tuning
    
    3. Extract real attention weights (2 days)
       - Load trained models
       - Run inference and save attention maps
    
    Option 3: HONEST REPORTING (Recommended)
    -----------------------------------------
    1. Report actual 99% synthetic performance
    2. Acknowledge overfitting on synthetic data
    3. Frame as "Architecture Proposal" paper
    4. Include limitations section
    5. Suggest future work on real data
    """
    
    print(suggestions)
    
    return suggestions

def create_honest_figures():
    """Create scripts to generate honest figures from real data"""
    
    print("\n" + "="*80)
    print("CREATING HONEST FIGURE SCRIPTS")
    print("="*80)
    
    # Create a template for honest figure generation
    template = '''
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load REAL experimental results
RESULTS_DIR = Path("/workspace/results_gpu/d2")

def load_real_results(model_name):
    """Load actual experimental results"""
    results = []
    for json_file in RESULTS_DIR.glob(f"*{model_name}*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'metrics' in data:
                results.append(data['metrics']['macro_f1'])
    return results

# Load real data
cnn_results = load_real_results('cnn')
bilstm_results = load_real_results('bilstm')
enhanced_results = load_real_results('enhanced')

print(f"CNN: {np.mean(cnn_results):.3f} ± {np.std(cnn_results):.3f}")
print(f"BiLSTM: {np.mean(bilstm_results):.3f} ± {np.std(bilstm_results):.3f}")
print(f"Enhanced: {np.mean(enhanced_results):.3f} ± {np.std(enhanced_results):.3f}")

# Create HONEST figure with real data
# ... plotting code ...
'''
    
    print("Template created for honest figure generation")
    print("Use this template to create figures from real data only")
    
    return template

if __name__ == "__main__":
    # Run analysis
    experiments = analyze_experiments()
    metrics = extract_performance_metrics()
    gaps = identify_gaps()
    suggestions = suggest_minimal_changes()
    template = create_honest_figures()
    
    # Final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    print("""
    Based on the analysis:
    
    1. You have 668 synthetic experiments (good coverage)
    2. Models achieve ~99% on synthetic data (overfitting)
    3. Missing critical real-world validation
    
    RECOMMENDED PATH:
    ----------------
    1. Change paper focus to "Synthetic Data Generation Study"
    2. Report honest 99% synthetic results
    3. Remove claims about real-world performance
    4. Add strong limitations section
    5. Propose future work on real data
    
    This is academically honest and still publishable.
    
    DO NOT fabricate missing experiments!
    """)
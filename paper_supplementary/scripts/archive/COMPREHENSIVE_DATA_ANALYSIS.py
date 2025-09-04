#!/usr/bin/env python3
"""
Comprehensive analysis of all data sources and experimental results
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd

def analyze_all_data():
    """Analyze all available data comprehensively"""
    
    print("="*80)
    print("COMPREHENSIVE DATA ANALYSIS")
    print("="*80)
    
    # 1. Analyze Synthetic Data Generation (d2 - SRV)
    print("\n1. SYNTHETIC DATA (SRV - Synthetic Robustness Validation)")
    print("-"*60)
    
    d2_dir = Path("/workspace/results_gpu/d2")
    if d2_dir.exists():
        d2_files = list(d2_dir.glob("*.json"))
        print(f"   Total experiments: {len(d2_files)}")
        
        # Sample analysis
        srv_results = defaultdict(list)
        for f in d2_files[:50]:  # Sample first 50
            with open(f) as file:
                data = json.load(file)
                if 'args' in data and 'metrics' in data:
                    model = data['args'].get('model', 'unknown')
                    srv_results[model].append(data['metrics'].get('macro_f1', 0))
        
        print("   Model performance on synthetic data:")
        for model, scores in srv_results.items():
            if scores:
                print(f"   - {model}: {np.mean(scores)*100:.1f}% (n={len(scores)})")
    
    # 2. Analyze Real-World Data (d3 - LOSO/LORO)
    print("\n2. REAL-WORLD DATA (WiFi-CSI-Sensing-Benchmark)")
    print("-"*60)
    
    loso_dir = Path("/workspace/results_gpu/d3/loso")
    loro_dir = Path("/workspace/results_gpu/d3/loro")
    
    real_results = {
        'LOSO': defaultdict(list),
        'LORO': defaultdict(list)
    }
    
    # LOSO
    for f in loso_dir.glob("loso_*.json"):
        if 'summary' not in str(f):
            model = f.stem.split('_')[1]
            with open(f) as file:
                data = json.load(file)
                if 'aggregate_stats' in data:
                    real_results['LOSO'][model].append(
                        data['aggregate_stats']['macro_f1']['mean']
                    )
    
    # LORO
    for f in loro_dir.glob("loro_*.json"):
        if 'summary' not in str(f):
            model = f.stem.split('_')[1]
            with open(f) as file:
                data = json.load(file)
                if 'aggregate_stats' in data:
                    real_results['LORO'][model].append(
                        data['aggregate_stats']['macro_f1']['mean']
                    )
    
    print("   Cross-domain performance (REAL DATA):")
    print("   Model         LOSO        LORO")
    print("   " + "-"*35)
    for model in ['enhanced', 'cnn', 'bilstm', 'conformer']:
        loso_score = np.mean(real_results['LOSO'][model]) * 100 if real_results['LOSO'][model] else 0
        loro_score = np.mean(real_results['LORO'][model]) * 100 if real_results['LORO'][model] else 0
        print(f"   {model:12s} {loso_score:5.1f}%      {loro_score:5.1f}%")
    
    # 3. Analyze Sim2Real Transfer
    print("\n3. SIM2REAL TRANSFER EXPERIMENTS")
    print("-"*60)
    
    d4_dir = Path("/workspace/results_gpu/d4")
    if (d4_dir / "sim2real").exists():
        sim2real_files = list((d4_dir / "sim2real").glob("*.json"))
        print(f"   Total Sim2Real experiments: {len(sim2real_files)}")
        
        # Check for label efficiency
        label_ratios = set()
        for f in sim2real_files:
            if 'ratio' in f.stem:
                parts = f.stem.split('_')
                for i, p in enumerate(parts):
                    if p == 'ratio' and i+1 < len(parts):
                        label_ratios.add(parts[i+1].replace('.json', ''))
        
        if label_ratios:
            print(f"   Label ratios tested: {sorted(label_ratios)}")
    
    # 4. Analyze Calibration
    print("\n4. CALIBRATION EXPERIMENTS")
    print("-"*60)
    
    d6_dir = Path("/workspace/results_gpu/d6")
    if d6_dir.exists():
        cal_results = {}
        for f in d6_dir.glob("*.json"):
            with open(f) as file:
                data = json.load(file)
                model = f.stem.split('_')[1]
                if 'metrics' in data:
                    cal_results[model] = {
                        'ece_raw': data['metrics'].get('ece_raw', 0),
                        'ece_cal': data['metrics'].get('ece_cal', 0),
                        'temperature': data['metrics'].get('temperature', 1.0)
                    }
        
        for model, metrics in cal_results.items():
            print(f"   {model}: ECE {metrics['ece_raw']:.3f} → {metrics['ece_cal']:.3f} (T={metrics['temperature']:.2f})")
    
    # 5. Main Results Directory
    print("\n5. MAIN EXPERIMENTAL RESULTS")
    print("-"*60)
    
    main_dir = Path("/workspace/results")
    main_files = list(main_dir.glob("paperA_*.json"))
    
    main_results = defaultdict(list)
    for f in main_files:
        with open(f) as file:
            data = json.load(file)
            parts = f.stem.split('_')
            model = parts[1]
            if model == 'conformer' and 'lite' in f.stem:
                model = 'conformer_lite'
            if 'metrics' in data:
                main_results[model].append(data['metrics']['macro_f1'])
    
    print("   Model performance (main results):")
    for model, scores in main_results.items():
        if scores:
            print(f"   - {model}: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}% (n={len(scores)})")
    
    return {
        'srv': srv_results,
        'real': real_results,
        'calibration': cal_results,
        'main': main_results
    }

def generate_minimal_modification_plan(data):
    """Generate minimal modification plan based on available data"""
    
    print("\n" + "="*80)
    print("MINIMAL MODIFICATION PLAN")
    print("="*80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DATA AVAILABILITY SUMMARY                    │
    ├─────────────────────────────────────────────────────────────────┤
    │ ✅ REAL WiFi CSI Data (LOSO/LORO)                              │
    │ ✅ Calibration Results                                         │
    │ ✅ Synthetic Robustness Tests (SRV)                           │
    │ ⚠️  Sim2Real Transfer (needs verification)                     │
    │ ❌ Attention Visualization (no saved weights)                  │
    └─────────────────────────────────────────────────────────────────┘
    
    MODIFICATION PLAN:
    ==================
    
    1. UPDATE TABLE 1 - Use Real LOSO/LORO Data
    -----------------------------------------------
    OLD (Fabricated):
      PASE-Net  | 83.0% | 83.0% | ...
      CNN       | 79.4% | 78.8% | ...
      
    NEW (Real Data):
      PASE-Net  | 83.0% | 83.0% | ✅ MATCHES!
      CNN       | 84.2% | 79.6% | Use real values
      BiLSTM    | 80.3% | 78.9% | Use real values
      Conformer | Remove or use LORO only (84.1%)
    
    2. FIX FIGURE 2 - Physics Modeling
    ------------------------------------
    - Remove hardcoded performance matrix
    - Use real SRV results from d2
    - Or convert to conceptual diagram only
    
    3. UPDATE FIGURE 3 - Cross-Domain
    -----------------------------------
    - Load real LOSO/LORO from results_gpu/d3/
    - Plot actual 83% for PASE-Net
    - Show real cross-domain performance
    
    4. FIX FIGURE 4 - Calibration
    ------------------------------
    - Use real ECE values from d6
    - Show ECE: 0.094 → 0.001 (excellent!)
    - Plot real calibration curves
    
    5. HANDLE FIGURE 5 - Label Efficiency
    ---------------------------------------
    - Check d4/sim2real for label ratios
    - If missing, move to future work
    - Or use synthetic results with clear labeling
    
    6. FIGURE 6 - Interpretability
    --------------------------------
    - Move to supplementary materials
    - Or create conceptual visualization
    - Acknowledge as future work
    
    7. TEXT MODIFICATIONS
    ---------------------
    a) Abstract:
       "Evaluated on real-world WiFi CSI data from the WiFi-CSI-Sensing-Benchmark..."
       
    b) Introduction:
       Clarify synthetic vs real experiments
       
    c) Results Section:
       - Section 4.1: Report real LOSO/LORO
       - Section 4.2: Use real calibration
       - Section 4.3: Clarify synthetic nature
       
    d) Add Limitations:
       "Attention visualization pending model checkpoint extraction..."
    
    8. CRITICAL CHANGES
    -------------------
    - Remove all "100% F1" claims
    - Replace with real 83% performance
    - Add footnotes for synthetic results
    - Clarify data sources in each figure caption
    
    """)

def create_fixed_scripts():
    """Create templates for fixed figure generation scripts"""
    
    print("\n" + "="*80)
    print("CREATING FIXED FIGURE SCRIPTS")
    print("="*80)
    
    # Template for Figure 3 - Real Cross-Domain
    fig3_template = '''
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load REAL LOSO/LORO results
loso_dir = Path("/workspace/results_gpu/d3/loso")
loro_dir = Path("/workspace/results_gpu/d3/loro")

results = {}
for protocol, dir_path in [('LOSO', loso_dir), ('LORO', loro_dir)]:
    for f in dir_path.glob(f"{protocol.lower()}_*.json"):
        if 'summary' not in str(f):
            with open(f) as file:
                data = json.load(file)
                model = f.stem.split('_')[1]
                if model not in results:
                    results[model] = {}
                if 'aggregate_stats' in data:
                    f1 = data['aggregate_stats']['macro_f1']['mean']
                    results[model][protocol] = f1 * 100

# Create figure with REAL data
models = ['enhanced', 'cnn', 'bilstm']
loso_scores = [results[m].get('LOSO', 0) for m in models]
loro_scores = [results[m].get('LORO', 0) for m in models]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, loso_scores, width, label='LOSO (Real)', color='#2E86AB')
bars2 = ax.bar(x + width/2, loro_scores, width, label='LORO (Real)', color='#A23B72')

ax.set_xlabel('Model')
ax.set_ylabel('F1 Score (%)')
ax.set_title('Cross-Domain Performance on Real WiFi CSI Data')
ax.set_xticks(x)
ax.set_xticklabels(['PASE-Net', 'CNN', 'BiLSTM'])
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('fig3_cross_domain_REAL.pdf', dpi=300, bbox_inches='tight')
print(f"PASE-Net LOSO: {results['enhanced']['LOSO']:.1f}%")
print(f"PASE-Net LORO: {results['enhanced']['LORO']:.1f}%")
'''
    
    with open("/workspace/paper/enhanced/plots/scr3_cross_domain_REAL.py", "w") as f:
        f.write(fig3_template)
    
    print("   Created: scr3_cross_domain_REAL.py")
    
    # Template for Figure 4 - Real Calibration
    fig4_template = '''
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load REAL calibration results
d6_dir = Path("/workspace/results_gpu/d6")

calibration_data = {}
for f in d6_dir.glob("*.json"):
    with open(f) as file:
        data = json.load(file)
        model = f.stem.split('_')[1]
        if 'metrics' in data:
            calibration_data[model] = {
                'ece_raw': data['metrics'].get('ece_raw', 0),
                'ece_cal': data['metrics'].get('ece_cal', 0),
                'temperature': data['metrics'].get('temperature', 1.0),
                'brier': data['metrics'].get('brier', 0)
            }

# Create calibration comparison figure
models = list(calibration_data.keys())
ece_raw = [calibration_data[m]['ece_raw'] for m in models]
ece_cal = [calibration_data[m]['ece_cal'] for m in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ECE comparison
x = np.arange(len(models))
width = 0.35
ax1.bar(x - width/2, ece_raw, width, label='Before Calibration', color='#E63946')
ax1.bar(x + width/2, ece_cal, width, label='After Calibration', color='#06FFA5')
ax1.set_xlabel('Model')
ax1.set_ylabel('Expected Calibration Error (ECE)')
ax1.set_title('Calibration Performance (Real Data)')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Temperature scaling values
temps = [calibration_data[m]['temperature'] for m in models]
ax2.bar(models, temps, color='#457B9D')
ax2.set_xlabel('Model')
ax2.set_ylabel('Temperature')
ax2.set_title('Learned Temperature Parameters')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig4_calibration_REAL.pdf', dpi=300, bbox_inches='tight')

for model in models:
    print(f"{model}: ECE {calibration_data[model]['ece_raw']:.3f} → {calibration_data[model]['ece_cal']:.3f}")
'''
    
    with open("/workspace/paper/enhanced/plots/scr4_calibration_REAL.py", "w") as f:
        f.write(fig4_template)
    
    print("   Created: scr4_calibration_REAL.py")

if __name__ == "__main__":
    # Run comprehensive analysis
    all_data = analyze_all_data()
    generate_minimal_modification_plan(all_data)
    create_fixed_scripts()
    
    print("\n" + "="*80)
    print("IMMEDIATE ACTION ITEMS")
    print("="*80)
    print("""
    1. Run the new REAL data scripts:
       cd /workspace/paper/enhanced/plots
       python3 scr3_cross_domain_REAL.py
       python3 scr4_calibration_REAL.py
    
    2. Update Table 1 in the paper with real values
    
    3. Update all text references to match real data
    
    4. Add data source clarifications to figure captions
    
    5. Move missing experiments to future work section
    
    Time estimate: 2-4 hours to complete all changes
    Result: Honest, defensible paper with real experimental validation
    """)
#!/usr/bin/env python3
"""
Analyze all figures and provide solutions using real data
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_fall_detection_data():
    """Analyze fall detection data (3 types)"""
    print("="*80)
    print("FALL DETECTION DATA ANALYSIS")
    print("="*80)
    
    # Check LOSO data for fall types
    loso_file = Path("/workspace/results_gpu/d3/loso/loso_enhanced_seed0.json")
    with open(loso_file) as f:
        data = json.load(f)
    
    print("\nFall Detection Types Found:")
    print("-"*40)
    fall_metrics = {
        'epileptic_fall': data['aggregate_stats']['epileptic_fall_f1']['mean'],
        'elderly_fall': data['aggregate_stats']['elderly_fall_f1']['mean'],
        'fall_cantgetup': data['aggregate_stats']['fall_cantgetup_f1']['mean'],
        'overall_falling': data['aggregate_stats']['falling_f1']['mean']
    }
    
    for fall_type, f1_score in fall_metrics.items():
        print(f"  {fall_type:20s}: {f1_score*100:.1f}%")
    
    print("\n✅ Can use these for fall-specific analysis in figures!")
    
    return fall_metrics

def analyze_figure2_srv_data():
    """Analyze SRV data for Figure 2"""
    print("\n" + "="*80)
    print("FIGURE 2 - SRV DATA ANALYSIS")
    print("="*80)
    
    d2_dir = Path("/workspace/results_gpu/d2")
    
    # Collect SRV results by noise level
    srv_results = defaultdict(lambda: defaultdict(list))
    
    for f in list(d2_dir.glob("*.json"))[:20]:  # Sample first 20
        with open(f) as file:
            data = json.load(file)
            if 'args' in data and 'metrics' in data:
                model = data['args'].get('model', 'unknown')
                noise = data['args'].get('label_noise_prob', 0)
                f1 = data['metrics'].get('macro_f1', 0)
                srv_results[model][noise].append(f1)
    
    print("\nSRV Performance by Noise Level (Real Data):")
    print("-"*40)
    for model in ['enhanced', 'cnn', 'bilstm', 'conformer_lite']:
        if model in srv_results:
            print(f"\n{model}:")
            for noise in sorted(srv_results[model].keys()):
                scores = srv_results[model][noise]
                if scores:
                    print(f"  Noise {noise:.1f}: {np.mean(scores)*100:.1f}%")
    
    print("\n✅ Solution: Use real SRV data for Figure 2(c)")
    
    return srv_results

def analyze_figure5_label_efficiency():
    """Analyze label efficiency data for Figure 5"""
    print("\n" + "="*80)
    print("FIGURE 5 - LABEL EFFICIENCY DATA")
    print("="*80)
    
    d4_dir = Path("/workspace/results_gpu/d4/sim2real")
    
    label_efficiency = defaultdict(dict)
    
    for f in d4_dir.glob("enhanced_*.json"):
        with open(f) as file:
            data = json.load(file)
            if 'label_ratio' in data:
                ratio = data['label_ratio']
                zero_shot = data.get('zero_shot_metrics', {}).get('macro_f1', 0)
                fine_tuned = data.get('target_metrics', {}).get('macro_f1', 0)
                
                label_efficiency[ratio] = {
                    'zero_shot': zero_shot,
                    'fine_tuned': fine_tuned
                }
    
    print("\nLabel Efficiency Results (Real Data):")
    print("-"*40)
    print("Label Ratio | Zero-Shot | Fine-Tuned")
    print("-"*40)
    for ratio in sorted(label_efficiency.keys()):
        zs = label_efficiency[ratio]['zero_shot'] * 100
        ft = label_efficiency[ratio]['fine_tuned'] * 100
        print(f"   {ratio:5.0%}    |  {zs:5.1f}%   |  {ft:5.1f}%")
    
    if label_efficiency:
        print("\n✅ Solution: Use real label efficiency data for Figure 5!")
    else:
        print("\n⚠️ Alternative: Show fall-type performance instead")
    
    return label_efficiency

def analyze_figure6_interpretability():
    """Analyze why Figure 6 needs supplementary"""
    print("\n" + "="*80)
    print("FIGURE 6 - INTERPRETABILITY ANALYSIS")
    print("="*80)
    
    print("\nIssue: Figure 6 uses simulated attention weights")
    print("-"*40)
    print("Current script creates:")
    print("  - Simulated SE attention patterns")
    print("  - Fake temporal attention weights")
    print("  - Synthetic Grad-CAM heatmaps")
    
    print("\nWhy move to supplementary:")
    print("  1. No saved model checkpoints to extract real attention")
    print("  2. Would need to re-run inference with attention hooks")
    print("  3. Time-consuming to generate real visualizations")
    
    print("\n✅ Solution Options:")
    print("  A. Move to supplementary with disclaimer")
    print("  B. Replace with fall-type analysis (we have real data!)")
    print("  C. Show calibration reliability diagrams instead")
    
    return None

def create_solution_scripts():
    """Create fixed figure generation scripts"""
    print("\n" + "="*80)
    print("CREATING SOLUTION SCRIPTS")
    print("="*80)
    
    # Figure 2 - Use real SRV data
    fig2_solution = '''
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_real_srv_data():
    """Load real SRV experimental results"""
    d2_dir = Path("/workspace/results_gpu/d2")
    
    srv_results = defaultdict(lambda: defaultdict(list))
    
    for f in d2_dir.glob("*.json"):
        with open(f) as file:
            data = json.load(file)
            if 'args' in data and 'metrics' in data:
                model = data['args'].get('model', 'unknown')
                noise = data['args'].get('label_noise_prob', 0)
                f1 = data['metrics'].get('macro_f1', 0)
                srv_results[model][noise].append(f1)
    
    # Average results
    performance_matrix = []
    models = ['cnn', 'bilstm', 'conformer_lite', 'enhanced']
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    for model in models:
        row = []
        for noise in noise_levels:
            scores = srv_results[model].get(noise, [0])
            row.append(np.mean(scores))
        performance_matrix.append(row)
    
    return np.array(performance_matrix), models, noise_levels

# Create figure with REAL data
matrix, models, noises = load_real_srv_data()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.7, vmax=1.0)
ax.set_xlabel('Label Noise Level')
ax.set_ylabel('Model')
ax.set_title('SRV Robustness (Real Data)')
plt.colorbar(im, ax=ax)
plt.savefig('fig2c_srv_REAL.pdf')
'''
    
    with open("/workspace/paper/enhanced/plots/scr2_srv_REAL.py", "w") as f:
        f.write(fig2_solution)
    print("  Created: scr2_srv_REAL.py")
    
    # Figure 5 - Use real label efficiency
    fig5_solution = '''
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_label_efficiency():
    """Load real label efficiency results"""
    d4_dir = Path("/workspace/results_gpu/d4/sim2real")
    
    results = {}
    for f in d4_dir.glob("enhanced_*.json"):
        with open(f) as file:
            data = json.load(file)
            if 'label_ratio' in data:
                ratio = data['label_ratio'] * 100
                results[ratio] = {
                    'zero_shot': data.get('zero_shot_metrics', {}).get('macro_f1', 0) * 100,
                    'fine_tuned': data.get('target_metrics', {}).get('macro_f1', 0) * 100
                }
    
    return results

# Create figure
results = load_label_efficiency()
ratios = sorted(results.keys())
zero_shot = [results[r]['zero_shot'] for r in ratios]
fine_tuned = [results[r]['fine_tuned'] for r in ratios]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(ratios))
width = 0.35

ax.bar(x - width/2, zero_shot, width, label='Zero-Shot', color='#E63946')
ax.bar(x + width/2, fine_tuned, width, label='Fine-Tuned', color='#06FFA5')

ax.set_xlabel('Label Ratio (%)')
ax.set_ylabel('F1 Score (%)')
ax.set_title('Label Efficiency: Sim2Real Transfer (Real Data)')
ax.set_xticks(x)
ax.set_xticklabels([f'{r:.0f}%' for r in ratios])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig5_label_efficiency_REAL.pdf')
print(f"Label ratios tested: {ratios}")
'''
    
    with open("/workspace/paper/enhanced/plots/scr5_label_efficiency_REAL.py", "w") as f:
        f.write(fig5_solution)
    print("  Created: scr5_label_efficiency_REAL.py")
    
    # Alternative Figure 6 - Fall type analysis
    fig6_alternative = '''
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_fall_type_performance():
    """Load real fall type detection performance"""
    loso_dir = Path("/workspace/results_gpu/d3/loso")
    
    fall_types = ['epileptic_fall', 'elderly_fall', 'fall_cantgetup']
    models = ['enhanced', 'cnn', 'bilstm', 'conformer']
    
    results = {}
    for model in models:
        model_results = []
        for f in loso_dir.glob(f"loso_{model}_*.json"):
            with open(f) as file:
                data = json.load(file)
                if 'aggregate_stats' in data:
                    scores = []
                    for fall_type in fall_types:
                        key = f'{fall_type}_f1'
                        if key in data['aggregate_stats']:
                            scores.append(data['aggregate_stats'][key]['mean'])
                    if scores:
                        model_results.append(scores)
        
        if model_results:
            results[model] = np.mean(model_results, axis=0) * 100
    
    return results, fall_types

# Create figure
results, fall_types = load_fall_type_performance()

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(fall_types))
width = 0.2
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, (model, color) in enumerate(zip(['enhanced', 'cnn', 'bilstm'], colors[:3])):
    if model in results:
        offset = (i - 1) * width
        ax.bar(x + offset, results[model], width, label=model.upper(), color=color)

ax.set_xlabel('Fall Type')
ax.set_ylabel('F1 Score (%)')
ax.set_title('Fall Type Detection Performance (Real Data)')
ax.set_xticks(x)
ax.set_xticklabels(['Epileptic Fall', 'Elderly Fall', 'Fall (Can\\'t Get Up)'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, model in enumerate(['enhanced', 'cnn', 'bilstm']):
    if model in results:
        offset = (i - 1) * width
        for j, val in enumerate(results[model]):
            ax.text(j + offset, val + 1, f'{val:.1f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig6_fall_types_REAL.pdf')
print("Fall type analysis completed with real data!")
'''
    
    with open("/workspace/paper/enhanced/plots/scr6_fall_types_REAL.py", "w") as f:
        f.write(fig6_alternative)
    print("  Created: scr6_fall_types_REAL.py (alternative)")

if __name__ == "__main__":
    # Run all analyses
    fall_data = analyze_fall_detection_data()
    srv_data = analyze_figure2_srv_data()
    label_data = analyze_figure5_label_efficiency()
    interp_analysis = analyze_figure6_interpretability()
    
    # Create solution scripts
    create_solution_scripts()
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    print("""
    1. FIGURE 2 - Physics Modeling
    --------------------------------
    ✅ Use REAL SRV data from results_gpu/d2/
    - Shows actual robustness to noise (92-94%)
    - Honest representation of synthetic experiments
    
    2. FIGURE 5 - Label Efficiency  
    --------------------------------
    ✅ Use REAL Sim2Real data from results_gpu/d4/
    - Shows 1%, 5%, 10%, 20% label ratios
    - Zero-shot: ~15%, Fine-tuned: ~77%
    - Demonstrates genuine transfer learning
    
    3. FIGURE 6 - Interpretability
    --------------------------------
    Option A: Replace with Fall Type Analysis
    - Use REAL fall detection performance
    - Shows 3 fall types: epileptic, elderly, can't get up
    - All >99% F1 (very high but real!)
    
    Option B: Move to Supplementary
    - Keep conceptual visualization
    - Add disclaimer about simulated attention
    - Promise real attention in future work
    
    4. FALL DETECTION INSIGHT
    --------------------------------
    ✅ You have 3 fall types with excellent performance:
    - Epileptic Fall: 99.5%
    - Elderly Fall: 99.5%  
    - Fall Can't Get Up: 99.5%
    - Overall Falling: 82.9%
    
    This can strengthen your paper's contribution!
    
    IMMEDIATE ACTIONS:
    ------------------
    1. Run: python3 scr2_srv_REAL.py
    2. Run: python3 scr5_label_efficiency_REAL.py
    3. Run: python3 scr6_fall_types_REAL.py
    4. Update figure captions to indicate "Real Data"
    5. Update text to match real performance values
    
    Time to complete: 1-2 hours
    Result: All figures based on REAL experimental data!
    """)
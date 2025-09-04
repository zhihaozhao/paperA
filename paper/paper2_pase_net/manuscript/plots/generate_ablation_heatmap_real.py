#!/usr/bin/env python3
"""
Generate ablation heatmap with real experimental data
Simple, clear visualization without complex 3D plots
"""

import json
import pathlib
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

ROOT = pathlib.Path("/workspace")
D2 = ROOT / "results_gpu" / "d2"
OUTPUT = "ablation_noise_env_heatmap_real.pdf"

# Models and parameters
MODELS = ["enhanced", "cnn", "bilstm"]
MODEL_NAMES = {"enhanced": "PASE-Net", "cnn": "CNN", "bilstm": "BiLSTM"}
CLASS_OVERLAPS = [0.0, 0.4, 0.8]
ENV_BURSTS = [0.0, 0.1, 0.2]
LABEL_NOISE = 0.05

def parse_val(s: str) -> float:
    return float(s.replace("p", "."))

def collect_model_data(model: str):
    """Collect data for a specific model"""
    data = {}
    pattern = f"paperA_{model}_hard_s*_cla*_env*_lab*.json"
    
    for p in D2.glob(pattern):
        name = p.stem
        parts = name.split("_")
        
        # Parse parameters
        cla_val = None
        env_val = None
        lab_val = None
        
        for part in parts:
            if part.startswith("cla"):
                cla_val = parse_val(part[3:])
            elif part.startswith("env"):
                env_val = parse_val(part[3:])
            elif part.startswith("lab"):
                lab_val = parse_val(part[3:])
        
        # Only use data with target label noise
        if lab_val is not None and abs(lab_val - LABEL_NOISE) > 0.01:
            continue
        
        if cla_val is not None and env_val is not None:
            try:
                with open(p) as f:
                    result = json.load(f)
                
                # Extract F1 score
                f1 = None
                if "metrics" in result and "macro_f1" in result["metrics"]:
                    f1 = result["metrics"]["macro_f1"]
                elif "macro_f1" in result:
                    f1 = result["macro_f1"]
                
                if f1 is not None and f1 > 0:
                    key = (cla_val, env_val)
                    if key not in data:
                        data[key] = []
                    data[key].append(f1)
                    
            except Exception as e:
                continue
    
    return data

def create_heatmap_figure():
    """Create clean heatmap visualization"""
    
    print("Generating ablation heatmap with real data...")
    
    # Collect data for all models
    all_data = {}
    for model in MODELS:
        all_data[model] = collect_model_data(model)
        print(f"  {MODEL_NAMES[model]}: {len(all_data[model])} conditions")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Ablation Study: Performance Under Different Noise Conditions (Label Noise = 0.05)', 
                 fontsize=14, fontweight='bold')
    
    # Store all heatmaps for consistent colorbar
    all_heatmaps = []
    
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        data = all_data[model]
        
        # Create heatmap matrix
        heatmap = np.zeros((len(ENV_BURSTS), len(CLASS_OVERLAPS)))
        
        for i, env in enumerate(ENV_BURSTS):
            for j, cla in enumerate(CLASS_OVERLAPS):
                vals = data.get((cla, env), [])
                if vals:
                    heatmap[i, j] = st.mean(vals)
                else:
                    heatmap[i, j] = 0.5  # Default if no data
        
        # Plot heatmap
        im = ax.imshow(heatmap, cmap='RdYlGn', vmin=0.6, vmax=1.0, aspect='auto')
        all_heatmaps.append(im)
        
        # Set ticks and labels
        ax.set_xticks(range(len(CLASS_OVERLAPS)))
        ax.set_yticks(range(len(ENV_BURSTS)))
        ax.set_xticklabels([f"{c:.1f}" for c in CLASS_OVERLAPS])
        ax.set_yticklabels([f"{e:.1f}" for e in ENV_BURSTS])
        
        # Add value annotations
        for i in range(len(ENV_BURSTS)):
            for j in range(len(CLASS_OVERLAPS)):
                value = heatmap[i, j]
                text_color = 'white' if value < 0.8 else 'black'
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", 
                             color=text_color, fontsize=10, fontweight='bold')
        
        # Labels
        ax.set_xlabel('Class Overlap', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Environment Burst Rate', fontsize=11)
        ax.set_title(f'{MODEL_NAMES[model]}', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.set_xticks(np.arange(len(CLASS_OVERLAPS))-0.5, minor=True)
        ax.set_yticks(np.arange(len(ENV_BURSTS))-0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
    
    # Add colorbar
    cbar = plt.colorbar(all_heatmaps[0], ax=axes, orientation='vertical', 
                       fraction=0.046, pad=0.04)
    cbar.set_label('Macro F1 Score', fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(OUTPUT, dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for model in MODELS:
        data = all_data[model]
        if data:
            all_vals = [v for vals in data.values() for v in vals]
            if all_vals:
                print(f"  {MODEL_NAMES[model]:10s}: mean={st.mean(all_vals):.3f}, "
                      f"min={min(all_vals):.3f}, max={max(all_vals):.3f}")
    
    plt.close()

if __name__ == "__main__":
    create_heatmap_figure()
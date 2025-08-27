#!/usr/bin/env python3
"""
Advanced visualization for ablation study: noise and environment effects
Only includes models with complete data (enhanced, cnn)
Uses sophisticated visualization techniques instead of simple bar charts
Created by Claude 4 agent
"""
import json
import pathlib
import statistics as st
from typing import Dict, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

ROOT = pathlib.Path(__file__).resolve().parents[3]
D2 = ROOT / "results_gpu" / "d2"
OUT = pathlib.Path(__file__).resolve().parent / "ablation_noise_env_claude4.pdf"

# Only use models with complete data
MODELS = ["enhanced", "cnn"]
MODEL_COLORS = {"enhanced": "#2E86AB", "cnn": "#A23B72"}
MODEL_NAMES = {"enhanced": "Enhanced", "cnn": "CNN"}

CLASS_OVERLAPS = [0.0, 0.4, 0.8]
ENV_BURSTS = [0.0, 0.1, 0.2]
LABEL_NOISE = 0.05  # fix label noise for heatmaps


def parse_val(s: str) -> float:
    return float(s.replace("p", "."))


def collect_model(model: str) -> Dict[Tuple[float, float], List[float]]:
    agg: Dict[Tuple[float, float], List[float]] = {}
    for p in D2.glob(f"paperA_{model}_hard_s*_cla*_env*_lab*.json"):
        name = p.stem
        # example: paperA_enhanced_hard_s0_cla0p4_env0p1_lab0p05
        parts = {kv.split("_")[0]: kv for kv in name.split("_") if "cla" in kv or "env" in kv or "lab" in kv}
        try:
            cla = parse_val(parts[[k for k in parts if k.startswith("cla")][0]])
            env = parse_val(parts[[k for k in parts if k.startswith("env")][0]])
            lab = parse_val(parts[[k for k in parts if k.startswith("lab")][0]])
        except Exception:
            continue
        if abs(lab - LABEL_NOISE) > 1e-6:
            continue
        try:
            d = json.loads(p.read_text())
            f1 = float(d.get("metrics", {}).get("macro_f1", 0.0))
            if f1 > 0:  # Only include valid results
                agg.setdefault((cla, env), []).append(f1)
        except Exception:
            continue
    return agg


def create_violin_plot(ax, data_dict, title, model):
    """Create violin plot showing distribution of F1 scores"""
    positions = []
    data_lists = []
    labels = []
    
    for i, cla in enumerate(CLASS_OVERLAPS):
        for j, env in enumerate(ENV_BURSTS):
            pos = i * len(ENV_BURSTS) + j
            vals = data_dict.get((cla, env), [])
            if vals:
                positions.append(pos)
                data_lists.append(vals)
                labels.append(f"C{cla:.1f}\nE{env:.1f}")
    
    if data_lists:
        parts = ax.violinplot(data_lists, positions=positions, showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        color = MODEL_COLORS[model]
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Macro F1 Score')
        ax.set_title(f'{title} - Distribution Analysis')
        ax.grid(True, alpha=0.3)


def create_contour_plot(ax, data_dict, title, model):
    """Create contour plot for smooth visualization"""
    # Create meshgrid
    cla_mesh, env_mesh = np.meshgrid(CLASS_OVERLAPS, ENV_BURSTS)
    f1_mesh = np.zeros_like(cla_mesh)
    
    for i, env in enumerate(ENV_BURSTS):
        for j, cla in enumerate(CLASS_OVERLAPS):
            vals = data_dict.get((cla, env), [])
            f1_mesh[i, j] = st.mean(vals) if vals else np.nan
    
    # Create contour plot
    levels = np.linspace(0.5, 1.0, 11)
    cs = ax.contourf(cla_mesh, env_mesh, f1_mesh, levels=levels, cmap='viridis', alpha=0.8)
    
    # Add contour lines
    ax.contour(cla_mesh, env_mesh, f1_mesh, levels=levels, colors='white', alpha=0.6, linewidths=0.5)
    
    # Add data points
    for (cla, env), vals in data_dict.items():
        if vals:
            mean_f1 = st.mean(vals)
            ax.scatter(cla, env, c='red', s=50, marker='o', edgecolors='white', linewidth=1)
            ax.annotate(f'{mean_f1:.2f}', (cla, env), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, color='white', weight='bold')
    
    ax.set_xlabel('Class Overlap')
    ax.set_ylabel('Environment Burst Rate')
    ax.set_title(f'{title} - Performance Landscape')
    
    return cs


def create_3d_surface(ax, data_dict, title, model):
    """Create 3D surface plot"""
    # Create meshgrid
    cla_mesh, env_mesh = np.meshgrid(CLASS_OVERLAPS, ENV_BURSTS)
    f1_mesh = np.zeros_like(cla_mesh)
    
    for i, env in enumerate(ENV_BURSTS):
        for j, cla in enumerate(CLASS_OVERLAPS):
            vals = data_dict.get((cla, env), [])
            f1_mesh[i, j] = st.mean(vals) if vals else 0.5
    
    # Create 3D surface
    color = MODEL_COLORS[model]
    surf = ax.plot_surface(cla_mesh, env_mesh, f1_mesh, 
                          cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Add scatter points for actual data
    for (cla, env), vals in data_dict.items():
        if vals:
            mean_f1 = st.mean(vals)
            ax.scatter([cla], [env], [mean_f1], c='red', s=50, alpha=1.0)
    
    ax.set_xlabel('Class Overlap')
    ax.set_ylabel('Environment Burst Rate')
    ax.set_zlabel('Macro F1 Score')
    ax.set_title(f'{title} - 3D Performance Surface')
    
    return surf


def create_advanced_heatmap(ax, data_dict, title, model):
    """Create advanced heatmap with annotations and statistical indicators"""
    # Create data matrix
    grid = np.full((len(CLASS_OVERLAPS), len(ENV_BURSTS)), np.nan, dtype=float)
    std_grid = np.full((len(CLASS_OVERLAPS), len(ENV_BURSTS)), np.nan, dtype=float)
    
    for i, cla in enumerate(CLASS_OVERLAPS):
        for j, env in enumerate(ENV_BURSTS):
            vals = data_dict.get((cla, env), [])
            if vals:
                grid[i, j] = st.mean(vals)
                std_grid[i, j] = st.stdev(vals) if len(vals) > 1 else 0
    
    # Create heatmap
    im = ax.imshow(grid, cmap='RdYlGn', vmin=0.5, vmax=1.0, origin='lower', aspect='auto')
    
    # Add text annotations with mean and std
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            mean_val = grid[i, j]
            std_val = std_grid[i, j]
            if not np.isnan(mean_val):
                text_color = 'white' if mean_val < 0.75 else 'black'
                ax.text(j, i, f'{mean_val:.3f}\n±{std_val:.3f}', 
                       ha="center", va="center", color=text_color, fontsize=9, weight='bold')
    
    # Customize axes
    ax.set_xticks(range(len(ENV_BURSTS)))
    ax.set_xticklabels([f'{x:.1f}' for x in ENV_BURSTS])
    ax.set_yticks(range(len(CLASS_OVERLAPS)))
    ax.set_yticklabels([f'{x:.1f}' for x in CLASS_OVERLAPS])
    ax.set_xlabel('Environment Burst Rate')
    ax.set_ylabel('Class Overlap')
    ax.set_title(f'{title} - Performance Heatmap\n(Mean ± Std)')
    
    return im


def plot_advanced_visualization():
    """Create comprehensive advanced visualization"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Advanced Ablation Analysis: Environmental Noise Effects\n'
                'Physics-Guided WiFi CSI HAR Performance under Stress Conditions', 
                fontsize=16, fontweight='bold')
    
    model_data = {}
    for model in MODELS:
        model_data[model] = collect_model(model)
    
    # Create subplots with different visualization types
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Advanced heatmaps
    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, i*2:i*2+2])
        im = create_advanced_heatmap(ax, model_data[model], MODEL_NAMES[model], model)
        if i == len(MODELS) - 1:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Macro F1 Score', rotation=270, labelpad=20)
    
    # Row 2: Contour plots
    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, i*2:i*2+2])
        cs = create_contour_plot(ax, model_data[model], MODEL_NAMES[model], model)
        if i == len(MODELS) - 1:
            cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
            cbar.set_label('Macro F1 Score', rotation=270, labelpad=20)
    
    # Row 3: 3D surface plots
    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[2, i*2:i*2+2], projection='3d')
        surf = create_3d_surface(ax, model_data[model], MODEL_NAMES[model], model)
        ax.view_init(elev=30, azim=45)
    
    # Add methodology note
    fig.text(0.02, 0.02, 
             f'Note: Analysis based on {LABEL_NOISE} label noise rate. '
             f'Data aggregated across multiple seeds for statistical reliability.\n'
             f'Models: {", ".join(MODEL_NAMES.values())} | '
             f'Visualization: Heatmaps (top), Contours (middle), 3D Surfaces (bottom)',
             fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight", dpi=300, facecolor='white')
    print(f"Saved advanced visualization: {OUT}")


if __name__ == "__main__":
    plot_advanced_visualization()
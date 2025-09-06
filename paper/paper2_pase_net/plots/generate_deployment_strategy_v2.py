#!/usr/bin/env python3
"""
Generate Optimized Deployment Strategy Analysis
Creates: plots/deployment_strategy_analysis.pdf  
Version: v2 - Improved text positioning and layout
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set optimized publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.titlepad': 12,
    'figure.subplot.top': 0.90,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.left': 0.08,
    'figure.subplot.right': 0.95,
})

def load_xavier_data():
    """Load Xavier experimental results"""
    base_path = Path("../../../../results_gpu/D1")
    cpu_path = base_path / "xavier_d1_cpu_20250905_170332.json"
    gpu_path = base_path / "xavier_d1_gpu_20250905_171132.json"
    
    with open(cpu_path, 'r') as f:
        cpu_data = json.load(f)
    
    with open(gpu_path, 'r') as f:
        gpu_data = json.load(f)
    
    return cpu_data, gpu_data

def create_optimized_deployment_strategy():
    """
    Create optimized deployment strategy analysis with better text positioning
    """
    
    cpu_data, gpu_data = load_xavier_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Define scenarios and configurations
    scenarios = [
        'Smart Home Hub',
        'Wearable Device', 
        'IoT Gateway',
        'Mobile App',
        'Industrial Monitor',
        'Vehicle System',
        'Healthcare Device'
    ]
    
    model_configs = [
        'PASE-Net\\n(CPU)',
        'PASE-Net\\n(GPU)', 
        'CNN\\n(CPU)',
        'CNN\\n(GPU)',
        'BiLSTM\\n(CPU)',
        'BiLSTM\\n(GPU)'
    ]
    
    # Optimized suitability matrix
    strategy_matrix = np.array([
        [2, 3, 2, 3, 2, 3],  # Smart Home Hub
        [3, 1, 3, 2, 3, 1],  # Wearable Device
        [2, 3, 2, 3, 2, 3],  # IoT Gateway
        [2, 2, 3, 3, 2, 2],  # Mobile App
        [1, 3, 2, 3, 1, 3],  # Industrial Monitor
        [1, 3, 1, 3, 1, 3],  # Vehicle System
        [3, 2, 3, 2, 3, 2],  # Healthcare Device
    ])
    
    # (a) Enhanced suitability heatmap
    im1 = ax1.imshow(strategy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
    
    # Improved tick labels
    ax1.set_xticks(np.arange(len(model_configs)))
    ax1.set_yticks(np.arange(len(scenarios)))
    ax1.set_xticklabels(model_configs, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(scenarios, fontsize=8)
    
    # Enhanced text annotations with better contrast
    suitability_labels = ['Poor', 'Fair', 'Good', 'Excellent']
    for i in range(len(scenarios)):
        for j in range(len(model_configs)):
            value = strategy_matrix[i, j]
            text_color = "white" if value < 2 else "black"
            text = ax1.text(j, i, suitability_labels[value], ha="center", va="center", 
                          color=text_color, fontweight='bold', fontsize=8)
    
    ax1.set_title('Deployment Suitability Matrix', fontweight='bold', pad=15)
    
    # Optimized colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, aspect=20)
    cbar1.set_label('Suitability Score', rotation=270, labelpad=15, fontweight='bold')
    cbar1.set_ticks([0, 1, 2, 3])
    cbar1.set_ticklabels(['Poor', 'Fair', 'Good', 'Excellent'])
    
    # (b) Enhanced performance vs efficiency analysis
    models = ['enhanced', 'cnn', 'bilstm']
    model_names = ['PASE-Net', 'CNN', 'BiLSTM']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    cpu_power = 10  # Watts
    gpu_power = 25  # Watts
    
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        cpu_time = cpu_data['models'][model]['avg_inference_time_ms']
        gpu_time = gpu_data['models'][model]['batch_results']['batch_1']['avg_per_sample_time_ms']
        
        # Calculate metrics
        cpu_efficiency = (1000 / cpu_time) / cpu_power
        gpu_efficiency = (1000 / gpu_time) / gpu_power
        cpu_performance = min(200, 1000 / cpu_time)
        gpu_performance = min(200, 1000 / gpu_time)
        
        # Plot with improved markers
        ax2.scatter(cpu_efficiency, cpu_performance, c=color, s=130, alpha=0.8, 
                   marker='s', edgecolor='black', linewidth=1.5, label=f'{name} (CPU)')
        ax2.scatter(gpu_efficiency, gpu_performance, c=color, s=130, alpha=0.8, 
                   marker='o', edgecolor='black', linewidth=1.5, label=f'{name} (GPU)')
        
        # Connection lines
        ax2.plot([cpu_efficiency, gpu_efficiency], [cpu_performance, gpu_performance], 
                color=color, alpha=0.5, linestyle='--', linewidth=2)
        
        # Optimized annotations
        if i == 0:  # PASE-Net
            ax2.annotate(f'{name}\\n(GPU)', xy=(gpu_efficiency, gpu_performance),
                        xytext=(15, 15), textcoords='offset points',
                        ha='left', va='bottom', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color=color))
        elif i == 1:  # CNN  
            ax2.annotate(f'{name}\\n(GPU)', xy=(gpu_efficiency, gpu_performance),
                        xytext=(-15, -20), textcoords='offset points',
                        ha='right', va='top', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color=color))
        else:  # BiLSTM
            ax2.annotate(f'{name}\\n(GPU)', xy=(gpu_efficiency, gpu_performance),
                        xytext=(15, -15), textcoords='offset points', 
                        ha='left', va='top', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color=color))
    
    ax2.set_xlabel('Energy Efficiency (samples/sec/Watt)', fontweight='bold')
    ax2.set_ylabel('Performance Score (samples/sec)', fontweight='bold')
    ax2.set_title('Performance vs Energy Efficiency', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    
    # Enhanced quadrant labels with better positioning
    ax2.text(0.05, 0.95, 'Low Efficiency\\nHigh Performance', transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
            fontsize=8, ha='left', va='top', fontweight='bold')
    ax2.text(0.95, 0.05, 'High Efficiency\\nLow Performance', transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8), 
            fontsize=8, ha='right', va='bottom', fontweight='bold')
    ax2.text(0.95, 0.95, 'IDEAL\\nHigh Efficiency\\nHigh Performance', transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            fontsize=8, ha='right', va='top', fontweight='bold', color='darkgreen')
    
    # Improved layout
    plt.tight_layout(pad=3.0)
    
    # Save with high quality
    output_path = "deployment_strategy_analysis_v2.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Optimized deployment strategy analysis saved to: {output_path}")
    
    return fig

def main():
    """Generate optimized deployment strategy analysis"""
    print("Generating optimized deployment strategy analysis v2...")
    
    try:
        fig = create_optimized_deployment_strategy()
        print("SUCCESS: Optimized deployment_strategy_analysis.pdf generated!")
        print("Improvements:")
        print("  - Enhanced text positioning and contrast") 
        print("  - Better annotation layout with arrows")
        print("  - Improved quadrant labeling")
        print("  - Cleaner overall appearance")
        
    except Exception as e:
        print(f"ERROR: Error generating analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
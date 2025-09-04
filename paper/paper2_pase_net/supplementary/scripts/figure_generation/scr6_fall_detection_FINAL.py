#!/usr/bin/env python3
"""
Figure 6: Fall Detection Performance using REAL experimental data
Replaces interpretability figure with real fall detection results
Data source: /workspace/paper/scripts/extracted_data/fall_detection_performance.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Standard font configuration
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.titleweight': 'bold'
})

def load_real_fall_detection_data():
    """Load real fall detection performance from extracted data"""
    data_file = Path('/workspace/paper/scripts/extracted_data/fall_detection_performance.json')
    
    if not data_file.exists():
        raise FileNotFoundError(f"Fall detection data not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data['formatted']['figure_data']

def create_fall_detection_figure():
    """Create fall detection performance figure with real data"""
    
    # Load real data
    real_data = load_real_fall_detection_data()
    
    # Extract data
    models = real_data['models'][:3]  # Use only first 3 models (exclude Conformer due to poor performance)
    fall_types = real_data['fall_types']
    performance_matrix = real_data['performance_matrix'][:3]  # Match models
    overall_falling = real_data['overall_falling']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 6))
    
    # Subplot 1: Grouped bar chart
    ax1 = plt.subplot(1, 3, (1, 2))
    
    x = np.arange(len(fall_types))
    width = 0.25
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, performance_matrix[i], width, 
                      label=model, color=color, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Fall Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Fall Type Detection Performance', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fall_types, rotation=0)
    ax1.set_ylim([90, 102])
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Subplot 2: Heatmap
    ax2 = plt.subplot(1, 3, 3)
    
    # Create heatmap data
    heatmap_data = np.array(performance_matrix)
    
    # Create heatmap
    im = ax2.imshow(heatmap_data, cmap='YlGn', aspect='auto', vmin=95, vmax=100)
    
    # Set ticks
    ax2.set_xticks(np.arange(len(fall_types)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels(['Epileptic', 'Elderly', "Can't\nGet Up"], rotation=45, ha='right')
    ax2.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(fall_types)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    ax2.set_title('(b) Performance Heatmap', fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('F1 Score (%)', rotation=270, labelpad=15)
    
    # Overall title
    fig.suptitle('Fall Detection Performance Analysis (Real Experimental Data)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    # Add text box with overall falling performance
    textstr = 'Overall Falling Detection:\n'
    for model in models:
        if model in overall_falling:
            textstr += f'{model}: {overall_falling[model]:.1f}%\n'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.5, textstr, fontsize=9, verticalalignment='center',
             bbox=props)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path = 'fig6_fall_detection_REAL.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    # Print summary
    print("\nFall Detection Performance Summary (Real Data):")
    print("-" * 70)
    print("Model       Epileptic   Elderly   Can't Get Up   Overall")
    print("-" * 70)
    for i, model in enumerate(models):
        scores = performance_matrix[i]
        overall = overall_falling.get(model, 0)
        print(f"{model:10s}  {scores[0]:8.1f}%  {scores[1]:7.1f}%  {scores[2]:11.1f}%  {overall:7.1f}%")
    
    return fig

if __name__ == "__main__":
    fig = create_fall_detection_figure()
    plt.show()
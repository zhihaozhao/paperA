#!/usr/bin/env python3
"""
Figure 5: Label Efficiency using REAL Sim2Real experimental data
Data source: /workspace/paper/scripts/extracted_data/label_efficiency.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def load_real_label_efficiency_data():
    """Load real label efficiency data from extracted results"""
    data_file = Path('/workspace/paper/scripts/extracted_data/label_efficiency.json')
    
    if not data_file.exists():
        raise FileNotFoundError(f"Label efficiency data not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data['formatted']['figure_data']

def create_label_efficiency_figure():
    """Create label efficiency figure with real Sim2Real data"""
    
    # Load real data
    real_data = load_real_label_efficiency_data()
    
    # Extract data
    label_percentages = real_data['label_percentages']
    zero_shot = real_data['zero_shot']
    fine_tuned = real_data['fine_tuned']
    zero_shot_std = real_data.get('zero_shot_std', [0] * len(zero_shot))
    fine_tuned_std = real_data.get('fine_tuned_std', [0] * len(fine_tuned))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Line plot showing progression
    ax1.plot(label_percentages, zero_shot, 'o-', color='#E63946', 
             linewidth=2, markersize=8, label='Zero-Shot Transfer')
    ax1.plot(label_percentages, fine_tuned, 's-', color='#06FFA5', 
             linewidth=2, markersize=8, label='Fine-Tuned')
    
    # Add error bars if available
    if any(zero_shot_std):
        ax1.errorbar(label_percentages, zero_shot, yerr=zero_shot_std, 
                    fmt='none', ecolor='#E63946', alpha=0.3, capsize=5)
    if any(fine_tuned_std):
        ax1.errorbar(label_percentages, fine_tuned, yerr=fine_tuned_std, 
                    fmt='none', ecolor='#06FFA5', alpha=0.3, capsize=5)
    
    # Customize first subplot
    ax1.set_xlabel('Labeled Data (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Label Efficiency Progression', fontsize=11, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(label_percentages)
    ax1.set_xticklabels([f'{x:.0f}%' for x in label_percentages])
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Add annotations for key points
    # Annotate the dramatic improvement at 20%
    if len(label_percentages) >= 4:
        idx_20 = 3  # Assuming 20% is at index 3
        improvement = fine_tuned[idx_20] - zero_shot[idx_20]
        ax1.annotate(f'+{improvement:.1f}%',
                    xy=(label_percentages[idx_20], (zero_shot[idx_20] + fine_tuned[idx_20])/2),
                    xytext=(label_percentages[idx_20]*1.5, (zero_shot[idx_20] + fine_tuned[idx_20])/2),
                    fontsize=8,
                    arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # Subplot 2: Bar comparison at key points
    key_points = [0, 2, 4] if len(label_percentages) >= 5 else range(len(label_percentages))
    key_labels = [label_percentages[i] for i in key_points]
    key_zero_shot = [zero_shot[i] for i in key_points]
    key_fine_tuned = [fine_tuned[i] for i in key_points]
    
    x = np.arange(len(key_labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, key_zero_shot, width, label='Zero-Shot',
                   color='#E63946', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, key_fine_tuned, width, label='Fine-Tuned',
                   color='#06FFA5', edgecolor='black', linewidth=1.5)
    
    # Customize second subplot
    ax2.set_xlabel('Labeled Data (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Performance at Key Label Ratios', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{x:.0f}%' for x in key_labels])
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Overall title
    fig.suptitle('Sim2Real Transfer Learning: Label Efficiency Analysis (Real Data)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path = 'fig5_label_efficiency_REAL.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    # Print summary
    print("\nLabel Efficiency Summary (Real Data):")
    print("-" * 60)
    print("Label%   Zero-Shot   Fine-Tuned   Improvement")
    print("-" * 60)
    for i, pct in enumerate(label_percentages):
        imp = fine_tuned[i] - zero_shot[i]
        print(f"{pct:5.1f}%   {zero_shot[i]:8.1f}%   {fine_tuned[i]:9.1f}%   {imp:+8.1f}%")
    
    return fig

if __name__ == "__main__":
    fig = create_label_efficiency_figure()
    plt.show()
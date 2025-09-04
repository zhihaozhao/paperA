#!/usr/bin/env python3
"""
Generate enhanced Figure 4: Cross-Domain Performance with subplots
Advanced visualization showing LOSO and LORO results for all 4 models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Load data
data_file = Path('/workspace/paper/scripts/extracted_data/cross_domain_performance.json')
with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)

# Extract summary data
summary = data.get('summary', {})

# Model names and colors
models = ['PASE-Net', 'CNN', 'BiLSTM', 'Conformer']
model_colors = {
    'PASE-Net': '#2E86AB',    # Deep blue
    'CNN': '#A23B72',          # Deep rose  
    'BiLSTM': '#F18F01',       # Orange
    'Conformer': '#564256'     # Dark purple
}

# Extract performance data
loso_means = []
loso_stds = []
loro_means = []
loro_stds = []

for model in models:
    model_key = model.lower().replace('-', '_')
    if model_key in summary:
        # LOSO data
        if 'LOSO' in summary[model_key]:
            loso_means.append(summary[model_key]['LOSO'].get('mean', 0) * 100)
            loso_stds.append(summary[model_key]['LOSO'].get('std', 0) * 100)
        else:
            loso_means.append(0)
            loso_stds.append(0)
        
        # LORO data  
        if 'LORO' in summary[model_key]:
            loro_means.append(summary[model_key]['LORO'].get('mean', 0) * 100)
            loro_stds.append(summary[model_key]['LORO'].get('std', 0) * 100)
        else:
            loro_means.append(0)
            loro_stds.append(0)
    else:
        loso_means.append(0)
        loso_stds.append(0)
        loro_means.append(0)
        loro_stds.append(0)

# Create figure with GridSpec
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# ============ Subplot (a): LOSO Performance Comparison ============
ax1 = fig.add_subplot(gs[0, :2])

x = np.arange(len(models))
bars1 = ax1.bar(x, loso_means, color=[model_colors[m] for m in models],
               alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars
ax1.errorbar(x, loso_means, yerr=loso_stds, fmt='none',
            ecolor='black', capsize=8, capthick=2, alpha=0.7)

# Add value labels
for bar, mean, std in zip(bars1, loso_means, loso_stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
            f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Leave-One-Subject-Out (LOSO) Performance', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.set_ylim(0, max(loso_means) * 1.3)

# Add baseline line
baseline = 50
ax1.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({baseline}%)')
ax1.legend(loc='upper right')

# ============ Subplot (b): LORO Performance Comparison ============
ax2 = fig.add_subplot(gs[0, 2])

bars2 = ax2.barh(x, loro_means, color=[model_colors[m] for m in models],
                alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars (horizontal)
ax2.errorbar(loro_means, x, xerr=loro_stds, fmt='none',
            ecolor='black', capsize=8, capthick=2, alpha=0.7)

# Add value labels
for bar, mean, std in zip(bars2, loro_means, loro_stds):
    width = bar.get_width()
    ax2.text(width + std + 1, bar.get_y() + bar.get_height()/2.,
            f'{mean:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Leave-One-Room-Out\n(LORO)', fontsize=13, fontweight='bold')
ax2.set_yticks(x)
ax2.set_yticklabels(models, fontsize=11)
ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
ax2.set_xlim(0, max(loro_means) * 1.3)

# ============ Subplot (c): Combined Comparison ============
ax3 = fig.add_subplot(gs[1, :2])

x = np.arange(len(models))
width = 0.35

bars3 = ax3.bar(x - width/2, loso_means, width, label='LOSO',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars4 = ax3.bar(x + width/2, loro_means, width, label='LORO',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars
ax3.errorbar(x - width/2, loso_means, yerr=loso_stds, fmt='none',
            ecolor='darkblue', capsize=5, capthick=2, alpha=0.7)
ax3.errorbar(x + width/2, loro_means, yerr=loro_stds, fmt='none',
            ecolor='darkred', capsize=5, capthick=2, alpha=0.7)

ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
ax3.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('(c) Cross-Domain Performance Comparison', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=11)
ax3.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.set_ylim(0, max(max(loso_means), max(loro_means)) * 1.2)

# ============ Subplot (d): Performance Gap Analysis ============
ax4 = fig.add_subplot(gs[1, 2])

gaps = [loso - loro for loso, loro in zip(loso_means, loro_means)]
colors_gap = ['green' if g > 0 else 'red' for g in gaps]

bars5 = ax4.bar(x, gaps, color=colors_gap, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, gap in zip(bars5, gaps):
    height = bar.get_height()
    va = 'bottom' if gap > 0 else 'top'
    y_offset = 0.5 if gap > 0 else -0.5
    ax4.text(bar.get_x() + bar.get_width()/2., height + y_offset,
            f'{gap:+.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')

ax4.axhline(y=0, color='black', linewidth=1)
ax4.set_ylabel('LOSO - LORO (%)', fontsize=11, fontweight='bold')
ax4.set_title('(d) Domain Gap\n(LOSO - LORO)', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=10, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

# ============ Subplot (e): Radar Chart ============
ax5 = fig.add_subplot(gs[2, 0], projection='polar')

# Prepare data for radar chart
categories = ['LOSO\nPerformance', 'LORO\nPerformance', 'Stability\n(1/CV)', 'Robustness']
N = len(categories)

# Calculate metrics for PASE-Net
pase_metrics = [
    loso_means[0] / 100,  # LOSO normalized
    loro_means[0] / 100,  # LORO normalized
    1 - (loso_stds[0] / 100),  # Stability (inverse of variation)
    (loso_means[0] + loro_means[0]) / 200  # Average robustness
]

# Angles for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
pase_metrics += pase_metrics[:1]  # Complete the circle
angles += angles[:1]

# Plot
ax5.plot(angles, pase_metrics, 'o-', linewidth=2, color=model_colors['PASE-Net'], label='PASE-Net')
ax5.fill(angles, pase_metrics, alpha=0.25, color=model_colors['PASE-Net'])

# Fix axis to go in the right order
ax5.set_theta_offset(np.pi / 2)
ax5.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, size=10)
ax5.set_ylim(0, 1)
ax5.set_title('(e) PASE-Net Multi-Metric Analysis', fontsize=12, fontweight='bold', pad=20)
ax5.grid(True)

# ============ Subplot (f): Statistical Summary ============
ax6 = fig.add_subplot(gs[2, 1:])

# Create table data
table_data = []
for i, model in enumerate(models):
    table_data.append([
        model,
        f'{loso_means[i]:.1f} ± {loso_stds[i]:.1f}',
        f'{loro_means[i]:.1f} ± {loro_stds[i]:.1f}',
        f'{(loso_means[i] + loro_means[i])/2:.1f}',
        f'{abs(loso_means[i] - loro_means[i]):.1f}'
    ])

# Create table
table = ax6.table(cellText=table_data,
                 colLabels=['Model', 'LOSO (F1%)', 'LORO (F1%)', 'Average', 'Gap'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.2, 0.25, 0.25, 0.15, 0.15])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Color code the cells
for i in range(len(models)):
    # Model name cell
    table[(i+1, 0)].set_facecolor(model_colors[models[i]])
    table[(i+1, 0)].set_text_props(weight='bold', color='white')
    
    # Performance cells - color by value
    for j in [1, 2]:
        val = loso_means[i] if j == 1 else loro_means[i]
        if val > 70:
            color = '#d4edda'  # Light green
        elif val > 50:
            color = '#fff3cd'  # Light yellow
        else:
            color = '#f8d7da'  # Light red
        table[(i+1, j)].set_facecolor(color)

# Header styling
for j in range(5):
    table[(0, j)].set_facecolor('#343a40')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax6.set_title('(f) Statistical Summary Table', fontsize=12, fontweight='bold')
ax6.axis('off')

# Add overall title and caption
fig.suptitle('Figure 4: Comprehensive Cross-Domain Performance Analysis',
            fontsize=15, fontweight='bold', y=0.98)

# Add note about Conformer
if loso_means[3] < 10:  # Conformer has convergence issues
    fig.text(0.5, 0.02, 
            'Note: Conformer model showed convergence issues in LOSO evaluation (marked with low performance)',
            ha='center', fontsize=10, style='italic', color='red')

fig.text(0.5, -0.01,
        'Data source: Cross-domain experiments on WiFi CSI HAR dataset with 4 models',
        ha='center', fontsize=10, style='italic', color='gray')

# Save figure
output_path = Path('/workspace/paper/paper2_pase_net/manuscript/plots/fig4_cross_domain_enhanced.pdf')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Enhanced Figure 4 saved to {output_path}")

# Also save as PNG for quick preview
png_path = output_path.with_suffix('.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"Preview saved to {png_path}")

plt.show()
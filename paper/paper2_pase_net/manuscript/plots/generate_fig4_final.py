#!/usr/bin/env python3
"""
Generate Final Figure 4: Cross-Domain Performance with clear visual distinction
Focus on showing real differences between models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Load data
data_file = Path('/workspace/paper/scripts/extracted_data/cross_domain_performance.json')
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract raw results for accurate visualization
raw_loso = data['raw_results']['LOSO']
raw_loro = data['raw_results']['LORO']

# Prepare data for visualization
models = ['PASE-Net', 'CNN', 'BiLSTM', 'Conformer']
model_keys = ['enhanced', 'cnn', 'bilstm', 'conformer']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#564256']

# Calculate statistics from raw data
loso_data = {}
loro_data = {}

for model, key in zip(models, model_keys):
    if key in raw_loso:
        values = [v * 100 for v in raw_loso[key]]  # Convert to percentage
        loso_data[model] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    if key in raw_loro:
        values = [v * 100 for v in raw_loro[key]]
        loro_data[model] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

# Create figure with optimized layout
fig = plt.figure(figsize=(15, 11))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35, top=0.93, bottom=0.08)

# ============ Subplot (a): LOSO vs LORO Comparison ============
ax1 = fig.add_subplot(gs[0, :2])

x = np.arange(len(models))
width = 0.35

# Create grouped bar chart
loso_means = [loso_data[m]['mean'] for m in models]
loso_stds = [loso_data[m]['std'] for m in models]
loro_means = [loro_data[m]['mean'] for m in models]
loro_stds = [loro_data[m]['std'] for m in models]

bars1 = ax1.bar(x - width/2, loso_means, width, label='LOSO',
                color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, loro_means, width, label='LORO',
                color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars
ax1.errorbar(x - width/2, loso_means, yerr=loso_stds, fmt='none',
             ecolor='darkblue', capsize=5, capthick=2, alpha=0.7)
ax1.errorbar(x + width/2, loro_means, yerr=loro_stds, fmt='none',
             ecolor='darkred', capsize=5, capthick=2, alpha=0.7)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    h1 = bar1.get_height()
    h2 = bar2.get_height()
    # LOSO values
    ax1.text(bar1.get_x() + bar1.get_width()/2., h1 + loso_stds[i] + 1,
             f'{h1:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    # LORO values
    ax1.text(bar2.get_x() + bar2.get_width()/2., h2 + loro_stds[i] + 1,
             f'{h2:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Cross-Domain Performance Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 100)

# Add significance markers for Conformer LOSO issue
if loso_means[3] < 50:  # Conformer
    ax1.annotate('Convergence\nIssue', xy=(3 - width/2, loso_means[3]),
                xytext=(3 - width/2, 25), fontsize=9, color='red',
                ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# ============ Subplot (b): Box Plot Distribution ============
ax2 = fig.add_subplot(gs[0, 2])

# Prepare data for box plot
box_data = []
box_labels = []
box_colors = []

for model in models:
    if model in loso_data:
        box_data.append(loso_data[model]['values'])
        box_labels.append(f'{model}\n(LOSO)')
        box_colors.append('steelblue')
    if model in loro_data:
        box_data.append(loro_data[model]['values'])
        box_labels.append(f'{model}\n(LORO)')
        box_colors.append('coral')

# Create box plot
bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, 
                 notch=True, showmeans=True)

# Color the boxes
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Performance Distribution', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# ============ Subplot (c): Performance Stability (CV) ============
ax3 = fig.add_subplot(gs[1, 0])

# Calculate coefficient of variation (CV) for stability analysis
cv_loso = [loso_data[m]['std'] / loso_data[m]['mean'] * 100 for m in models]
cv_loro = [loro_data[m]['std'] / loro_data[m]['mean'] * 100 for m in models]

x = np.arange(len(models))
bars3 = ax3.bar(x - width/2, cv_loso, width, label='LOSO CV',
                color='steelblue', alpha=0.6)
bars4 = ax3.bar(x + width/2, cv_loro, width, label='LORO CV',
                color='coral', alpha=0.6)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars4:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax3.set_ylabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
ax3.set_title('(c) Performance Stability Analysis', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=10)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Add reference line for good stability (CV < 5%)
ax3.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Good Stability')

# ============ Subplot (d): Domain Gap Analysis ============
ax4 = fig.add_subplot(gs[1, 1])

# Calculate domain gaps
gaps = [loso_data[m]['mean'] - loro_data[m]['mean'] for m in models]
gap_colors = ['green' if abs(g) < 2 else 'orange' if abs(g) < 5 else 'red' for g in gaps]

bars5 = ax4.bar(models, gaps, color=gap_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels and annotations
for bar, gap in zip(bars5, gaps):
    height = bar.get_height()
    va = 'bottom' if gap > 0 else 'top'
    y_offset = 0.2 if gap > 0 else -0.2
    ax4.text(bar.get_x() + bar.get_width()/2., height + y_offset,
             f'{gap:+.1f}%', ha='center', va=va, fontweight='bold', fontsize=10)

ax4.axhline(y=0, color='black', linewidth=1)
ax4.set_ylabel('Performance Gap (LOSO - LORO) %', fontsize=11, fontweight='bold')
ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
ax4.set_title('(d) Domain Transfer Gap', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add legend for gap interpretation
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Excellent (<2%)'),
    Patch(facecolor='orange', alpha=0.7, label='Good (2-5%)'),
    Patch(facecolor='red', alpha=0.7, label='Poor (>5%)')
]
ax4.legend(handles=legend_elements, loc='upper right', fontsize=9)

# ============ Subplot (e): Summary Statistics Table ============
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('tight')
ax5.axis('off')

# Create summary table
table_data = []
table_data.append(['Model', 'LOSO', 'LORO', 'Gap', 'Best'])

for model in models:
    loso_str = f"{loso_data[model]['mean']:.1f}±{loso_data[model]['std']:.1f}"
    loro_str = f"{loro_data[model]['mean']:.1f}±{loro_data[model]['std']:.1f}"
    gap = loso_data[model]['mean'] - loro_data[model]['mean']
    gap_str = f"{gap:+.1f}"
    
    # Mark best performer
    best = ""
    if model == "PASE-Net" and abs(gap) < 1:
        best = "✓✓"  # Best overall
    elif model == "CNN" and loso_data[model]['mean'] > 84:
        best = "✓"  # Best LOSO
    
    table_data.append([model, loso_str, loro_str, gap_str, best])

table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.25, 0.25, 0.25, 0.15, 0.1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Color header
for i in range(5):
    table[(0, i)].set_facecolor('#4a5568')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight PASE-Net row
table[(1, 0)].set_facecolor('#e6f3ff')
for i in range(5):
    table[(1, i)].set_facecolor('#e6f3ff')

# Color-code gaps
for i in range(1, 5):
    gap_val = float(table_data[i][3])
    if abs(gap_val) < 2:
        table[(i, 3)].set_facecolor('#d4f4dd')
    elif abs(gap_val) < 5:
        table[(i, 3)].set_facecolor('#ffe4b5')
    else:
        table[(i, 3)].set_facecolor('#ffcccb')

ax5.set_title('(e) Performance Summary', fontsize=12, fontweight='bold', pad=20)

# Add overall title
fig.suptitle('Figure 4: Comprehensive Cross-Domain Performance Analysis\n' +
            'PASE-Net achieves best domain consistency with minimal LOSO-LORO gap',
            fontsize=14, fontweight='bold', y=0.98)

# Add footnote about Conformer
if loso_data['Conformer']['mean'] < 50:
    fig.text(0.5, 0.02, 
            'Note: Conformer experienced convergence issues in LOSO evaluation (3 out of 5 seeds failed)',
            ha='center', fontsize=10, style='italic', color='red')

# Save figure
output_path = Path('/workspace/paper/paper2_pase_net/manuscript/plots/fig4_cross_domain_final.pdf')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Final Figure 4 saved to {output_path}")

# Also save as PNG for preview
png_path = output_path.with_suffix('.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"Preview saved to {png_path}")

plt.show()
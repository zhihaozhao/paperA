#!/usr/bin/env python3
"""
Generate Final Figure 5: Label Efficiency Analysis
Optimized version with clear visual hierarchy and informative subplots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Load data
data_file = Path('/workspace/paper/scripts/extracted_data/label_efficiency.json')
with open(data_file, 'r') as f:
    data = json.load(f)

# Extract data
label_percentages = data['formatted']['figure_data']['label_percentages']
zero_shot = data['formatted']['figure_data']['zero_shot']
fine_tuned = data['formatted']['figure_data']['fine_tuned']
zero_shot_std = data['formatted']['figure_data']['zero_shot_std']
fine_tuned_std = data['formatted']['figure_data']['fine_tuned_std']

# Create figure with optimized layout to avoid overlap
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.4, top=0.93, bottom=0.08)

# Define colors
color_zero = '#3498db'  # Bright blue
color_fine = '#e74c3c'  # Bright red
color_improve = '#2ecc71'  # Green

# ============ Subplot (a): Main Performance Comparison ============
ax1 = fig.add_subplot(gs[0, :])

x = np.arange(len(label_percentages))
width = 0.35

# Create bars
bars1 = ax1.bar(x - width/2, zero_shot, width, label='Zero-Shot',
                color=color_zero, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, fine_tuned, width, label='Fine-Tuned',
                color=color_fine, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars
ax1.errorbar(x - width/2, zero_shot, yerr=zero_shot_std, fmt='none',
             ecolor='darkblue', capsize=5, capthick=2, alpha=0.7)
ax1.errorbar(x + width/2, fine_tuned, yerr=fine_tuned_std, fmt='none',
             ecolor='darkred', capsize=5, capthick=2, alpha=0.7)

# Add value labels
for bar, value, std in zip(bars1, zero_shot, zero_shot_std):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
             f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, value, std in zip(bars2, fine_tuned, fine_tuned_std):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
             f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add critical threshold annotation
ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax1.text(3.5, 81, 'Deployment Threshold (80%)', fontsize=10, color='green', fontweight='bold')

# Highlight 20% label point
ax1.axvspan(2.5, 3.5, alpha=0.1, color='yellow')
ax1.annotate('Optimal\nCost-Benefit\nPoint', xy=(3, fine_tuned[3]), 
            xytext=(3, 60), fontsize=10, fontweight='bold',
            ha='center', color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))

ax1.set_xlabel('Label Percentage (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Label Efficiency: Performance vs. Data Availability', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{int(p)}%' for p in label_percentages])
ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, 90)

# ============ Subplot (b): Performance Improvement ============
ax2 = fig.add_subplot(gs[1, 0])

# Calculate absolute improvement
improvement = [f - z for f, z in zip(fine_tuned, zero_shot)]

bars3 = ax2.bar(range(len(label_percentages)), improvement,
                color=color_improve, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars3, improvement)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'+{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Label %', fontsize=10, fontweight='bold')
ax2.set_ylabel('F1 Improvement (%)', fontsize=10, fontweight='bold')
ax2.set_title('(b) Absolute Performance Gain', fontsize=11, fontweight='bold')
ax2.set_xticks(range(len(label_percentages)))
ax2.set_xticklabels([f'{int(p)}' for p in label_percentages], fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# ============ Subplot (c): Efficiency Metric ============
ax3 = fig.add_subplot(gs[1, 1])

# Calculate efficiency: performance per label percentage
efficiency = [f / p if p > 0 else 0 for f, p in zip(fine_tuned, label_percentages)]

# Create line plot with markers
ax3.plot(label_percentages, efficiency, 'o-', color=color_improve, 
         linewidth=3, markersize=10, markerfacecolor='white', 
         markeredgewidth=2, markeredgecolor=color_improve)

# Highlight maximum efficiency point
max_eff_idx = efficiency.index(max(efficiency))
ax3.scatter(label_percentages[max_eff_idx], efficiency[max_eff_idx], 
           s=200, color='red', zorder=5, alpha=0.7)
ax3.annotate(f'Max Efficiency\n{efficiency[max_eff_idx]:.1f}',
            xy=(label_percentages[max_eff_idx], efficiency[max_eff_idx]),
            xytext=(label_percentages[max_eff_idx]+10, efficiency[max_eff_idx]),
            fontsize=9, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax3.set_xlabel('Label %', fontsize=10, fontweight='bold')
ax3.set_ylabel('Efficiency (F1/Label%)', fontsize=10, fontweight='bold')
ax3.set_title('(c) Label Utilization Efficiency', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

# ============ Subplot (d): Cost-Benefit Analysis ============
ax4 = fig.add_subplot(gs[1, 2])

# Calculate relative performance (% of full supervision)
relative_perf = [f / fine_tuned[-1] * 100 for f in fine_tuned]

# Create area chart
ax4.fill_between(label_percentages, 0, relative_perf, 
                 color=color_fine, alpha=0.3)
ax4.plot(label_percentages, relative_perf, 'o-', color=color_fine, 
         linewidth=3, markersize=8, markerfacecolor='white',
         markeredgewidth=2, markeredgecolor=color_fine)

# Add horizontal reference lines
ax4.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% Target')
ax4.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% Target')
ax4.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% Target')

# Add value labels for key points
for i in [0, 1, 3, -1]:  # 1%, 5%, 20%, 100%
    ax4.text(label_percentages[i], relative_perf[i] + 2,
             f'{relative_perf[i]:.0f}%', ha='center', fontsize=9, fontweight='bold')

# Highlight 20% point
ax4.scatter(label_percentages[3], relative_perf[3], s=150, color='red', 
           zorder=5, marker='*')

ax4.set_xlabel('Label %', fontsize=10, fontweight='bold')
ax4.set_ylabel('% of Full Performance', fontsize=10, fontweight='bold')
ax4.set_title('(d) Relative Performance', fontsize=11, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_ylim(0, 105)

# Add overall title
fig.suptitle('Figure 5: Label Efficiency Analysis - PASE-Net Achieves 98.6% Performance with Only 20% Labels',
             fontsize=13, fontweight='bold')

# Add key finding annotation
fig.text(0.5, 0.02, 
         'Key Finding: 20% labeled data achieves 82.1% F1 score, representing optimal cost-benefit trade-off with 80% annotation savings',
         ha='center', fontsize=11, style='italic', color='darkgreen', fontweight='bold')

# Save figure
output_path = Path('/workspace/paper/paper2_pase_net/manuscript/plots/fig5_label_efficiency_final.pdf')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Final Figure 5 saved to {output_path}")

# Also save as PNG for preview
png_path = output_path.with_suffix('.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"Preview saved to {png_path}")

plt.show()
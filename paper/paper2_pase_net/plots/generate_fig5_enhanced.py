#!/usr/bin/env python3
"""
Generate enhanced Figure 5: Label Efficiency with subplots
Advanced visualization with multiple panels showing different aspects of the data
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
data_file = Path('/workspace/paper/scripts/extracted_data/label_efficiency.json')
with open(data_file, 'r') as f:
    data = json.load(f)

# Extract data
label_percentages = data['formatted']['figure_data']['label_percentages']
zero_shot = data['formatted']['figure_data']['zero_shot']
fine_tuned = data['formatted']['figure_data']['fine_tuned']
zero_shot_std = data['formatted']['figure_data']['zero_shot_std']
fine_tuned_std = data['formatted']['figure_data']['fine_tuned_std']

# Create figure with GridSpec for complex layout
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.35)

# Define colors
color_zero = '#2E86AB'  # Deep blue
color_fine = '#A23B72'  # Deep rose
color_improve = '#F18F01'  # Orange

# ============ Subplot (a): Main Performance Comparison ============
ax1 = fig.add_subplot(gs[0, :2])

# Plot with error bars and markers
x = np.arange(len(label_percentages))
width = 0.35

# Create bars with gradient effect
bars1 = ax1.bar(x - width/2, zero_shot, width, label='Zero-Shot',
                color=color_zero, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, fine_tuned, width, label='Fine-Tuned',
                color=color_fine, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars
ax1.errorbar(x - width/2, zero_shot, yerr=zero_shot_std, fmt='none',
             ecolor='darkblue', capsize=5, capthick=2, alpha=0.7)
ax1.errorbar(x + width/2, fine_tuned, yerr=fine_tuned_std, fmt='none',
             ecolor='darkred', capsize=5, capthick=2, alpha=0.7)

# Add value labels on bars
for bar, value in zip(bars1, zero_shot):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, value in zip(bars2, fine_tuned):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Label Percentage (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Performance vs. Label Availability', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{int(p)}%' for p in label_percentages])
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, 95)

# ============ Subplot (b): Improvement Ratio ============
ax2 = fig.add_subplot(gs[0, 2])

improvement = [(f - z) / z * 100 if z > 0 else 0 
               for f, z in zip(fine_tuned, zero_shot)]

# Create bar plot with gradient colors
bars3 = ax2.bar(range(len(label_percentages)), improvement, 
                color=[color_improve] * len(improvement),
                alpha=0.8, edgecolor='black', linewidth=1.5)

# Add percentage labels
for i, (bar, val) in enumerate(zip(bars3, improvement)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
             f'+{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Label %', fontsize=11, fontweight='bold')
ax2.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Relative Improvement', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(label_percentages)))
ax2.set_xticklabels([f'{int(p)}' for p in label_percentages], fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_ylim(0, max(improvement) * 1.2)

# ============ Subplot (c): Learning Curve ============
ax3 = fig.add_subplot(gs[1, 0])

# Plot smooth curves with shaded regions using numpy interpolation
x_smooth = np.linspace(0, len(label_percentages)-1, 300)

# For zero-shot (relatively flat)
y_zero_smooth = np.interp(x_smooth, range(len(label_percentages)), zero_shot)

# For fine-tuned (steep learning curve)
y_fine_smooth = np.interp(x_smooth, range(len(label_percentages)), fine_tuned)

ax3.plot(x_smooth, y_zero_smooth, color=color_zero, linewidth=3, 
         label='Zero-Shot', alpha=0.9)
ax3.plot(x_smooth, y_fine_smooth, color=color_fine, linewidth=3,
         label='Fine-Tuned', alpha=0.9)

# Add shaded confidence regions
ax3.fill_between(x_smooth, y_zero_smooth - 2, y_zero_smooth + 2,
                 color=color_zero, alpha=0.2)
ax3.fill_between(x_smooth, y_fine_smooth - 5, y_fine_smooth + 5,
                 color=color_fine, alpha=0.2)

# Mark actual data points
ax3.scatter(range(len(label_percentages)), zero_shot, 
           color=color_zero, s=100, zorder=5, edgecolor='white', linewidth=2)
ax3.scatter(range(len(label_percentages)), fine_tuned,
           color=color_fine, s=100, zorder=5, edgecolor='white', linewidth=2)

ax3.set_xlabel('Label Percentage Index', fontsize=11, fontweight='bold')
ax3.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
ax3.set_title('(c) Learning Curves', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(label_percentages)))
ax3.set_xticklabels([f'{int(p)}%' for p in label_percentages], fontsize=9)
ax3.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0, 90)

# ============ Subplot (d): Performance Gap Analysis ============
ax4 = fig.add_subplot(gs[1, 1])

gap = [f - z for f, z in zip(fine_tuned, zero_shot)]

# Create area plot
ax4.fill_between(range(len(label_percentages)), 0, gap,
                 color=color_improve, alpha=0.6, label='Performance Gap')
ax4.plot(range(len(label_percentages)), gap, 
         color=color_improve, linewidth=3, marker='o', markersize=8,
         markerfacecolor='white', markeredgewidth=2)

# Add annotations for key points
max_gap_idx = gap.index(max(gap))
ax4.annotate(f'Max Gap\n{gap[max_gap_idx]:.1f}%',
            xy=(max_gap_idx, gap[max_gap_idx]),
            xytext=(max_gap_idx-0.5, gap[max_gap_idx]+10),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold', ha='center')

ax4.set_xlabel('Label Percentage (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('F1 Score Gap (%)', fontsize=11, fontweight='bold')
ax4.set_title('(d) Performance Gap Analysis', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(label_percentages)))
ax4.set_xticklabels([f'{int(p)}%' for p in label_percentages], fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_ylim(0, max(gap) * 1.3)

# ============ Subplot (e): Efficiency Metric ============
ax5 = fig.add_subplot(gs[1, 2])

# Calculate efficiency: performance gain per label percentage
efficiency = [f / p if p > 0 else 0 
             for f, p in zip(fine_tuned, label_percentages)]

# Create scatter plot with size encoding
scatter = ax5.scatter(label_percentages, efficiency,
                     s=[e*5 for e in efficiency],  # Size proportional to efficiency
                     c=label_percentages, cmap='viridis',
                     alpha=0.7, edgecolors='black', linewidth=2)

# Add trend line
z = np.polyfit(label_percentages, efficiency, 2)
p = np.poly1d(z)
x_trend = np.linspace(min(label_percentages), max(label_percentages), 100)
ax5.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Trend')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Label %', fontsize=10)

ax5.set_xlabel('Label Percentage (%)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Efficiency (F1/Label%)', fontsize=11, fontweight='bold')
ax5.set_title('(e) Label Efficiency', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.legend(loc='upper right', fontsize=10)

# Add overall title
fig.suptitle('Figure 5: Comprehensive Label Efficiency Analysis of PASE-Net',
             fontsize=14, fontweight='bold', y=1.02)

# Add method annotation
fig.text(0.5, -0.02, 
         'Data source: Sim2Real transfer experiments on WiFi CSI HAR dataset',
         ha='center', fontsize=10, style='italic', color='gray')

# Save figure
output_path = Path('/workspace/paper/paper2_pase_net/manuscript/plots/fig5_label_efficiency_enhanced.pdf')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Enhanced Figure 5 saved to {output_path}")

# Also save as PNG for quick preview
png_path = output_path.with_suffix('.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"Preview saved to {png_path}")

plt.show()
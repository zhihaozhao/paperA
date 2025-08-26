#!/usr/bin/env python3
"""
Figure 2: System Flowchart
Left side: Complete process modules
Right side: Enhanced model diagram (in dashed box)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
fig.suptitle('WiFi CSI Fall Detection System Architecture', fontsize=16, fontweight='bold')

# Color scheme
colors = {
    'input': '#E8F4FD',
    'processing': '#B3D9FF', 
    'enhanced': '#FF9999',
    'output': '#90EE90',
    'arrow': '#4169E1',
    'text': '#2F4F4F'
}

# Left side: Complete process flow
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.set_title('(a) Complete Processing Pipeline', fontsize=14, fontweight='bold')

# Input layer
input_box = FancyBboxPatch((1, 10), 8, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], edgecolor='black', linewidth=1.5)
ax1.add_patch(input_box)
ax1.text(5, 10.75, 'WiFi CSI Data\n(Amplitude & Phase)', ha='center', va='center', 
         fontsize=11, fontweight='bold')

# Preprocessing
preproc_box = FancyBboxPatch((1, 8), 8, 1.5, boxstyle="round,pad=0.1",
                            facecolor=colors['processing'], edgecolor='black', linewidth=1.5)
ax1.add_patch(preproc_box)
ax1.text(5, 8.75, 'Preprocessing\n(Filtering, Normalization)', ha='center', va='center', fontsize=11)

# Feature extraction
feature_box = FancyBboxPatch((1, 6), 8, 1.5, boxstyle="round,pad=0.1",
                            facecolor=colors['processing'], edgecolor='black', linewidth=1.5)
ax1.add_patch(feature_box)
ax1.text(5, 6.75, 'Feature Extraction\n(Temporal & Spectral)', ha='center', va='center', fontsize=11)

# Enhanced model (highlighted)
enhanced_box = FancyBboxPatch((1, 4), 8, 1.5, boxstyle="round,pad=0.1",
                             facecolor=colors['enhanced'], edgecolor='red', linewidth=2)
ax1.add_patch(enhanced_box)
ax1.text(5, 4.75, 'Enhanced Model\n(Physics-Guided + Confidence Prior)', ha='center', va='center', 
         fontsize=11, fontweight='bold')

# Classification output
output_box = FancyBboxPatch((1, 2), 8, 1.5, boxstyle="round,pad=0.1",
                           facecolor=colors['output'], edgecolor='black', linewidth=1.5)
ax1.add_patch(output_box)
ax1.text(5, 2.75, 'Fall Detection Output\n(Fall/No-Fall + Confidence)', ha='center', va='center', 
         fontsize=11, fontweight='bold')

# Arrows for left side
arrow_props = dict(arrowstyle='->', lw=2, color=colors['arrow'])
ax1.annotate('', xy=(5, 8), xytext=(5, 9.5), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 6), xytext=(5, 7.5), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 4), xytext=(5, 5.5), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 2), xytext=(5, 3.5), arrowprops=arrow_props)

# Arrow pointing to right side
ax1.annotate('', xy=(9.5, 4.75), xytext=(9, 4.75), 
             arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax1.text(9.75, 5.2, 'Detail', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')

ax1.set_aspect('equal')
ax1.axis('off')

# Right side: Enhanced model details
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.set_title('(b) Enhanced Model Architecture', fontsize=14, fontweight='bold')

# Dashed box around the enhanced model
dashed_box = patches.Rectangle((0.5, 1), 9, 10, linewidth=2, edgecolor='red', 
                              facecolor='none', linestyle='--')
ax2.add_patch(dashed_box)

# Input layer
input_layer = FancyBboxPatch((1, 9.5), 8, 1, boxstyle="round,pad=0.1",
                            facecolor=colors['input'], edgecolor='black', linewidth=1)
ax2.add_patch(input_layer)
ax2.text(5, 10, 'Input Features\n(T×C×F)', ha='center', va='center', fontsize=10)

# Encoder layers
encoder_box = FancyBboxPatch((1, 7.5), 8, 1.5, boxstyle="round,pad=0.1",
                            facecolor=colors['processing'], edgecolor='black', linewidth=1)
ax2.add_patch(encoder_box)
ax2.text(5, 8.25, 'Temporal Encoder\n(LSTM/TCN/Transformer)\nHidden dim: 128', 
         ha='center', va='center', fontsize=10)

# Physics-guided component
physics_box = FancyBboxPatch((1, 5.5), 8, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#FFE4B5', edgecolor='orange', linewidth=1.5)
ax2.add_patch(physics_box)
ax2.text(5, 6.25, 'Physics-Guided Features\n(Doppler, Path Loss, Multipath)', 
         ha='center', va='center', fontsize=10, fontweight='bold')

# Confidence prior
confidence_box = FancyBboxPatch((1, 3.5), 8, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#E6E6FA', edgecolor='purple', linewidth=1.5)
ax2.add_patch(confidence_box)
ax2.text(5, 4.25, 'Confidence Prior\n(Logit Norm Regularization)\nλ = 0.01', 
         ha='center', va='center', fontsize=10, fontweight='bold')

# Output layer
output_layer = FancyBboxPatch((1, 1.5), 8, 1.5, boxstyle="round,pad=0.1",
                             facecolor=colors['output'], edgecolor='black', linewidth=1)
ax2.add_patch(output_layer)
ax2.text(5, 2.25, 'Output Layer\n(Softmax + Calibration)\n2 classes', 
         ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows for right side
ax2.annotate('', xy=(5, 7.5), xytext=(5, 9.5), arrowprops=arrow_props)
ax2.annotate('', xy=(5, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)
ax2.annotate('', xy=(5, 3.5), xytext=(5, 5.5), arrowprops=arrow_props)
ax2.annotate('', xy=(5, 1.5), xytext=(5, 3.5), arrowprops=arrow_props)

ax2.set_aspect('equal')
ax2.axis('off')

# Add enhanced model label
ax2.text(5, 0.5, 'Enhanced Model Detail', ha='center', va='center', 
         fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('/workspace/paper/figures/fig2_flowchart.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/workspace/paper/figures/fig2_flowchart.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 2 (flowchart) saved to paper/figures/")
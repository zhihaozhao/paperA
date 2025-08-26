#!/usr/bin/env python3
"""
Figure 3: Enhanced Model Details
2D visualization of model layers with parameter counts and dimensions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
fig.suptitle('Enhanced Model Architecture - Layer Details', fontsize=16, fontweight='bold')

ax.set_xlim(0, 12)
ax.set_ylim(0, 14)

# Color scheme for different layer types
colors = {
    'input': '#E8F4FD',
    'conv': '#FFE4B5',
    'rnn': '#B3D9FF',
    'attention': '#DDA0DD',
    'physics': '#98FB98',
    'confidence': '#F0E68C',
    'dense': '#FFA07A',
    'output': '#90EE90'
}

# Layer specifications
layers = [
    {'name': 'Input Layer', 'type': 'input', 'shape': '(B, T, C, F)', 'params': '0', 'pos': (1, 12), 'size': (10, 1)},
    {'name': 'Temporal Conv1D', 'type': 'conv', 'shape': '(B, T, 64)', 'params': '1.3K', 'pos': (1, 10.5), 'size': (10, 1)},
    {'name': 'LSTM Encoder', 'type': 'rnn', 'shape': '(B, T, 128)', 'params': '98.8K', 'pos': (1, 9), 'size': (10, 1)},
    {'name': 'Self-Attention', 'type': 'attention', 'shape': '(B, T, 128)', 'params': '49.2K', 'pos': (1, 7.5), 'size': (10, 1)},
    {'name': 'Physics Features', 'type': 'physics', 'shape': '(B, T, 32)', 'params': '4.1K', 'pos': (1, 6), 'size': (10, 1)},
    {'name': 'Feature Fusion', 'type': 'dense', 'shape': '(B, T, 160)', 'params': '25.8K', 'pos': (1, 4.5), 'size': (10, 1)},
    {'name': 'Confidence Prior', 'type': 'confidence', 'shape': '(B, T, 160)', 'params': '0', 'pos': (1, 3), 'size': (10, 1)},
    {'name': 'Dense + Dropout', 'type': 'dense', 'shape': '(B, 64)', 'params': '10.3K', 'pos': (1, 1.5), 'size': (10, 1)},
    {'name': 'Output (Softmax)', 'type': 'output', 'shape': '(B, 2)', 'params': '130', 'pos': (1, 0), 'size': (10, 1)}
]

# Draw layers
for i, layer in enumerate(layers):
    # Main layer box
    box = FancyBboxPatch(layer['pos'], layer['size'][0], layer['size'][1], 
                        boxstyle="round,pad=0.05", 
                        facecolor=colors[layer['type']], 
                        edgecolor='black', 
                        linewidth=1.5)
    ax.add_patch(box)
    
    # Layer name and details
    x_center = layer['pos'][0] + layer['size'][0]/2
    y_center = layer['pos'][1] + layer['size'][1]/2
    
    ax.text(x_center, y_center + 0.2, layer['name'], 
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(x_center, y_center - 0.2, f"Shape: {layer['shape']} | Params: {layer['params']}", 
           ha='center', va='center', fontsize=9)

# Add arrows between layers
arrow_props = dict(arrowstyle='->', lw=2, color='#4169E1')
for i in range(len(layers)-1):
    start_y = layers[i]['pos'][1]
    end_y = layers[i+1]['pos'][1] + layers[i+1]['size'][1]
    ax.annotate('', xy=(6, end_y), xytext=(6, start_y), arrowprops=arrow_props)

# Add parameter count annotations on the right
total_params = 0
param_counts = [0, 1300, 98800, 49200, 4100, 25800, 0, 10300, 130]
for i, (layer, count) in enumerate(zip(layers, param_counts)):
    total_params += count
    if count > 0:
        # Add parameter visualization (circles scaled by param count)
        circle_size = min(0.3, max(0.05, count / 100000 * 0.3))
        circle = Circle((11.5, layer['pos'][1] + 0.5), circle_size, 
                       facecolor='lightcoral', edgecolor='red', alpha=0.7)
        ax.add_patch(circle)

# Add legend for layer types
legend_x = 0.2
legend_y = 5.5
legend_items = [
    ('Input/Output', colors['input']),
    ('Convolutional', colors['conv']),
    ('Recurrent', colors['rnn']),
    ('Attention', colors['attention']),
    ('Physics-Guided', colors['physics']),
    ('Dense', colors['dense']),
    ('Confidence Prior', colors['confidence'])
]

for i, (name, color) in enumerate(legend_items):
    rect = Rectangle((legend_x, legend_y - i*0.4), 0.3, 0.3, 
                    facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(legend_x + 0.4, legend_y - i*0.4 + 0.15, name, 
           va='center', fontsize=9)

# Add total parameter count
ax.text(6, -1, f'Total Parameters: {total_params:,}', 
       ha='center', va='center', fontsize=12, fontweight='bold',
       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

# Add physics-guided features detail box
physics_detail_box = FancyBboxPatch((7.5, 6), 4, 2.5, boxstyle="round,pad=0.1",
                                   facecolor='#F0FFF0', edgecolor='green', 
                                   linewidth=1.5, linestyle='--')
ax.add_patch(physics_detail_box)
ax.text(9.5, 7.6, 'Physics Features Detail', ha='center', va='center', 
       fontsize=10, fontweight='bold', color='green')
ax.text(9.5, 7.2, '• Doppler shift estimation', ha='center', va='center', fontsize=8)
ax.text(9.5, 6.9, '• Path loss calculation', ha='center', va='center', fontsize=8)
ax.text(9.5, 6.6, '• Multipath delay spread', ha='center', va='center', fontsize=8)
ax.text(9.5, 6.3, '• Channel impulse response', ha='center', va='center', fontsize=8)

# Add confidence prior detail box
conf_detail_box = FancyBboxPatch((7.5, 2.5), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#FFFACD', edgecolor='orange', 
                                linewidth=1.5, linestyle='--')
ax.add_patch(conf_detail_box)
ax.text(9.5, 3.6, 'Confidence Prior Detail', ha='center', va='center', 
       fontsize=10, fontweight='bold', color='orange')
ax.text(9.5, 3.2, 'L_reg = λ × ||z||₂²', ha='center', va='center', fontsize=9)
ax.text(9.5, 2.9, 'λ = 0.01 (tuned)', ha='center', va='center', fontsize=8)

# Add dimension annotations
ax.text(11.8, 12.5, 'B: Batch size', ha='left', va='center', fontsize=8, style='italic')
ax.text(11.8, 12.2, 'T: Time steps', ha='left', va='center', fontsize=8, style='italic')
ax.text(11.8, 11.9, 'C: Channels', ha='left', va='center', fontsize=8, style='italic')
ax.text(11.8, 11.6, 'F: Features', ha='left', va='center', fontsize=8, style='italic')

# Parameter size legend (for circles)
ax.text(11.5, 9.5, 'Parameter Count', ha='center', va='center', 
       fontsize=9, fontweight='bold')
sizes = [0.05, 0.15, 0.3]
labels = ['<10K', '10K-50K', '>50K']
for i, (size, label) in enumerate(zip(sizes, labels)):
    circle = Circle((11.5, 8.8 - i*0.4), size, 
                   facecolor='lightcoral', edgecolor='red', alpha=0.7)
    ax.add_patch(circle)
    ax.text(11.9, 8.8 - i*0.4, label, ha='left', va='center', fontsize=8)

ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('/workspace/paper/figures/fig3_model_details.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/workspace/paper/figures/fig3_model_details.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 3 (model details) saved to paper/figures/")
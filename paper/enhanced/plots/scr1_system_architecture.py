#!/usr/bin/env python3
"""
Figure 1: PASE-Net System Architecture and Model Overview
Creates a comprehensive architecture diagram with system flow and detailed model structure
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

# Standard font configuration
plt.rcParams.update({
    'font.size': 10,           # Base text size
    'axes.titlesize': 14,      # Title size  
    'axes.labelsize': 10,      # Axis label size
    'xtick.labelsize': 10,     
    'ytick.labelsize': 10,     
    'legend.fontsize': 10,     
    'figure.titlesize': 14,    
    'axes.titleweight': 'bold'
})

def create_system_overview(ax):
    """Create system overview with data flow"""
    # Define system components
    components = [
        {'name': 'WiFi CSI\nSignals', 'xy': (0.15, 0.8), 'size': (0.12, 0.15), 'color': '#E3F2FD'},
        {'name': 'Physics\nModeling', 'xy': (0.15, 0.6), 'size': (0.12, 0.12), 'color': '#E8F5E8'},
        {'name': 'Synthetic\nData Gen', 'xy': (0.15, 0.4), 'size': (0.12, 0.12), 'color': '#FFF3E0'},
        {'name': 'PASE-Net\nModel', 'xy': (0.15, 0.2), 'size': (0.12, 0.12), 'color': '#F3E5F5'},
        {'name': 'Calibrated\nOutput', 'xy': (0.15, 0.05), 'size': (0.12, 0.1), 'color': '#FFEBEE'}
    ]
    
    # Draw components
    for comp in components:
        rect = FancyBboxPatch(
            (comp['xy'][0] - comp['size'][0]/2, comp['xy'][1] - comp['size'][1]/2),
            comp['size'][0], comp['size'][1],
            boxstyle="round,pad=0.01",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(comp['xy'][0], comp['xy'][1], comp['name'], 
                ha='center', va='center', fontsize=9, weight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='darkblue')
    arrow_positions = [(0.8, 0.6), (0.6, 0.4), (0.4, 0.2), (0.2, 0.05)]
    for i in range(len(arrow_positions)-1):
        ax.annotate('', xy=(0.15, arrow_positions[i+1][1] + 0.05), 
                    xytext=(0.15, arrow_positions[i][1] - 0.05),
                    arrowprops=arrow_props)
    
    # Add evaluation protocols on the right
    eval_protocols = [
        {'name': 'SRV\n668 trials', 'xy': (0.35, 0.7), 'color': '#FFE0B2'},
        {'name': 'CDAE\nLOSO/LORO', 'xy': (0.35, 0.5), 'color': '#C8E6C9'},
        {'name': 'STEA\nSim2Real', 'xy': (0.35, 0.3), 'color': '#FFCDD2'}
    ]
    
    for protocol in eval_protocols:
        rect = FancyBboxPatch(
            (protocol['xy'][0] - 0.05, protocol['xy'][1] - 0.08),
            0.1, 0.16,
            boxstyle="round,pad=0.01",
            facecolor=protocol['color'],
            edgecolor='gray',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(protocol['xy'][0], protocol['xy'][1], protocol['name'], 
                ha='center', va='center', fontsize=8, weight='bold')
    
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) System Overview', fontsize=14, weight='bold', pad=15)

def create_model_architecture(ax):
    """Create detailed PASE-Net architecture"""
    # Define model layers with detailed specifications
    layers = [
        {'name': 'CSI Input\n[B, 128, 52]', 'xy': (0.5, 0.95), 'size': (0.25, 0.08), 'color': '#E3F2FD'},
        
        # Convolutional blocks
        {'name': 'Conv2D Block1\n32 filters, 3×3\nBN + ReLU + Pool', 'xy': (0.5, 0.8), 'size': (0.3, 0.1), 'color': '#F3E5F5'},
        {'name': 'Conv2D Block2\n64 filters, 3×3\nBN + ReLU + Pool', 'xy': (0.5, 0.65), 'size': (0.3, 0.1), 'color': '#F3E5F5'},
        {'name': 'Conv2D Block3\n128 filters, 3×3\nBN + ReLU + Pool', 'xy': (0.5, 0.5), 'size': (0.3, 0.1), 'color': '#F3E5F5'},
        
        # Attention mechanisms
        {'name': 'SE Attention\nReduc=16, FC Layers\nSigmoid Gating', 'xy': (0.2, 0.35), 'size': (0.25, 0.12), 'color': '#FFE0B2'},
        {'name': 'Temporal Attention\nMulti-Head=8\nSelf-Attention', 'xy': (0.8, 0.35), 'size': (0.25, 0.12), 'color': '#C8E6C9'},
        
        # Fusion and output
        {'name': 'Feature Fusion\nConcatenation +\nLinear Projection', 'xy': (0.5, 0.2), 'size': (0.3, 0.1), 'color': '#FFCDD2'},
        {'name': 'Calibrated Output\nTemperature Scaling\n8 Activity Classes', 'xy': (0.5, 0.05), 'size': (0.3, 0.1), 'color': '#DCEDC8'}
    ]
    
    # Draw layers
    for layer in layers:
        rect = FancyBboxPatch(
            (layer['xy'][0] - layer['size'][0]/2, layer['xy'][1] - layer['size'][1]/2),
            layer['size'][0], layer['size'][1],
            boxstyle="round,pad=0.01",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(layer['xy'][0], layer['xy'][1], layer['name'], 
                ha='center', va='center', fontsize=8, weight='bold')
    
    # Draw main flow arrows
    main_flow_arrows = [
        [(0.5, 0.91), (0.5, 0.85)],  # Input to Conv1
        [(0.5, 0.75), (0.5, 0.7)],   # Conv1 to Conv2
        [(0.5, 0.6), (0.5, 0.55)],   # Conv2 to Conv3
    ]
    
    # Branching arrows after Conv3
    branch_arrows = [
        [(0.5, 0.45), (0.2, 0.41)],   # Conv3 to SE
        [(0.5, 0.45), (0.8, 0.41)],   # Conv3 to Temporal
        [(0.2, 0.29), (0.5, 0.25)],   # SE to Fusion
        [(0.8, 0.29), (0.5, 0.25)],   # Temporal to Fusion
        [(0.5, 0.15), (0.5, 0.1)]     # Fusion to Output
    ]
    
    arrow_props = dict(arrowstyle='->', lw=1.5, color='darkblue')
    
    # Draw main flow arrows
    for start, end in main_flow_arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    
    # Draw branching arrows
    for start, end in branch_arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    
    # Add parameter counts
    param_text = "Parameters: 2.3M\nFLOPs: 3.2G\nCapacity-matched"
    ax.text(0.05, 0.1, param_text, fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.02", facecolor='lightgray', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) PASE-Net Architecture', fontsize=14, weight='bold', pad=15)

def create_combined_figure():
    """Create the complete Figure 1"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    create_system_overview(ax1)
    create_model_architecture(ax2)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = create_combined_figure()
    output_path = "fig1_system_architecture.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1: {output_path}")
    plt.close()
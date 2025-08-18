#!/usr/bin/env python3
"""
Enhanced Model 3D Architecture Visualization
Inspired by SenseFi and SE-Networks architecture diagrams
IEEE IoTJ Paper - WiFi CSI HAR
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure for IEEE IoTJ standards
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_3d_block(ax, center, size, color, label, text_color='black'):
    """Create a 3D block with label"""
    x, y, z = center
    dx, dy, dz = size
    
    # Define vertices of the cube
    vertices = np.array([
        [x-dx/2, y-dy/2, z-dz/2], [x+dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y+dy/2, z-dz/2], [x-dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y-dy/2, z+dz/2], [x+dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y+dy/2, z+dz/2], [x-dx/2, y+dy/2, z+dz/2]
    ])
    
    # Define faces
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # left
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[2], vertices[3]]   # bottom
    ]
    
    # Create 3D collection
    collection = Poly3DCollection(faces, facecolors=color, edgecolors='black', 
                                 linewidths=1, alpha=0.8)
    ax.add_collection3d(collection)
    
    # Add label
    ax.text(x, y, z, label, ha='center', va='center', 
           fontsize=9, fontweight='bold', color=text_color)
    
    return collection

def create_enhanced_model_architecture():
    """
    Create comprehensive 3D architecture diagram for Enhanced model
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different components
    colors = {
        'input': '#E8F4FD',      # Light blue
        'cnn': '#FFE5B4',        # Light peach
        'se': '#FFD700',         # Gold (key innovation)
        'lstm': '#98FB98',       # Light green
        'attention': '#DDA0DD',  # Plum (key innovation)
        'output': '#F0E68C'      # Khaki
    }
    
    # === Input Layer ===
    create_3d_block(ax, [0, 0, 2], [1.5, 1.5, 0.8], colors['input'], 
                   'WiFi CSI Input\n(T×F×N)')
    
    # Add input dimensions annotation
    ax.text(0, 0, 1.2, 'T: Time Steps\nF: Frequency\nN: Antennas', 
           ha='center', va='center', fontsize=8, 
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # === CNN Feature Extraction Layers ===
    cnn_layers = [
        {'pos': [3, 0, 2.5], 'size': [1.2, 1.2, 0.6], 'label': 'Conv1D\n32 filters'},
        {'pos': [4.5, 0, 2.5], 'size': [1.2, 1.2, 0.6], 'label': 'Conv1D\n64 filters'},
        {'pos': [6, 0, 2.5], 'size': [1.2, 1.2, 0.6], 'label': 'Conv1D\n128 filters'}
    ]
    
    for i, layer in enumerate(cnn_layers):
        create_3d_block(ax, layer['pos'], layer['size'], colors['cnn'], layer['label'])
        
        # Add arrows between CNN layers
        if i < len(cnn_layers) - 1:
            start = layer['pos']
            end = cnn_layers[i+1]['pos']
            ax.plot([start[0]+0.6, end[0]-0.6], [start[1], end[1]], [start[2], end[2]], 
                   'b-', linewidth=2, alpha=0.7)
    
    # === Squeeze-and-Excitation Module (Key Innovation) ===
    create_3d_block(ax, [8, 0, 3], [1.8, 1.8, 1], colors['se'], 
                   'SE Module\n(Channel Attention)')
    
    # SE module detailed components
    ax.text(8, 0, 1.8, 'Global Avg Pool\n↓\nFC-ReLU-FC\n↓\nSigmoid', 
           ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    # === BiLSTM Layer ===
    create_3d_block(ax, [10.5, 0, 2.5], [2, 1.5, 0.8], colors['lstm'], 
                   'BiLSTM\n(Temporal Modeling)')
    
    # BiLSTM detail
    ax.text(10.5, 0, 1.5, 'Forward LSTM\n+\nBackward LSTM', 
           ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    # === Temporal Attention Mechanism (Key Innovation) ===
    create_3d_block(ax, [13, 0, 3.2], [1.8, 1.8, 1.2], colors['attention'], 
                   'Temporal\nAttention')
    
    # Attention detail
    ax.text(13, 0, 1.8, 'Query-Key-Value\n↓\nSoftmax\n↓\nWeighted Sum', 
           ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    # === Output Classification ===
    create_3d_block(ax, [15.5, 0, 2], [1.5, 1.5, 0.6], colors['output'], 
                   'Classification\nOutput')
    
    # === Data Flow Arrows ===
    # Input to CNN
    ax.plot([0.75, 2.4], [0, 0], [2, 2.5], 'k-', linewidth=3, alpha=0.8)
    
    # CNN to SE
    ax.plot([6.6, 7.1], [0, 0], [2.5, 3], 'b-', linewidth=3, alpha=0.8)
    
    # SE to BiLSTM
    ax.plot([8.9, 9.5], [0, 0], [3, 2.5], 'g-', linewidth=3, alpha=0.8)
    
    # BiLSTM to Attention
    ax.plot([11.5, 12.1], [0, 0], [2.5, 3.2], 'purple', linewidth=3, alpha=0.8)
    
    # Attention to Output
    ax.plot([13.9, 14.75], [0, 0], [3.2, 2], 'r-', linewidth=3, alpha=0.8)
    
    # === Key Innovation Highlights ===
    # SE module highlight
    ax.plot([8, 8, 8, 8, 8], [0.9, -0.9, -0.9, 0.9, 0.9], 
           [3.5, 3.5, 2.5, 2.5, 3.5], 'gold', linewidth=4, alpha=0.9)
    
    # Attention highlight
    ax.plot([13, 13, 13, 13, 13], [0.9, -0.9, -0.9, 0.9, 0.9], 
           [3.8, 3.8, 2.6, 2.6, 3.8], 'gold', linewidth=4, alpha=0.9)
    
    # === Architecture Advantages Text ===
    ax.text(8, 2.5, 4.5, '⭐ Key Innovations', fontsize=12, fontweight='bold', 
           color='darkorange', ha='center')
    ax.text(8, 2.5, 4.2, '• SE: Channel-wise attention', fontsize=10, 
           color='darkgoldenrod', ha='center')
    ax.text(8, 2.5, 3.9, '• Temporal: Long-range dependencies', fontsize=10, 
           color='purple', ha='center')
    
    # === Performance Results Integration ===
    ax.text(7.5, -2.5, 1.5, 'Performance Achievements:', fontsize=11, fontweight='bold', 
           color='darkblue', ha='center')
    ax.text(7.5, -2.5, 1.2, '• 83.0±0.1% Cross-Domain F1', fontsize=10, 
           color='darkgreen', ha='center')
    ax.text(7.5, -2.5, 0.9, '• Perfect LOSO-LORO Consistency', fontsize=10, 
           color='darkgreen', ha='center')
    ax.text(7.5, -2.5, 0.6, '• CV < 0.2% Exceptional Stability', fontsize=10, 
           color='darkgreen', ha='center')
    
    # === Model Comparison Context ===
    # Add small comparison models on the side
    comparison_models = [
        {'pos': [2, 3, 1], 'label': 'CNN\nBaseline', 'color': '#ADD8E6'},
        {'pos': [4, 3, 1], 'label': 'BiLSTM\nBaseline', 'color': '#FFB6C1'},
        {'pos': [6, 3, 1], 'label': 'Conformer\nLite', 'color': '#DDA0DD'}
    ]
    
    for model in comparison_models:
        create_3d_block(ax, model['pos'], [1, 1, 0.4], model['color'], model['label'])
    
    # Comparison arrow
    ax.text(4, 3.8, 1.5, 'Baseline Models', fontsize=10, fontweight='bold', 
           ha='center', color='gray')
    
    # === 3D Visualization Enhancements ===
    # Add data dimension visualization
    ax.text(0, -1.5, 3.5, 'Input Dimensions', fontsize=11, fontweight='bold', 
           color='darkblue', ha='center')
    
    # Time dimension
    ax.plot([0, 0], [-1.2, -0.8], [3.8, 4.2], 'r-', linewidth=3)
    ax.text(0, -1.0, 4.4, 'Time', fontsize=9, color='red', ha='center')
    
    # Frequency dimension  
    ax.plot([-0.4, 0.4], [-1.2, -1.2], [3.8, 3.8], 'g-', linewidth=3)
    ax.text(0, -1.4, 3.8, 'Frequency', fontsize=9, color='green', ha='center')
    
    # Antenna dimension
    ax.plot([0, 0], [-1.2, -1.2], [3.4, 4.2], 'b-', linewidth=3)
    ax.text(0.2, -1.2, 3.8, 'Antennas', fontsize=9, color='blue', ha='center')
    
    # === Customization ===
    ax.set_title('Enhanced Model 3D Architecture\n(CNN + SE + Temporal Attention)', 
                fontsize=16, fontweight='bold', pad=30)
    
    ax.set_xlabel('Processing Flow →', fontweight='bold', labelpad=15)
    ax.set_ylabel('Model Components', fontweight='bold', labelpad=15)
    ax.set_zlabel('Abstraction Level ↑', fontweight='bold', labelpad=15)
    
    # Set optimal viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Set axis limits
    ax.set_xlim(-1, 17)
    ax.set_ylim(-3, 4)
    ax.set_zlim(0, 5)
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Add subtle grid
    ax.grid(True, alpha=0.2)
    
    return fig, ax

def create_data_flow_diagram():
    """
    Create 2D data flow diagram inspired by reference papers
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # === CSI Data Input ===
    # Multiple antenna representation
    antennas = ['MIMO\nAntennas', 'Multiple\nChannels', 'Time\nSeries']
    for i, antenna in enumerate(antennas):
        rect = patches.FancyBboxPatch((0.5, 6-i*1.5), 1.5, 1, 
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightblue', 
                                     edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.25, 6.5-i*1.5, antenna, ha='center', va='center', 
               fontweight='bold', fontsize=10)
    
    # CSI Data visualization
    csi_rect = patches.FancyBboxPatch((2.5, 4), 2, 2, 
                                     boxstyle="round,pad=0.1",
                                     facecolor='#E8F4FD', 
                                     edgecolor='navy', linewidth=2)
    ax.add_patch(csi_rect)
    ax.text(3.5, 5, 'CSI Data\n(T×F×N)', ha='center', va='center', 
           fontweight='bold', fontsize=12)
    
    # === Enhanced Model Pipeline ===
    pipeline_components = [
        {'pos': (5.5, 5.5), 'size': (1.5, 1), 'label': 'CNN\nFeatures', 'color': '#FFE5B4'},
        {'pos': (7.5, 5.5), 'size': (1.5, 1), 'label': 'SE Module\n(Channel)', 'color': '#FFD700'},
        {'pos': (9.5, 5.5), 'size': (1.5, 1), 'label': 'BiLSTM\n(Temporal)', 'color': '#98FB98'},
        {'pos': (11.5, 5.5), 'size': (1.5, 1), 'label': 'Attention\n(Long-range)', 'color': '#DDA0DD'},
        {'pos': (13.5, 5.5), 'size': (1.5, 1), 'label': 'Classifier\nOutput', 'color': '#F0E68C'}
    ]
    
    for i, comp in enumerate(pipeline_components):
        rect = patches.FancyBboxPatch(comp['pos'], comp['size'][0], comp['size'][1],
                                     boxstyle="round,pad=0.05",
                                     facecolor=comp['color'], 
                                     edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2, 
               comp['label'], ha='center', va='center', 
               fontweight='bold', fontsize=10)
        
        # Add arrows between components
        if i < len(pipeline_components) - 1:
            start_x = comp['pos'][0] + comp['size'][0]
            end_x = pipeline_components[i+1]['pos'][0]
            y_pos = comp['pos'][1] + comp['size'][1]/2
            
            ax.annotate('', xy=(end_x, y_pos), xytext=(start_x, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Arrow from CSI Data to CNN
    ax.annotate('', xy=(5.5, 6), xytext=(4.5, 5),
               arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    # === Baseline Models Comparison ===
    ax.text(8, 3.5, 'Baseline Architectures', ha='center', fontsize=12, 
           fontweight='bold', color='gray')
    
    baselines = [
        {'pos': (2, 2.5), 'label': 'CNN Only\n84.2±2.5%', 'color': '#FFB6C1'},
        {'pos': (5, 2.5), 'label': 'BiLSTM Only\n80.3±2.2%', 'color': '#98FB98'},
        {'pos': (8, 2.5), 'label': 'Conformer-lite\n40.3±38.6%', 'color': '#DDA0DD'},
        {'pos': (11, 2.5), 'label': 'Enhanced (Ours)\n83.0±0.1%', 'color': '#FFD700'}
    ]
    
    for baseline in baselines:
        rect = patches.FancyBboxPatch(baseline['pos'], 2.5, 1,
                                     boxstyle="round,pad=0.05",
                                     facecolor=baseline['color'], 
                                     edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(baseline['pos'][0] + 1.25, baseline['pos'][1] + 0.5, 
               baseline['label'], ha='center', va='center', 
               fontweight='bold', fontsize=9)
    
    # Highlight Enhanced model
    enhanced_highlight = patches.FancyBboxPatch((10.8, 2.3), 2.9, 1.4,
                                               boxstyle="round,pad=0.05",
                                               facecolor='none', 
                                               edgecolor='gold', linewidth=4)
    ax.add_patch(enhanced_highlight)
    
    # === Performance Metrics Integration ===
    ax.text(8, 1, 'Key Achievements', ha='center', fontsize=12, 
           fontweight='bold', color='darkgreen')
    
    achievements = [
        '• Perfect Cross-Domain Consistency: 83.0±0.1% F1',
        '• Minimal Performance Gap: |LOSO - LORO| = 0.000',
        '• Exceptional Stability: CV < 0.2%', 
        '• Label Efficiency: 82.1% F1 @ 20% real data'
    ]
    
    for i, achievement in enumerate(achievements):
        ax.text(1, 0.5 - i*0.15, achievement, fontsize=10, 
               fontweight='bold', color='darkgreen')
    
    # === Title and Layout ===
    ax.set_title('Enhanced Model Architecture: Comprehensive Framework Overview\n(Inspired by SenseFi and SE-Networks)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 7.5)
    ax.axis('off')
    
    # Add subtle background grid
    for i in range(1, 16):
        ax.axvline(x=i, color='lightgray', alpha=0.2, linewidth=0.5)
    for i in range(0, 8):
        ax.axhline(y=i, color='lightgray', alpha=0.2, linewidth=0.5)
    
    return fig, ax

def export_architecture_data():
    """Export architecture data for documentation"""
    architecture_data = {
        'Component': ['Input', 'CNN_Conv1', 'CNN_Conv2', 'CNN_Conv3', 'SE_Module', 'BiLSTM', 'Attention', 'Output'],
        'Type': ['Data', 'Feature_Extraction', 'Feature_Extraction', 'Feature_Extraction', 
                'Channel_Attention', 'Temporal_Modeling', 'Temporal_Attention', 'Classification'],
        'Key_Innovation': [False, False, False, False, True, False, True, False],
        'Parameters': ['T×F×N', '32_filters', '64_filters', '128_filters', 'Squeeze_Excite', 
                      'Bidirectional', 'Query_Key_Value', 'Softmax'],
        'Color_Code': ['#E8F4FD', '#FFE5B4', '#FFE5B4', '#FFE5B4', '#FFD700', '#98FB98', '#DDA0DD', '#F0E68C']
    }
    
    import pandas as pd
    arch_df = pd.DataFrame(architecture_data)
    arch_df.to_csv('enhanced_model_architecture.csv', index=False)
    
    print("💾 Architecture data exported: enhanced_model_architecture.csv")

if __name__ == "__main__":
    print("🏗️ Generating Enhanced Model 3D Architecture...")
    print("📊 Inspired by SenseFi and SE-Networks visualization style")
    
    # Generate 3D architecture
    print("\n🎯 Creating 3D Architecture Diagram...")
    fig1, ax1 = create_enhanced_model_architecture()
    
    # Generate 2D data flow diagram
    print("📊 Creating 2D Data Flow Diagram...")
    fig2, ax2 = create_data_flow_diagram()
    
    # Save figures
    output_files = [
        ('enhanced_model_3d_architecture.pdf', fig1),
        ('enhanced_model_3d_architecture.png', fig1),
        ('enhanced_model_dataflow.pdf', fig2),
        ('enhanced_model_dataflow.png', fig2)
    ]
    
    for filename, fig in output_files:
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved: {filename}")
    
    # Export data
    export_architecture_data()
    
    # Display plots
    plt.show()
    
    print("\n🎉 Enhanced Model Architecture Visualization Complete!")
    print("🏗️ Features:")
    print("• 3D block-based architecture representation")
    print("• Key innovations highlighted (SE + Attention)")
    print("• Data flow arrows and component relationships")
    print("• Performance achievements integration")
    print("• Professional IEEE IoTJ design quality")
    print("• Inspired by leading papers in the field")
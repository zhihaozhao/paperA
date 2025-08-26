#!/usr/bin/env python3
"""
Figure 2: Experimental Evaluation Protocols
D2, CDAE, STEA, and PSTA/ESTA Protocol Visualization
IEEE IoTJ Paper - WiFi CSI HAR
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle, Circle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Set publication-ready style (fallback safe)
try:
    plt.style.use('seaborn-v0_8-paper')
except Exception:
    try:
        plt.style.use('seaborn-paper')
    except Exception:
        pass

# Configure for IEEE IoTJ standards
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def scale_fonts(fig: plt.Figure, factor: float) -> None:
    for text in fig.findobj(match=lambda x: isinstance(x, plt.Text)):
        try:
            text.set_fontsize(text.get_fontsize() * factor)
        except Exception:
            pass

def create_protocol_box(ax, xy, width, height, title, content, color, title_color='black'):
    """Create a protocol description box"""
    # Main box
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Title box
    title_box = FancyBboxPatch(
        xy, width, 0.4,
        boxstyle="round,pad=0.01",
        facecolor='#DDE7FF',
        edgecolor='black',
        linewidth=1,
        alpha=0.95
    )
    ax.add_patch(title_box)
    
    # Title text
    ax.text(xy[0] + width/2, xy[1] + 0.2, title, 
           ha='center', va='center', 
           fontsize=10, fontweight='bold', color=title_color, wrap=True)
    
    # Content text
    ax.text(xy[0] + width/2, xy[1] + height/2 - 0.1, content, 
           ha='center', va='center', 
           fontsize=9, color='black', wrap=True,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    return box

def create_flow_arrow(ax, start, end, label='', color='blue', curve=0):
    """Create a curved flow arrow with label"""
    if curve == 0:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc=color, ec=color, linewidth=2)
    else:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              connectionstyle=f"arc3,rad={curve}",
                              mutation_scale=20, fc=color, ec=color, linewidth=2)
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + curve * 0.5
        ax.text(mid_x, mid_y, label, ha='center', va='center',
               fontsize=7, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    return arrow

def create_comprehensive_protocols():
    """
    Create comprehensive experimental protocols diagram
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Title
    ax.text(8, 11.5, 'Comprehensive Experimental Evaluation Protocols', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Protocol colors
    protocol_colors = {
        'd2': '#FFE5B4',      # Peach
        'cdae': '#C8E6C9',    # Light green
        'stea': '#E1F5FE'     # Light blue
    }
    
    # === Synthetic Robustness Validation (Left Column) ===
    ax.text(3, 10.5, 'Synthetic Robustness\nValidation (SRD)', fontsize=15, fontweight='bold', color='darkorange', wrap=True)
    ax.text(3, 10.2, 'Noise, Overlap, Difficulty Sweeps', fontsize=11, color='darkorange', style='italic')
    
    d2_content = """Objective: Validate synthetic data quality
    • 540 configurations tested
    • Noise levels: {0.0, 0.4, 0.8}
    • Overlap conditions: {0.0, 0.05, 0.1}
    • Difficulty: {easy, medium, hard}
    • Models: 4 architectures × 5 seeds"""
    
    create_protocol_box(ax, (0.5, 8.5), 5, 1.8, 'Synthetic Robustness: 540 Configs', 
                       d2_content, protocol_colors['d2'])
    
    # D2 Flow diagram
    d2_boxes = [
        {'pos': (0.8, 7.5), 'size': (1.2, 0.6), 'label': 'Synthetic\nGeneration', 'color': '#FFCC99'},
        {'pos': (2.2, 7.5), 'size': (1.2, 0.6), 'label': 'Noise\nInjection', 'color': '#FFB366'},
        {'pos': (3.6, 7.5), 'size': (1.2, 0.6), 'label': 'Multi-Model\nTesting', 'color': '#FF9933'}
    ]
    
    for i, box_info in enumerate(d2_boxes):
        FancyBboxPatch(box_info['pos'], box_info['size'][0], box_info['size'][1],
                      boxstyle="round,pad=0.01", facecolor=box_info['color'],
                      edgecolor='black', linewidth=1)
        ax.add_patch(FancyBboxPatch(box_info['pos'], box_info['size'][0], box_info['size'][1],
                                   boxstyle="round,pad=0.01", facecolor=box_info['color'],
                                   edgecolor='black', linewidth=1))
        ax.text(box_info['pos'][0] + box_info['size'][0]/2, 
               box_info['pos'][1] + box_info['size'][1]/2,
               box_info['label'], ha='center', va='center', fontsize=8, fontweight='bold')
        
        if i < len(d2_boxes) - 1:
            create_flow_arrow(ax, 
                            (box_info['pos'][0] + box_info['size'][0], box_info['pos'][1] + box_info['size'][1]/2),
                            (d2_boxes[i+1]['pos'][0], d2_boxes[i+1]['pos'][1] + box_info['size'][1]/2),
                            color='darkorange')
    
    # === CDAE Protocol (Middle Column) ===
    ax.text(8, 10.5, 'CDAE Protocol', fontsize=14, fontweight='bold', color='darkgreen')
    ax.text(8, 10.2, 'Cross-Domain Adaptation Evaluation', fontsize=11, color='darkgreen', style='italic')
    
    cdae_content = """Objective: Cross-domain generalization
    • LOSO: Leave-One-Subject-Out
    • LORO: Leave-One-Room-Out
    • 40 configurations total
    • 4 models × 2 protocols × 5 seeds
    • Target: Domain-agnostic performance"""
    
    create_protocol_box(ax, (5.5, 8.5), 5, 1.8, 'CDAE: Cross-Domain Excellence', 
                       cdae_content, protocol_colors['cdae'])
    
    # CDAE Visual representation
    # LOSO representation
    ax.text(6.5, 7.8, 'LOSO Evaluation', fontsize=10, fontweight='bold', color='darkgreen')
    subjects = ['S1', 'S2', 'S3', 'S4', 'Test']
    for i, subj in enumerate(subjects):
        color = '#FF6B6B' if subj == 'Test' else '#4ECDC4'
        circle = Circle((6.2 + i*0.4, 7.4), 0.15, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(6.2 + i*0.4, 7.4, subj, ha='center', va='center', fontsize=7, fontweight='bold')
    
    # LORO representation
    ax.text(8.5, 7.8, 'LORO Evaluation', fontsize=10, fontweight='bold', color='darkgreen')
    rooms = ['R1', 'R2', 'R3', 'Test']
    for i, room in enumerate(rooms):
        color = '#FF6B6B' if room == 'Test' else '#95E1D3'
        rect = Rectangle((8.2 + i*0.4, 7.25), 0.3, 0.3, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(8.35 + i*0.4, 7.4, room, ha='center', va='center', fontsize=7, fontweight='bold')
    
    # === STEA Protocol (Right Column) ===
    ax.text(13, 10.5, 'STEA Protocol', fontsize=14, fontweight='bold', color='darkblue')
    ax.text(13, 10.2, 'Sim2Real Transfer Efficiency Assessment', fontsize=11, color='darkblue', style='italic')
    
    stea_content = """Objective: Label efficiency quantification
    • Transfer methods: 4 approaches
    • Label ratios: {1%, 5%, 10%, 15%, 20%, 50%, 100%}
    • 56 configurations completed
    • Target: Minimal real data requirement
    • Result: 82.1% F1 @ 20% labels"""
    
    create_protocol_box(ax, (10.5, 8.5), 5, 1.8, 'STEA: Transfer Efficiency', 
                       stea_content, protocol_colors['stea'])
    
    # STEA Transfer methods visualization
    transfer_methods = [
        {'name': 'Zero-shot', 'pos': (11, 7.5), 'color': '#FF9999'},
        {'name': 'Linear Probe', 'pos': (12, 7.5), 'color': '#FFCC99'},
        {'name': 'Fine-tune', 'pos': (13.5, 7.5), 'color': '#99FF99'},
        {'name': 'Temp Scaling', 'pos': (14.8, 7.5), 'color': '#99CCFF'}
    ]
    
    for method in transfer_methods:
        circle = Circle(method['pos'], 0.2, facecolor=method['color'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(method['pos'][0], method['pos'][1], method['name'][:4], 
               ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Label efficiency arrow
    ax.annotate('', xy=(14.5, 6.8), xytext=(11.5, 6.8),
               arrowprops=dict(arrowstyle='<->', lw=2, color='darkblue'))
    ax.text(13, 6.5, 'Label Efficiency: 1% -> 100%', ha='center', fontsize=9, 
           fontweight='bold', color='darkblue')
    
    # === Integration Flow ===
    ax.text(8, 6, 'Protocol Integration and Results', fontsize=14, fontweight='bold', color='darkred')
    
    # Integration boxes
    integration_boxes = [
        {'pos': (2, 5), 'size': (3, 0.8), 'label': 'Synthetic Robustness\n✓ Synthetic data quality', 'color': '#FFE5B4'},
        {'pos': (6.5, 5), 'size': (3, 0.8), 'label': 'CDAE Generalization\n✓ 83.0±0.1% Cross-domain', 'color': '#C8E6C9'},
        {'pos': (11, 5), 'size': (3, 0.8), 'label': 'STEA Efficiency\n✓ 82.1% @ 20% labels', 'color': '#E1F5FE'}
    ]
    
    for box_info in integration_boxes:
        FancyBboxPatch(box_info['pos'], box_info['size'][0], box_info['size'][1],
                      boxstyle="round,pad=0.02", facecolor=box_info['color'],
                      edgecolor='darkred', linewidth=2)
        ax.add_patch(FancyBboxPatch(box_info['pos'], box_info['size'][0], box_info['size'][1],
                                   boxstyle="round,pad=0.02", facecolor=box_info['color'],
                                   edgecolor='darkred', linewidth=2))
        ax.text(box_info['pos'][0] + box_info['size'][0]/2, 
               box_info['pos'][1] + box_info['size'][1]/2,
               box_info['label'], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Convergence arrows
    create_flow_arrow(ax, (3.5, 5), (7, 4), color='darkred', curve=0.2)
    create_flow_arrow(ax, (8, 5), (8, 4), color='darkred')
    create_flow_arrow(ax, (12.5, 5), (9, 4), color='darkred', curve=-0.2)

    # PSTA/ESTA stress and stability annotations
    psta_box = Rectangle((2.0, 3.5), 4.0, 0.8, facecolor='#FFF3CD', edgecolor='darkorange', linewidth=1.5)
    ax.add_patch(psta_box)
    ax.text(4.0, 3.9, 'PSTA: Progressive Stress-Test', ha='center', va='center', fontsize=10, fontweight='bold', color='darkorange')

    esta_box = Rectangle((10.0, 3.5), 4.0, 0.8, facecolor='#D6EAF8', edgecolor='darkblue', linewidth=1.5)
    ax.add_patch(esta_box)
    ax.text(12.0, 3.9, 'ESTA: Extended Stability', ha='center', va='center', fontsize=10, fontweight='bold', color='darkblue')

    # Final result box
    create_protocol_box(ax, (6, 2.5), 4, 1.2, 'Breakthrough Results', 
                       """✓ First systematic Sim2Real study in WiFi CSI HAR
    ✓ 83.0±0.1% F1 perfect cross-domain consistency
    ✓ 82.1% F1 using only 20% labeled real data
    ✓ 80% labeling cost reduction achieved
    ✓ Publication-ready trustworthy evaluation""", 
                       '#FFD700', title_color='black')
    
    # === Statistical Significance Section ===
    ax.text(2, 1.8, 'Statistical Validation', fontsize=12, fontweight='bold', color='purple')
    
    stats_content = [
        "• Significance testing: p-values computed",
        "• Confidence intervals: 95% CI reported", 
        "• Effect sizes: Cohen's d calculated",
        "• Multiple comparisons: Bonferroni correction",
        "• Cross-validation: 5-fold repeated"
    ]
    
    for i, stat in enumerate(stats_content):
        ax.text(0.5, 1.4 - i*0.15, stat, fontsize=8, color='purple')
    
    # === Performance Metrics Table ===
    ax.text(12, 1.8, 'Key Performance Summary', fontsize=13, fontweight='bold', color='darkblue')
    
    metrics = [
        ["Protocol", "Key Metric", "Achievement"],
        ["Synthetic Robustness", "Robustness", "540 configs validated"],
        ["CDAE", "Consistency", "CV < 0.2%"],
        ["STEA", "Efficiency", "80% cost reduction"]
    ]
    
    for i, row in enumerate(metrics):
        for j, cell in enumerate(row):
            weight = 'bold' if i == 0 else 'normal'
            color = 'darkblue' if i == 0 else 'black'
            ax.text(11 + j*1.5, 1.4 - i*0.15, cell, fontsize=8, 
                   fontweight=weight, color=color)
    
    # Set axis properties
    ax.set_xlim(0, 16)
    ax.set_ylim(2, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add subtle background grid
    for i in range(1, 16):
        ax.axvline(x=i, color='lightgray', alpha=0.2, linewidth=0.5)
    for i in range(2, 12):
        ax.axhline(y=i, color='lightgray', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    
    return fig, ax

def create_protocol_flowchart():
    """
    Create a simplified protocol flowchart for better understanding
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.text(7, 7.5, 'Experimental Protocol Flowchart', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Define flowchart stages
    stages = [
        {'pos': (1, 6), 'size': (2.5, 0.8), 'label': 'Physics-Guided\nSynthetic Generation', 'color': '#FFE5B4'},
        {'pos': (5, 6.5), 'size': (2, 0.6), 'label': 'Synthetic Robustness\nValidation', 'color': '#FFD700'},
        {'pos': (5, 5.5), 'size': (2, 0.6), 'label': 'Enhanced Model\nTraining', 'color': '#FFC0CB'},
        {'pos': (9, 6.5), 'size': (2, 0.6), 'label': 'CDAE Protocol\nCross-Domain', 'color': '#C8E6C9'},
        {'pos': (9, 5.5), 'size': (2, 0.6), 'label': 'STEA Protocol\nTransfer Efficiency', 'color': '#E1F5FE'},
        {'pos': (12.5, 6), 'size': (2, 0.8), 'label': 'Trustworthy\nEvaluation', 'color': '#E6E6FA'}
    ]
    
    # Create boxes
    for stage in stages:
        box = FancyBboxPatch(
            stage['pos'], stage['size'][0], stage['size'][1],
            boxstyle="round,pad=0.02",
            facecolor=stage['color'],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(box)
        
        ax.text(stage['pos'][0] + stage['size'][0]/2, 
               stage['pos'][1] + stage['size'][1]/2,
               stage['label'], ha='center', va='center',
               fontsize=10, fontweight='bold')
    
    # Add arrows between stages
    arrows = [
        ((3.5, 6.4), (5, 6.8)),      # Generation to SRV
        ((3.5, 6.4), (5, 5.8)),      # Generation to Training
        ((7, 6.8), (9, 6.8)),        # SRV to CDAE
        ((7, 5.8), (9, 5.8)),        # Training to STEA  
        ((11, 6.8), (12.5, 6.5)),    # CDAE to Evaluation
        ((11, 5.8), (12.5, 6.3))     # STEA to Evaluation
    ]
    
    for start, end in arrows:
        create_flow_arrow(ax, start, end, color='darkblue')
    
    # Add results summary
    ax.text(7, 4, 'Breakthrough Results Achieved', ha='center', fontsize=14, 
           fontweight='bold', color='darkred')
    
    results_box = Rectangle((3, 2.5), 8, 1.2, facecolor='#FFE4E1', 
                           edgecolor='darkred', linewidth=2)
    ax.add_patch(results_box)
    
    results_text = """• Perfect Cross-Domain Consistency: 83.0±0.1% F1 (LOSO = LORO)
    • Revolutionary Label Efficiency: 82.1% F1 using only 20% real data
    • Unprecedented Cost Reduction: 80% labeling cost savings
    • Statistical Significance: Rigorous p-value testing and confidence intervals"""
    
    ax.text(7, 3.1, results_text, ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax.set_xlim(0, 15)
    ax.set_ylim(2, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig, ax

def export_protocols_data():
    """
    Export experimental protocols data
    """
    # Protocol specifications
    import pandas as pd
    
    protocol_data = {
        'Protocol': ['Synthetic Robustness', 'CDAE', 'STEA'],
        'Full_Name': ['Robustness Analysis', 'Cross-Domain Adaptation Evaluation', 'Sim2Real Transfer Efficiency Assessment'],
        'Configurations': [540, 40, 56],
        'Key_Metric': ['Synthetic_Quality', 'Cross_Domain_F1', 'Label_Efficiency'],
        'Achievement': ['Validated', '83.0±0.1%', '82.1% @ 20%'],
        'Innovation': ['Multi_Parameter', 'Perfect_Consistency', 'Cost_Reduction']
    }
    
    protocols_df = pd.DataFrame(protocol_data)
    protocols_df.to_csv('figure2_protocols_summary.csv', index=False)
    
    # Detailed configuration data
    config_data = {
        'Protocol': ['Synthetic Robustness']*9 + ['CDAE']*8 + ['STEA']*7,
        'Parameter': ['Models', 'Seeds', 'Noise_Levels', 'Overlap_Conditions', 'Difficulty_Levels', 
                     'Time_Steps', 'Feature_Dims', 'Burst_Rates', 'Total_Configs'] +
                    ['Models', 'Protocols', 'Seeds', 'Subjects', 'Rooms', 'Cross_Validation', 
                     'Statistical_Tests', 'Total_Configs'] +
                    ['Transfer_Methods', 'Label_Ratios', 'Seeds', 'Baselines', 'Metrics', 
                     'Efficiency_Phases', 'Total_Configs'],
        'Value': [4, 5, 3, 3, 3, 3, 3, 3, 540] +
                [4, 2, 5, 'Multiple', 'Multiple', '5-fold', 'p-values', 40] +
                [4, 7, 5, 'Zero-shot', 'F1/ECE/Brier', 3, 56]
    }
    
    config_df = pd.DataFrame(config_data)
    config_df.to_csv('figure2_detailed_configurations.csv', index=False)
    
    print("\n💾 Figure 2 Data Export Complete:")
    print("• figure2_protocols_summary.csv - Protocol overview")
    print("• figure2_detailed_configurations.csv - Detailed configurations")

if __name__ == "__main__":
    print("📋 Generating Figure 2: Experimental Protocols...")
    print("🔬 D2, CDAE, STEA, and PSTA/ESTA Protocol Visualization")

    # Resolve output directory to paper/figures
    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / "paper" / "figures"
    FIGS.mkdir(parents=True, exist_ok=True)

    # Generate comprehensive protocols diagram for canonical (larger font) export
    fig_full, ax_full = create_comprehensive_protocols()
    scale_fonts(fig_full, 1.15)
    fig_full.savefig(FIGS / 'fig4_experimental_protocols.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("✅ Saved: fig4_experimental_protocols.pdf")

    # Generate simplified flowchart
    fig_flow, ax_flow = create_protocol_flowchart()
    scale_fonts(fig_flow, 1.05)
    fig_flow.savefig(FIGS / 'fig4_protocol_flowchart.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("✅ Saved: fig4_protocol_flowchart.pdf")

    # Overview (compact) export: regenerate and downscale fonts to avoid overlap
    fig_over, ax_over = create_comprehensive_protocols()
    fig_over.set_size_inches(7.2, 4.5)
    scale_fonts(fig_over, 0.80)
    fig_over.savefig(FIGS / 'fig4_experimental_overview.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.06)
    print("✅ Saved: figure4_experimental_overview.pdf (double-column)")

    # Export data
    export_protocols_data()

    # Display plots
    plt.show()

    print("\n🎉 Figure 2 Generation Complete!")
    print("📋 Comprehensive experimental protocol visualization ready")
    print("🔬 Features: D2 validation + CDAE cross-domain + STEA efficiency + statistical rigor")
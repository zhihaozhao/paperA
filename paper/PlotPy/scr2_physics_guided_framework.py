#!/usr/bin/env python3
"""
Figure 2: Physics-Guided Sim2Real Framework (2D flowchart)
Saves: paper/figures/fig2_physics_guided_framework.pdf
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import matplotlib.patheffects as pe

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def box(ax, x, y, w, h, label, fc):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03', facecolor=fc,
                          edgecolor='black', linewidth=1.5, alpha=0.98)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=11, color='black', wrap=True)


def arrow(ax, x1, y1, x2, y2, color='black'):
    cp = ConnectionPatch((x1, y1), (x2, y2), 'data', 'data', arrowstyle='->',
                         mutation_scale=15, fc=color, ec=color, linewidth=1.8)
    ax.add_patch(cp)


def create_fig():
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    ax.set_title('Physics-Guided Sim2Real Framework (2D)', fontweight='bold', color='black')

    # Left column: complete process modules (top to bottom)
    left_x = 0.6
    w = 3.0
    h = 0.9
    y_positions = [5.1, 4.0, 2.9, 1.8, 0.7]
    labels = [
        'Data Acquisition\n(Raw CSI Streams)',
        'Preprocessing\n(Sync, Denoise, Normalization)',
        'Physics Modeling\n(Multipath, Human, Environment)',
        'Parameterized Synthesis\n(Scenario + Channel Control)',
        'Enhanced Model\n(CNN + SE + Temporal Attention)'
    ]
    colors = ['#E6F2FF', '#EAF7E6', '#D4EDDA', '#FFF3CD', '#F8D7DA']
    for y, label, fc in zip(y_positions, labels, colors):
        box(ax, left_x, y, w, h, label, fc)

    # Connect left column top-down arrows
    for i in range(len(y_positions) - 1):
        y1 = y_positions[i]
        y2 = y_positions[i+1]
        arrow(ax, left_x + w/2, y1, left_x + w/2, y2 + h)

    # Right dashed model box (rough model diagram)
    dashed = Rectangle((7.2, 1.2), 4.0, 3.8, fill=False, linestyle='--', linewidth=2.0, edgecolor='black')
    ax.add_patch(dashed)
    ax.text(9.2, 4.9, 'Enhanced Model (Rough Diagram)', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Contents inside dashed model box (rough blocks)
    bx_x, bx_y, bx_w, bx_h = 7.5, 3.8, 3.4, 0.7
    box(ax, bx_x, bx_y, bx_w, bx_h, 'Conv Blocks\n(3×3, C=[32,64,128], Stride=[1,2,2])', '#FDEBD0')
    box(ax, bx_x, bx_y - 0.9, bx_w, bx_h, 'SE Block\n(Reduction Ratio r=16)', '#FADBD8')
    box(ax, bx_x, bx_y - 1.8, bx_w, bx_h, 'BiLSTM\n(Hidden=128 per direction)', '#D5F5E3')
    box(ax, bx_x, bx_y - 2.7, bx_w, bx_h, 'Temporal Attention\n(Heads=4)', '#E8DAEF')

    # Arrow from Enhanced Model (left column) to dashed model box
    enhanced_y = y_positions[-1]
    arrow(ax, left_x + w, enhanced_y + h/2, 7.2, 3.1)

    return fig

if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / 'paper' / 'figures'
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = create_fig()
    out = FIGS / 'fig2_physics_guided_framework.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('✅ Saved:', out)
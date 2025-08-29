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
    ax.set_ylim(-0.8, 7.8)
    ax.axis('off')

    ax.set_title('Physics-Guided Sim2Real Framework', fontweight='bold', color='black', pad=6)

    # Left column: complete process modules (bottom to top) with larger spacing
    left_x = 0.6
    w = 3.2
    h = 1.0
    spacing = 0.35
    base_y = 0.7 - (h + spacing)
    labels = [
        'Data Acquisition\n(Raw CSI\nStreams)',
        'Preprocessing\n(Sync, Denoise,\nNormalization)',
        'Physics Modeling\n(Multipath, Human,\nEnvironment)',
        'Parameterized Synthesis\n(Scenario + Channel\nControl)',
        'Enhanced Model\n(CNN + SE + Temporal\nAttention)'
    ]
    colors = ['#E6F2FF', '#EAF7E6', '#D4EDDA', '#FFF3CD', '#F8D7DA']
    y_positions = []
    y_current = base_y
    for label, fc in zip(labels, colors):
        box(ax, left_x, y_current, w, h, label, fc)
        y_positions.append(y_current)
        y_current += h + spacing

    # Connect bottom-to-top arrows
    for i in range(len(y_positions) - 1):
        y1 = y_positions[i] + h
        y2 = y_positions[i+1]
        arrow(ax, left_x + w/2, y1, left_x + w/2, y2)

    # Right dashed model box (rough model diagram); extend to accommodate taller blocks
    # dashed_x, dashed_y, dashed_w, dashed_h = 7.2, 0.6, 4.6, 5.2
    dashed_x, dashed_y, dashed_w, dashed_h = 7.2, 0.6, 4.6, 6.0
    dashed = Rectangle((dashed_x, dashed_y), dashed_w, dashed_h, fill=False, linestyle='--', linewidth=2.0, edgecolor='black')
    ax.add_patch(dashed)
    ax.text(dashed_x + dashed_w/2, dashed_y + dashed_h + 0.12, 'Enhanced Model ',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Contents inside dashed model box (rough blocks)
    bx_x, bx_w = dashed_x + 0.3, 3.9
    bx_h = 1.20
    bx_y = dashed_y + dashed_h - (bx_h + 0.15)
    box(ax, bx_x, bx_y, bx_w, bx_h, 'Conv Blocks\n(3×3)\nC=[32,64,128]\nStride=[1,2,2]', '#FDEBD0')
    box(ax, bx_x, bx_y - (bx_h + 0.25), bx_w, bx_h, 'SE Block\n(Reduction Ratio r=16)', '#FADBD8')
    box(ax, bx_x, bx_y - 2*(bx_h + 0.25), bx_w, bx_h, 'BiLSTM\n(Hidden=128 per\ndirection)', '#D5F5E3')
    box(ax, bx_x, bx_y - 3*(bx_h + 0.25), bx_w, bx_h, 'Temporal Attention\n(Heads=4)', '#E8DAEF')

    # Arrow from Enhanced Model (left column) to dashed model box
    enhanced_y = y_positions[-1]
    arrow(ax, left_x + w, enhanced_y + h/2, dashed_x, dashed_y + dashed_h/2)

    return fig

if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / 'paper' / 'figures'
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = create_fig()
    out = FIGS / 'fig2_physics_guided_framework.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('✅ Saved:', out)
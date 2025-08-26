#!/usr/bin/env python3
"""
Figure 3: Enhanced Model Dataflow (3D-style schematic)
Saves: paper/figures/fig3_enhanced_model_dataflow.pdf
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

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
                          edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=11, color='black', wrap=True)


def arrow(ax, x1, y1, x2, y2, color='black'):
    cp = ConnectionPatch((x1, y1), (x2, y2), 'data', 'data', arrowstyle='->',
                         mutation_scale=15, fc=color, ec=color, linewidth=1.8)
    ax.add_patch(cp)


def create_fig():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.set_title('Enhanced Model Dataflow (CNN + SE + Temporal Attention)', fontweight='bold')

    # Input and preprocessing
    box(ax, 0.6, 4.2, 2.2, 1.0, 'Input CSI\n(Preprocessed)', '#E8F4FD')
    arrow(ax, 2.8, 4.7, 3.2, 4.7)

    # CNN feature extraction
    box(ax, 3.2, 4.2, 2.2, 1.0, 'CNN Feature\nExtraction', '#FDEBD0')
    arrow(ax, 5.4, 4.7, 5.8, 4.7)

    # SE channel attention
    box(ax, 5.8, 4.2, 2.2, 1.0, 'Squeeze-Excitation\nChannel Attention', '#FADBD8')
    arrow(ax, 8.0, 4.7, 8.4, 4.7)

    # Temporal attention
    box(ax, 8.4, 4.2, 2.2, 1.0, 'Temporal\nAttention', '#E8DAEF')
    arrow(ax, 10.6, 4.7, 11.0, 4.7)

    # Output / Classifier
    box(ax, 11.0, 4.2, 1.2, 1.0, 'Classifier', '#D6EAF8')

    # Side panel: BiLSTM representation
    box(ax, 3.8, 2.2, 6.0, 0.9, 'Bidirectional LSTM (Context Integration)', '#D5F5E3')

    # Legend
    ax.text(1.5, 1.0, 'Legend: Input → CNN → SE → Temporal Attention → Classifier', fontsize=10, color='dimgray')

    return fig

if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / 'paper' / 'figures'
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = create_fig()
    out = FIGS / 'fig3_enhanced_model_dataflow.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('✅ Saved:', out)
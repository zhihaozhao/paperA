#!/usr/bin/env python3
"""
Figure 3: Enhanced Model Dataflow (2D detailed diagram)
Saves: paper/figures/fig3_enhanced_model_dataflow.pdf
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
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


def arrow(ax, x1, y1, x2, y2, color='black', elbow=False):
    if elbow:
        # Draw an elbowed arrow using 3 segments (— then | then →)
        knee_x = x1 + (x2 - x1) * 0.5
        knee_y = y1 + 0.5
        seg1 = ConnectionPatch((x1, y1), (knee_x, y1), 'data', 'data', arrowstyle='-',
                               mutation_scale=15, fc=color, ec=color, linewidth=1.8)
        seg2 = ConnectionPatch((knee_x, y1), (knee_x, knee_y), 'data', 'data', arrowstyle='-',
                               mutation_scale=15, fc=color, ec=color, linewidth=1.8)
        seg3 = ConnectionPatch((knee_x, knee_y), (x2, y2), 'data', 'data', arrowstyle='->',
                               mutation_scale=15, fc=color, ec=color, linewidth=1.8)
        ax.add_patch(seg1)
        ax.add_patch(seg2)
        ax.add_patch(seg3)
    else:
        cp = ConnectionPatch((x1, y1), (x2, y2), 'data', 'data', arrowstyle='->',
                             mutation_scale=15, fc=color, ec=color, linewidth=1.8)
        ax.add_patch(cp)


def create_fig():
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    ax.set_title('Enhanced Model Architecture (2D): Parameters per Module', fontweight='bold', color='black')

    y = 4.2
    h = 1.2

    # Input block
    x = 0.6; w = 3.0
    box(ax, x, y, w, h, 'Input CSI\n(T×C×F)', '#E6F2FF')
    arrow(ax, x + w, y + h/2, x + w + 0.6, y + h/2, elbow=True)

    # Conv Block 1
    x += w + 0.6; w = 3.4
    box(ax, x, y, w, h, 'Conv1\nKernels=32\nKernel=3×3, Stride=1', '#FDEBD0')
    ax.text(x + w/2, y - 0.38, 'BN + ReLU + Dropout(0.1)', ha='center', fontsize=9, color='dimgray')
    arrow(ax, x + w, y + h/2, x + w + 0.6, y + h/2, elbow=True)

    # Conv Block 2
    x += w + 0.6; w = 3.4
    box(ax, x, y, w, h, 'Conv2\nKernels=64\nKernel=3×3, Stride=2', '#FDEBD0')
    ax.text(x + w/2, y - 0.38, 'BN + ReLU + Dropout(0.1)', ha='center', fontsize=9, color='dimgray')
    arrow(ax, x + w, y + h/2, x + w + 0.6, y + h/2, elbow=True)

    # Conv Block 3
    x += w + 0.6; w = 3.4
    box(ax, x, y, w, h, 'Conv3\nKernels=128\nKernel=3×3, Stride=2', '#FDEBD0')
    ax.text(x + w/2, y - 0.38, 'BN + ReLU + Dropout(0.1)', ha='center', fontsize=9, color='dimgray')
    arrow(ax, x + w, y + h/2, x + w + 0.6, y + h/2, elbow=True)

    # SE Block
    x += w + 0.6; w = 3.2
    box(ax, x, y, w, h, 'SE Block\nReduction r=16\nGAP→FC(128→8)→ReLU\nFC(8→128)→Sigmoid', '#FADBD8')
    ax.text(x + w/2, y - 0.38, 'Channel-wise reweighting', ha='center', fontsize=9, color='dimgray')
    arrow(ax, x + w, y + h/2, x + w + 0.6, y + h/2, elbow=True)

    # BiLSTM
    x += w + 0.6; w = 3.4
    box(ax, x, y, w, h, 'BiLSTM\nHidden=128×2\nOutput=256 (concat)', '#D5F5E3')
    ax.text(x + w/2, y - 0.38, 'Context integration', ha='center', fontsize=9, color='dimgray')
    arrow(ax, x + w, y + h/2, x + w + 0.6, y + h/2, elbow=True)

    # Temporal Attention
    x += w + 0.6; w = 3.4
    box(ax, x, y, w, h, 'Temporal Attention\nHeads=4\nd_model=256, Window=5\nDropout=0.1', '#E8DAEF')
    ax.text(x + w/2, y - 0.38, 'Scaled Dot-Product', ha='center', fontsize=9, color='dimgray')
    arrow(ax, x + w, y + h/2, x + w + 0.6, y + h/2, elbow=True)

    # Classifier
    x += w + 0.6; w = 3.0
    box(ax, x, y, w, h, 'Classifier\nFC(256→H)\nSoftmax', '#D6EAF8')

    # Notes panel below
    ax.text(10.0, 1.2, 'Training: AdamW, lr=3e-4, batch=64; Label efficiency experiments (20% labels).\nMetrics: LOSO/LORO Macro-F1, ECE calibration; Stability Index; Deployment Readiness.',
            ha='center', va='center', fontsize=10, color='black')

    return fig

if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / 'paper' / 'figures'
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = create_fig()
    out = FIGS / 'fig3_enhanced_model_dataflow.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('✅ Saved:', out)
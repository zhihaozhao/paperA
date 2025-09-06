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
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=12, color='black', wrap=True)


def arrow(ax, x1, y1, x2, y2, color='black', elbow=False):
    if elbow:
        # Draw an elbowed arrow using two segments: horizontal then vertical with arrow head
        # Route via the target x to keep a clean L-shape
        seg1 = ConnectionPatch((x1, y1), (x2, y1), 'data', 'data', arrowstyle='-',
                               mutation_scale=15, fc=color, ec=color, linewidth=1.8)
        seg2 = ConnectionPatch((x2, y1), (x2, y2), 'data', 'data', arrowstyle='->',
                               mutation_scale=15, fc=color, ec=color, linewidth=1.8)
        ax.add_patch(seg1)
        ax.add_patch(seg2)
    else:
        cp = ConnectionPatch((x1, y1), (x2, y2), 'data', 'data', arrowstyle='->',
                             mutation_scale=15, fc=color, ec=color, linewidth=1.8)
        ax.add_patch(cp)


def create_fig():
    fig, ax = plt.subplots(figsize=(14.5, 6.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.set_title('Enhanced Model Architecture (2D): Parameters per Module', fontweight='bold', color='black')

    # Layout: 2 rows × 4 columns
    h = 1.3
    w = 4.4
    gap_x = 0.8
    gap_y = 1.4
    left = 0.8
    top_y = 6.0
    bottom_y = top_y - (h + gap_y)

    def col_x(col_idx):
        return left + col_idx * (w + gap_x)

    # Module labels (multi-line) and colors
    labels = [
        'Input CSI\n(T×C×F)',
        'Conv1\nKernels=32\nKernel=3×3, Stride=1',
        'Conv2\nKernels=64\nKernel=3×3, Stride=2',
        'Conv3\nKernels=128\nKernel=3×3, Stride=2',
        'SE Block\nReduction r=16\nGAP→FC(128→8)→ReLU\nFC(8→128)→Sigmoid',
        'BiLSTM\nHidden=128×2\nOutput=256 (concat)',
        'Temporal Attention\nHeads=4\nd_model=256, Window=5\nDropout=0.1',
        'Classifier\nFC(256→H)\nSoftmax'
    ]
    colors = ['#E6F2FF', '#FDEBD0', '#FDEBD0', '#FDEBD0', '#FADBD8', '#D5F5E3', '#E8DAEF', '#D6EAF8']

    # Top row: modules 0..3
    for i in range(4):
        x = col_x(i)
        box(ax, x, top_y, w, h, labels[i], colors[i])
        if i < 3:
            arrow(ax, x + w, top_y + h/2, col_x(i+1), top_y + h/2, elbow=False)

    # Bottom row: modules 4..7 placed from right to left
    for k, j in enumerate(range(4, 8)):
        x = col_x(3 - k)
        box(ax, x, bottom_y, w, h, labels[j], colors[j])
        if j < 7:
            # Arrow from current box's left edge to next left neighbor's right edge
            arrow(ax, x, bottom_y + h/2, col_x(3 - (k + 1)) + w, bottom_y + h/2, elbow=False)

    # Transition arrow from top row last box to bottom row first box: vertical drop at rightmost col
    arrow(ax, col_x(3) + w/2, top_y, col_x(3) + w/2, bottom_y + h, elbow=False)

    # Notes panel below
    ax.text(left + 2*(w + gap_x), 1.1,
            'Training: AdamW, lr=3e-4, batch=64; 20% labels.\nMetrics: LOSO/LORO Macro-F1, ECE calibration; Stability Index; Deployment Readiness.',
            ha='center', va='center', fontsize=12, color='black')

    return fig

if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / 'paper' / 'figures'
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = create_fig()
    out = FIGS / 'fig3_enhanced_model_dataflow.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('✅ Saved:', out)
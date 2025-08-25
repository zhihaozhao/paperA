#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
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
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.set_title('Physics-Guided Sim2Real Framework', fontweight='bold')

    box(ax, 0.6, 4.2, 2.6, 1.0, 'Physics Modeling\n(Multipath, Human, Environment)', '#D4EDDA')
    box(ax, 3.7, 4.2, 2.6, 1.0, 'Synthetic CSI Generator\n(Parameterized, Caching)', '#FFF3CD')
    box(ax, 6.8, 4.2, 2.6, 1.0, 'Enhanced Model\n(CNN + SE + Attention)', '#F8D7DA')

    arrow(ax, 3.2, 4.7, 3.7, 4.7)
    arrow(ax, 6.3, 4.7, 6.8, 4.7)

    box(ax, 2.2, 2.5, 2.6, 1.0, 'Synthetic Robustness (SRD)\n540 configurations', '#E1F5FE')
    box(ax, 5.2, 2.5, 2.6, 1.0, 'CDAE\nLOSO / LORO', '#E1F5FE')
    box(ax, 8.0, 2.5, 2.6, 1.0, 'STEA\nLabel Efficiency', '#E1F5FE')

    arrow(ax, 4.0, 4.2, 3.5, 3.5)
    arrow(ax, 7.1, 4.2, 6.5, 3.5)
    arrow(ax, 8.7, 4.2, 8.8, 3.5)

    ax.text(5.0, 1.0, 'Key Results: 83.0±0.1% (LOSO=LORO); 82.1% @ 20% labels; ECE=0.0072',
            ha='center', fontsize=11, color='darkblue')

    return fig

if __name__ == '__main__':
    fig = create_fig()
    for out in ['figure2_physics-guided.pdf', 'figure6.pdf']:
        try:
            fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f'✅ Saved {out}')
        except Exception:
            pass


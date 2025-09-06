#!/usr/bin/env python3
"""
Figure 3 (3D): Enhanced Model Dataflow (brick layout)
Saves: paper/figures/fig3_enhanced_model_dataflow_3d.pdf
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def cuboid(center, size):
    cx, cy, cz = center; sx, sy, sz = size
    x = [cx - sx/2, cx + sx/2]
    y = [cy - sy/2, cy + sy/2]
    z = [cz - sz/2, cz + sz/2]
    v = np.array([[x[0], y[0], z[0]],[x[1], y[0], z[0]],[x[1], y[1], z[0]],[x[0], y[1], z[0]],
                  [x[0], y[0], z[1]],[x[1], y[0], z[1]],[x[1], y[1], z[1]],[x[0], y[1], z[1]]])
    faces = [[v[j] for j in [0,1,2,3]], [v[j] for j in [4,5,6,7]], [v[j] for j in [0,1,5,4]],
             [v[j] for j in [2,3,7,6]], [v[j] for j in [1,2,6,5]], [v[j] for j in [4,7,3,0]]]
    return faces

def brick(ax, center, size, color, label=None, label_offset=(0,0,0.55)):
    p = Poly3DCollection(cuboid(center, size), facecolors=color, edgecolors='black', linewidths=0.8, alpha=0.95)
    ax.add_collection3d(p)
    if label:
        lx = center[0] + label_offset[0]
        ly = center[1] + label_offset[1]
        lz = center[2] + label_offset[2]*size[2]
        ax.text(lx, ly, lz, label, ha='center', va='bottom')

def create_fig():
    fig = plt.figure(figsize=(9.0, 6.0))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Enhanced Model Architecture (3D)')

    ax.set_xlim(0, 20); ax.set_ylim(0, 10); ax.set_zlim(0, 6)
    ax.view_init(elev=22, azim=-45)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()

    bx, by, bz = 2.6, 1.4, 0.7

    # Left-to-right bricks representing dataflow
    x0, y0, z0 = 3.0, 3.0, 0.9
    stages = [
        ('Input CSI\n(T×C×F)', '#E6F2FF'),
        ('Conv1\n32@3×3 s=1', '#FDEBD0'),
        ('Conv2\n64@3×3 s=2', '#FDEBD0'),
        ('Conv3\n128@3×3 s=2', '#FDEBD0'),
        ('SE r=16', '#FADBD8'),
        ('BiLSTM\nHidden 128×2', '#D5F5E3'),
        ('Temporal Attn\nHeads=4 d=256', '#E8DAEF'),
        ('Classifier\nFC→Softmax', '#D6EAF8')
    ]
    for i, (lab, col) in enumerate(stages):
        brick(ax, (x0 + i*(bx+0.5), y0, z0), (bx, by, bz), col, label=lab)

    # Layering effect: add a second row (small offset) for depth impression on key blocks
    for i in [1, 3, 6]:
        brick(ax, (x0 + i*(bx+0.5), y0+1.0, z0+0.5), (bx*0.9, by*0.9, bz*0.6), '#FFFFFF', label=None)

    return fig

if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / 'paper' / 'figures'
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = create_fig()
    out = FIGS / 'fig3_enhanced_model_dataflow_3d.pdf'
    fig.savefig(out, dpi=300, facecolor='white', edgecolor='none')
    print('✅ Saved:', out)


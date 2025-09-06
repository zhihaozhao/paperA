#!/usr/bin/env python3
"""
Figure 2 (3D): Physics-Guided Sim2Real Framework (brick layout)
Saves: paper/figures/fig2_physics_guided_framework_3d.pdf
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Fonts: text 12, title 14
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

def cuboid_data(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    # 8 vertices
    x = [cx - sx/2, cx + sx/2]
    y = [cy - sy/2, cy + sy/2]
    z = [cz - sz/2, cz + sz/2]
    # vertices of a cube
    verts = np.array([[x[0], y[0], z[0]],
                      [x[1], y[0], z[0]],
                      [x[1], y[1], z[0]],
                      [x[0], y[1], z[0]],
                      [x[0], y[0], z[1]],
                      [x[1], y[0], z[1]],
                      [x[1], y[1], z[1]],
                      [x[0], y[1], z[1]]])
    faces = [[verts[j] for j in [0,1,2,3]],
             [verts[j] for j in [4,5,6,7]],
             [verts[j] for j in [0,1,5,4]],
             [verts[j] for j in [2,3,7,6]],
             [verts[j] for j in [1,2,6,5]],
             [verts[j] for j in [4,7,3,0]]]
    return faces

def draw_brick(ax, center, size, color, edge_color='black', alpha=0.95, label=None, label_offset=(0,0,0.55)):
    faces = cuboid_data(center, size)
    poly3d = Poly3DCollection(faces, facecolors=color, edgecolors=edge_color, linewidths=0.8, alpha=alpha)
    ax.add_collection3d(poly3d)
    if label:
        lx = center[0] + label_offset[0]
        ly = center[1] + label_offset[1]
        lz = center[2] + label_offset[2]*size[2]
        ax.text(lx, ly, lz, label, zdir='z', ha='center', va='bottom')

def create_fig():
    fig = plt.figure(figsize=(8.8, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Physics-Guided Sim2Real Framework (3D)')

    # Axes ranges and view
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 5)
    ax.view_init(elev=22, azim=-45)
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        pass
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Layered bricks: left (physics stack), middle (synthesis stack), right (enhanced model stack), far right (trustworthy eval)
    # Common brick size
    bx, by, bz = 2.2, 1.2, 0.6

    # Left stack: Physics modeling (3 bricks)
    base_x, base_y = 2.0, 2.0
    colors_left = ['#D4EDDA', '#CFEBD4', '#BFE3C6']
    labels_left = ['Multipath', 'Human Interaction', 'Environment']
    for i in range(3):
        draw_brick(ax, (base_x, base_y, 0.8 + i*(bz+0.15)), (bx, by, bz), colors_left[i], label=labels_left[i])

    # Middle stack: Synthetic CSI generator (2 bricks)
    base_x_m, base_y_m = 6.0, 2.0
    colors_mid = ['#FFF3CD', '#FFE8A1']
    labels_mid = ['Parameterized Synthesis', 'Caching Engine']
    for i in range(2):
        draw_brick(ax, (base_x_m, base_y_m, 0.8 + i*(bz+0.15)), (bx, by, bz), colors_mid[i], label=labels_mid[i])

    # Right stack: Enhanced model (4 bricks)
    base_x_r, base_y_r = 9.5, 2.0
    colors_right = ['#FDEBD0', '#FADBD8', '#D5F5E3', '#E8DAEF']
    labels_right = ['Conv Blocks', 'SE Attention', 'BiLSTM', 'Temporal Attention']
    for i in range(4):
        draw_brick(ax, (base_x_r, base_y_r, 0.8 + i*(bz+0.15)), (bx, by, bz), colors_right[i], label=labels_right[i])

    # Far right: Trustworthy evaluation (single large brick)
    draw_brick(ax, (11.5, 5.2, 1.4), (2.2, 1.6, 1.0), '#D6EAF8', label='Trustworthy\nEvaluation')

    # Titles on stacks (as 3D text near top bricks)
    ax.text(2.0, 1.2, 2.8, 'Physics Modeling', ha='center', va='bottom')
    ax.text(6.0, 1.2, 2.1, 'Synthetic CSI', ha='center', va='bottom')
    ax.text(9.5, 1.2, 3.2, 'Enhanced Model', ha='center', va='bottom')

    # Hide panes
    ax.set_axis_off()
    return fig

if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / 'paper' / 'figures'
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = create_fig()
    out = FIGS / 'fig2_physics_guided_framework_3d.pdf'
    fig.savefig(out, dpi=300, facecolor='white', edgecolor='none')
    print('✅ Saved:', out)


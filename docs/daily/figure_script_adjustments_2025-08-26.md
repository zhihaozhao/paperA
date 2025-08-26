# Figure scripts – size/position adjustments (2025-08-26)

This note summarizes how size and layout are controlled in each figure script. It points to the exact constructs (APIs/variables) that change panel sizes, axis positions, inter-panel spacings, and legend placements.

## paper/PlotPy/scr2_physics_guided_framework.py (Figure 2)
- Overall canvas and axis bounds
  - `fig, ax = plt.subplots(figsize=(10.5, 6.0))`
  - `ax.set_xlim(0, 12)`; `ax.set_ylim(-0.8, 7.8)` (expanded from 0–6.5 to avoid clipping after shifting the left column downward)
- Left-column vertical layout (bottom-to-top)
  - Base and spacing: `h=1.0`, `spacing=0.35`, `base_y = 0.7 - (h + spacing)`
  - Box placement loop creates five modules with increasing `y_current += h + spacing`
- Arrow routing
  - Vertical connectors between stacked blocks: `arrow(ax, left_x + w/2, y1, left_x + w/2, y2)`
- Right dashed container and internal blocks
  - Dashed container increased to fit multi-line text: `dashed_x, dashed_y, dashed_w, dashed_h = 7.2, 0.6, 4.6, 5.2`
  - Internal block geometry enlarged to prevent overflow:
    - `bx_w = 3.9`, `bx_h = 1.20`, with stacked spacing `0.25`
  - First-layer text explicitly wrapped (e.g., `Conv Blocks\n(3×3)\nC=[32,64,128]\nStride=[1,2,2]`)
- Cross-panel arrow
  - From Enhanced (left) to dashed container center-right: `arrow(ax, left_x + w, enhanced_y + h/2, dashed_x, dashed_y + dashed_h/2)`

## paper/PlotPy/scr3_enhanced_model_dataflow.py (Figure 3)
- Overall canvas and axis bounds
  - `fig, ax = plt.subplots(figsize=(12.5, 6.2))`
  - `ax.set_xlim(0, 20)`; `ax.set_ylim(0, 7.5)`
- Two-row, four-column layout (2×4)
  - Column helper: `def col_x(col_idx): return left + col_idx * (w + gap_x)`
  - Row anchors: `top_y = 6.0`; `bottom_y = top_y - (h + gap_y)`
  - Box widths are widened to contain multi-line labels: `w = 3.0/3.4`; `h = 1.2`
- Elbow arrows and routing
  - Robust L-shaped connectors for straight segments without angle errors:
    - `elbow=True` draws two segments (horizontal then vertical) ending with an arrowhead
  - Bottom row flows right→left: boxes placed from rightmost to leftmost; arrows point sequentially left
  - Top→bottom transition via vertical drop at rightmost column: `arrow(ax, col_x(3)+w/2, top_y, col_x(3)+w/2, bottom_y + h)`
- Text wrapping
  - All module labels use explicit `\n` to enforce wrapping within wider boxes

## paper/PlotPy/scr4_experimental_protocols.py (Figure 4)
- Global fonts
  - `rcParams` unified: text/ticks/legend 12, titles 14
- Canvases
  - Comprehensive protocols: `plt.subplots(figsize=(17.5, 12))`
  - Flowchart: `plt.subplots(figsize=(14, 8))`
  - Double-column overview: `fig_over.set_size_inches(7.8, 5.2)`
- Axis bounds (comprehensive)
  - `ax.set_xlim(0, 18)`; `ax.set_ylim(2, 12)`
- Protocol boxes widened/tallened to avoid overflow
  - `create_protocol_box(..., width=6.0, height=2.0)` with taller title band (0.5 high)
  - D2 step boxes: increased sizes `(1.6, 0.8)`; CDAE/LOSO/LORO glyphs enlarged; STEA circles radius `0.24`
  - Integration/PSTA/ESTA boxes enlarged (e.g., widths 3.6–4.4; heights 0.9)
- Text wrapping and placement
  - Titles 14pt; content 12pt with `wrap=True` and padding; section annotations raised accordingly

## paper/PlotPy/scr5_cross_domain.py (Figure 5)
- Global fonts
  - `rcParams` unified: labels/ticks/legend 12; titles 14
- Base layout
  - `fig = plt.figure(figsize=(19.2, 16.8))` (1.5× original canvas)
  - `gs = GridSpec(3, 2, height_ratios=[1.35, 1.05, 1.1], hspace=0.82, wspace=0.36)`
  - Additional margins: `plt.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.08)`
- Individual panel adjustments
  - (a) Heatmap auto aspect; colorbar fraction/pad set
  - (b) Radar: pushed further down by manual axis position edit:
    - `pos = ax2.get_position(); ax2.set_position([pos.x0, pos.y0-0.05, pos.width, pos.height])`
  - (c) Correlation: rotated xticks `rotation=45, ha='right'`
  - (d) Composite ranking: restored bars; nudged downward:
    - `pos4 = ax4.get_position(); ax4.set_position([pos4.x0, pos4.y0-0.03, pos4.width, pos4.height])`
  - (e) LOSO/LORO line plot: nudged downward similarly; rotated xticks 45°
- Targeted enlargement of (b) and (c)
  - Post-layout axis scaling utility:
    - `scale_axes(ax, w_scale=1.5, h_scale=1.5)` re-centers and clamps within [left,right]/[bottom,top]
    - Applied only to `ax2` (b) and `ax3` (c)

## paper/PlotPy/scr6_pca_analysis.py (Figure 6)
- Global fonts
  - `rcParams` unified: labels/ticks/legend 12; titles 14
- Base layout (restored heights)
  - `fig = plt.figure(figsize=(16.0, 14.0))`
  - `gs = GridSpec(4, 2, height_ratios=[1.2, 1.0, 1.2, 1.1], hspace=0.56, wspace=0.18)`
- Panel-specific width controls
  - (d) Model separation heatmap width preserved by moving colorbar to a side axes via axes_grid1:
    - `divider = make_axes_locatable(ax3); cax = divider.append_axes('right', size='3%', pad=0.05)`
  - (e) 3D plot visual area and legend placement:
    - `ax5.set_box_aspect((1.3, 1.0, 0.8))`; `ax5.set_proj_type('ortho')`
    - Legend outside left: `ax5.legend(..., bbox_to_anchor=(-0.22, 0.5))`
- General spacing
  - `plt.subplots_adjust(left=0.06, bottom=0.07, right=0.98, top=0.95)`

## Commit references (chronological highlights)
- fig5: targeted (b)/(c) enlargement, (d) data restore, rotated ticks — 3173c1d, f0b0a1d, 497263f
- fig6: row3 enlargement, legend shift, width preservation, unified fonts — 628a096, 3ee9c47, 53e5640, a74cb7c
- fig2: 2D redesign, reversed left column, taller right-box blocks — 07915a6, 6a8caed, 6356c26, a74cb7c
- fig3: 2D redesign, 2×4 layout, elbow arrows, bottom row R→L — 10aea7a, 7c58d91
- fig4: fonts and widths unified; overflow fixes — 6b52a91
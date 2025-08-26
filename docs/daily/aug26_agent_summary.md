### Daily Summary (Aug 26) — Agent Worklog

- Thesis build fixes
  - Added robust XeLaTeX setup in `Thesis/main.tex` and Chinese thesis, resolved CJK font issues, `\headheight` warning, missing `physics.sty`/`IEEEtran.bst`, and escaped LaTeX percent in appendices.
  - Temporarily skipped corrupted chapter include; ensured clean build path.

- Git sync and results mirroring
  - Mirrored `results_gpu` and `results` from `results/main` into `feat/enhanced-model-and-sweep`.
  - Resolved push rejections via `git pull --rebase`; handled transient `git mv` state.

- D5/D6 → PSTA/ESTA analysis, plots, and paper integration
  - Implemented parser/summary/plot pipeline: `paper/PlotPy/plot_d5_d6.py`.
  - Produced CSV summaries and a composite figure with two panels (Macro F1, Brier) and legends.
  - Renamed terminology across paper and plots from D5/D6 to academic names PSTA/ESTA.
  - Updated `paper/main.tex`: detailed subsection (motivation, design, how-to, training, validation), captions, table header, and references; inlined results table.

- Figure normalization and script relocation
  - Normalized figure naming in `paper/main.tex` (e.g., `figure2_experimental_protocols.pdf`, `figure5_cross_domain.pdf`, `figure7_label_efficiency.pdf`).
  - Moved plotting logic into `paper/PlotPy/` and kept thin runners in `paper/figures/`.
  - Added aliases where needed to preserve existing includes.

- Figure 2 (protocols) updates
  - Enhanced `paper/figures/figure2-experimental-protocols.py` to annotate PSTA/ESTA in the overview.
  - Canonical output ensured: `figures/figure2_experimental_protocols.pdf` (replaces legacy `figure-4.pdf`).

- Documentation collation
  - Created `scripts/collate_docs.py` to gather Markdown into `docs/collated/` and generate `docs/collated/MANIFEST.md`.
  - Added `docs/FIGURE_CATALOG.md` describing each figure, script, and rationale.
  - Wrote `docs/daily/aug26.md` previously; this file adds an agent-focused log.

- Build/runtime setup and troubleshooting
  - Installed missing TeX packages and Python libs (numpy/matplotlib via apt). Avoided pip EME restriction.
  - Addressed LaTeX syntax errors (escaped `%`, removed math-mode `±` where problematic in generated LaTeX).
  - Provided Windows guidance for Git unlink failures (close lockers, `attrib`, `icacls`, force restore).

- Commit references
  - Move plot script and imports: chore(plots): move plot_d5_d6.py to paper/PlotPy and update runner imports.
  - Composite updates: style(figures): D5/D6 composite -> 2 panels (F1,Brier), legends, rotated ticks.
  - Terminology: docs(paper): rename D5/D6 to PSTA/ESTA; update legends, captions, table headers.

How to regenerate locally
- D5/D6 (PSTA/ESTA) figures:
  - `python3 paper/figures/plot_d5_d6.py`
  - Outputs: `paper/figures/d5_d6_composite.pdf` and PNG.
- Protocols overview (Figure 2):
  - Ensure seaborn installed or comment it out
  - `python3 paper/figures/figure2-experimental-protocols.py`
  - Outputs: `paper/figures/figure2_experimental_protocols.pdf`
- Build paper:
  - Compile `paper/main.tex` in your LaTeX environment after pulling.

Next steps (handoff)
- Validate regenerated figures on your machine; update colors/fonts if journal style requires.
- Ensure all figures follow ordered naming: `figure1_...`, `figure2_...`, etc. Current key files:
  - Keep: `figures/figure1_system_architecture.pdf`
  - Use: `figures/figure2_experimental_protocols.pdf` (protocols with PSTA/ESTA)
  - Paper references updated accordingly; review for any lingering legacy names.
- If Windows shows unlink errors, remove file locks/read-only then `git restore --worktree -- paper/main.tex`.
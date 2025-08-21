# Reproducibility Package (Thesis)

This folder aggregates the figures, data and scripts used in the thesis review (DFHAR) and experiments chapters.

## Structure
- `figures/`: high-resolution images used in the thesis (subset from DFHAR and generated plots)
- `data/`: CSVs that back the quantitative figures/tables
- `scripts/`: scripts to regenerate plots from CSVs (Python 3; requires pandas, matplotlib)

## Quick start
1. (Optional) Create a virtual environment and install deps
   ```bash
   python -m venv .venv && .venv\Scripts\activate  # Windows
   pip install pandas matplotlib
   ```
2. Generate plots
   ```bash
   python scripts/gen_har_plots.py
   ```
3. Figures will be saved under `figures/`.

## Data sources
- DFHAR repository: `https://github.com/zhihaozhao/DFHAR`
- Internal benchmark CSVs: `data/har_master.csv`

## License & citation
- Please cite the thesis and DFHAR repository when reusing the materials.

# 📚 Paper Supplementary Materials

This directory contains all supplementary materials for the PASE-Net paper submission to IEEE TMC.

## Directory Structure

```
paper_supplementary/
├── README.md                    # This file
├── scripts/                     # All data processing and figure generation scripts
│   ├── data_extraction/         # Scripts to extract data from experiments
│   ├── figure_generation/       # Scripts to generate paper figures
│   └── validation/              # Data validation and consistency checks
├── data/                        # Extracted and processed data
│   ├── raw/                     # Raw experimental results (linked)
│   └── processed/               # Processed data for figures and tables
├── figures/                     # Generated figures
│   ├── main/                    # Main paper figures
│   └── supplementary/           # Additional analysis figures
└── docs/                        # Documentation and analysis reports
    ├── data_pipeline/           # Data processing documentation
    ├── analysis_reports/        # Various analysis reports
    └── submission/              # Journal submission materials
```

## Quick Start

### 1. Extract All Data
```bash
cd scripts/data_extraction
python3 extract_all_data.py
```

### 2. Generate All Figures
```bash
cd scripts/figure_generation
python3 generate_all_figures.py
```

### 3. Validate Data Consistency
```bash
cd scripts/validation
python3 validate_all_data.py
```

## Main Paper Files

The main paper files remain in:
- `/workspace/paper/enhanced/enhanced_claude_v1.tex` - Main LaTeX document
- `/workspace/paper/enhanced/enhanced_refs.bib` - Bibliography
- `/workspace/paper/enhanced/SUPPLEMENTARY_MATERIALS.tex` - Supplementary materials

## Data Sources

- **Experimental Results**: `/workspace/results_gpu/` (668+ files)
- **Benchmark Dataset**: WiFi-CSI-Sensing-Benchmark (real WiFi CSI data)
- **Processed Data**: `data/processed/` (extracted metrics)

## Key Results

| Metric | Value | Source |
|--------|-------|--------|
| LOSO Performance | 83.0% | Real WiFi CSI data |
| LORO Performance | 83.0% | Real WiFi CSI data |
| Calibration (ECE) | 0.094→0.001 | Real test data |
| Fall Detection | >99% | Real LOSO experiments |
| Label Efficiency | 82% @ 20% labels | Sim2Real experiments |

## Version Control

- **Current Version**: TMC_v1
- **Branch**: feat/enhanced-model-and-sweep
- **Last Updated**: 2024-12-04

## Contact

For questions about the supplementary materials, please refer to the corresponding author.
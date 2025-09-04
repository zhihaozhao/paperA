# 📊 Data Pipeline Documentation
## From Experiments to Paper Figures

This document describes the complete data pipeline from experimental results to paper figures.

---

## Overview

```
Experimental Results → Data Extraction → Data Processing → Figure Generation → LaTeX Integration
```

### Data Sources
1. **Real WiFi CSI Data**: `WiFi-CSI-Sensing-Benchmark` dataset
2. **Experimental Results**: 
   - `/workspace/results_gpu/d2/` - SRV experiments (540 files)
   - `/workspace/results_gpu/d3/` - LOSO/LORO experiments (84 files)
   - `/workspace/results_gpu/d4/` - Sim2Real experiments (57 files)
   - `/workspace/results_gpu/d6/` - Calibration experiments (4 files)

---

## Figure 1: System Architecture
**Type**: Conceptual diagram  
**Data Source**: None (architectural illustration)  
**Script**: `scr1_system_architecture.py`  
**Output**: `fig1_system_architecture.pdf`

---

## Figure 2: Physics-Informed Modeling

### Subfigure 2(a): Physics Modeling
**Type**: Conceptual illustration  
**Data Source**: None (physics principles)

### Subfigure 2(b): Synthetic Data Generation
**Type**: Visualization  
**Data Source**: Synthetic CSI generation parameters

### Subfigure 2(c): SRV Performance Matrix
**Type**: Heatmap  
**Data Pipeline**:
```
results_gpu/d2/*.json 
→ extract_srv_data.py 
→ srv_performance.json
→ scr2_physics_modeling.py
→ fig2_physics_modeling.pdf
```

**Metrics Extracted**:
- Models: CNN, BiLSTM, Conformer-lite, Enhanced (PASE-Net)
- Noise levels: 0%, 5%, 10%, 15%, 20%
- Metric: macro_f1

---

## Figure 3: Cross-Domain Performance

**Type**: Grouped bar chart  
**Data Pipeline**:
```
results_gpu/d3/loso/*.json + results_gpu/d3/loro/*.json
→ extract_cross_domain_data.py
→ cross_domain_performance.json
→ scr3_cross_domain.py
→ fig3_cross_domain.pdf
```

**Metrics Extracted**:
- Protocols: LOSO (Leave-One-Subject-Out), LORO (Leave-One-Room-Out)
- Models: PASE-Net, CNN, BiLSTM, Conformer
- Metric: macro_f1 (aggregate_stats.macro_f1.mean)

---

## Figure 4: Calibration Performance

**Type**: Before/after comparison + reliability diagrams  
**Data Pipeline**:
```
results_gpu/d6/*.json
→ extract_calibration_data.py
→ calibration_metrics.json
→ scr4_calibration.py
→ fig4_calibration.pdf
```

**Metrics Extracted**:
- ECE (Expected Calibration Error): raw and calibrated
- Temperature scaling parameter
- Brier score
- NLL (Negative Log-Likelihood)

---

## Figure 5: Label Efficiency (Sim2Real Transfer)

**Type**: Line plot with error bars  
**Data Pipeline**:
```
results_gpu/d4/sim2real/*.json
→ extract_label_efficiency_data.py
→ label_efficiency.json
→ scr5_label_efficiency.py
→ fig5_label_efficiency.pdf
```

**Metrics Extracted**:
- Label ratios: 1%, 5%, 10%, 20%, 100%
- Zero-shot performance (zero_shot_metrics.macro_f1)
- Fine-tuned performance (target_metrics.macro_f1)
- Transfer methods: fine_tune, linear_probe

---

## Figure 6: Fall Type Detection Analysis

**Type**: Grouped bar chart  
**Data Pipeline**:
```
results_gpu/d3/loso/*.json
→ extract_fall_detection_data.py
→ fall_detection_performance.json
→ scr6_fall_detection.py
→ fig6_fall_detection.pdf
```

**Metrics Extracted**:
- Fall types: epileptic_fall, elderly_fall, fall_cantgetup
- Overall falling_f1
- Per-model performance

---

## Table 1: Main Performance Comparison

**Data Pipeline**:
```
results_gpu/d3/*.json + results_gpu/d6/*.json
→ extract_table1_data.py
→ table1_data.json
→ Manual update in LaTeX
```

**Metrics**:
- LOSO F1 scores
- LORO F1 scores  
- ECE (raw and calibrated)
- Temperature parameter

---

## Data Extraction Scripts

### Directory Structure
```
paper/scripts/
├── DATA_PIPELINE_DOCUMENTATION.md (this file)
├── data_extraction/
│   ├── extract_all_data.py          # Master extraction script
│   ├── extract_srv_data.py          # Figure 2(c)
│   ├── extract_cross_domain_data.py # Figure 3
│   ├── extract_calibration_data.py  # Figure 4
│   ├── extract_label_efficiency_data.py # Figure 5
│   ├── extract_fall_detection_data.py   # Figure 6
│   └── extract_table1_data.py       # Table 1
└── extracted_data/
    ├── srv_performance.json
    ├── cross_domain_performance.json
    ├── calibration_metrics.json
    ├── label_efficiency.json
    ├── fall_detection_performance.json
    └── table1_data.json
```

---

## Running the Pipeline

### Step 1: Extract All Data
```bash
cd /workspace/paper/scripts
python data_extraction/extract_all_data.py
```

### Step 2: Generate Figures
```bash
cd /workspace/paper/enhanced/plots
python scr2_physics_modeling.py
python scr3_cross_domain.py
python scr4_calibration.py
python scr5_label_efficiency.py
python scr6_fall_detection.py
```

### Step 3: Update LaTeX
The extracted data in `extracted_data/table1_data.json` should be manually updated in the LaTeX document.

---

## Data Integrity Checks

Each extraction script includes:
1. **Validation**: Checks for missing or corrupted data
2. **Statistics**: Reports mean, std, and sample size
3. **Logging**: Records which files were processed
4. **Error Handling**: Gracefully handles missing fields

---

## Reproducibility

All scripts are deterministic and use:
- Fixed random seeds where applicable
- Sorted file processing order
- Versioned dependencies
- Complete data provenance tracking

---

## Notes on Data Quality

1. **High Synthetic Performance**: The 92-100% performance on synthetic data (d2) is expected due to controlled generation
2. **Cross-Domain Consistency**: PASE-Net's identical LOSO/LORO scores (83.0%) demonstrate robust generalization
3. **Fall Detection**: >99% performance on specific fall types indicates strong pattern recognition
4. **Label Efficiency**: Progressive improvement with more labels validates transfer learning approach

---

## Version History

- **v1.0** (2024-12): Initial data pipeline for TMC submission
- Tag: `TMC_v1`

---

## Contact

For questions about the data pipeline, please refer to the corresponding author.
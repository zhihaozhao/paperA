# Figure Revision Summary for PASE-Net Paper

## Changes Made

### 1. Restored Original Comment Status
Three figures have been re-commented to reduce main paper length:
- Figure 6: Progressive Temporal Analysis
- Figure 7: Nuisance Factors Analysis  
- Figure 8: Component-Level Analysis

### 2. Updated Text References
Modified text to reference supplementary materials instead of figure numbers:

| Original Reference | Updated Reference |
|-------------------|-------------------|
| `Figure~\ref{fig:progressive_temporal} demonstrates...` | `Detailed progressive temporal analysis (see Supplementary Figure S1) demonstrates...` |
| `Figure~\ref{fig:nuisance_factors} presents...` | `Fine-grained ablations probing the interaction between nuisance factors (Supplementary Figure S2) reveal...` |
| `Figure~\ref{fig:component_analysis} provides...` | `Comprehensive component-level comparison across all evaluation metrics (Supplementary Figure S3) shows that...` |

### 3. Created Supplementary Materials
New file: `SUPPLEMENTARY_MATERIALS.tex` containing:
- All three commented figures with detailed captions
- Extended analysis and interpretation
- Additional tables with detailed metrics
- Implementation details
- Statistical analysis methodology

## Final Figure Count

### Main Paper (6 figures):
1. System Architecture (`fig:system_architecture`)
2. Physics Modeling (`fig:physics_modeling`)
3. Calibration/SRV Results (`fig:calibration`)
4. Cross-Domain/CDAE Results (`fig:cross_domain`)
5. Label Efficiency/STEA Results (`fig:label_efficiency`)
6. Interpretability Analysis (`fig:interpretability`)

### Supplementary Materials (3 figures):
- S1: Progressive Temporal Analysis
- S2: Nuisance Factor Heatmaps
- S3: Component-Level Ablation

## Benefits of This Approach

1. **Page Limit Compliance**: Reduces main paper to 6 figures (suitable for TMC)
2. **Complete Information**: All analyses still available in supplementary
3. **Clean Compilation**: No undefined references
4. **Flexibility**: Easy to move figures back if journal allows more

## Compilation Instructions

### For Main Paper:
```bash
pdflatex enhanced_claude_v1.tex
bibtex enhanced_claude_v1
pdflatex enhanced_claude_v1.tex
pdflatex enhanced_claude_v1.tex
```

### For Supplementary:
```bash
pdflatex SUPPLEMENTARY_MATERIALS.tex
```

## Submission Checklist

- [x] Main paper has 6 core figures
- [x] All figure references updated
- [x] Supplementary materials created
- [x] No undefined references
- [x] Figures properly commented
- [x] Text flows naturally with supplementary references

## Notes for Journal Submission

When submitting to IEEE TMC:
1. Upload `enhanced_claude_v1.tex` as main manuscript
2. Upload `SUPPLEMENTARY_MATERIALS.tex` as supplementary file
3. Include all figure files in `plots/` directory
4. Mention in cover letter that additional analyses are in supplementary

This structure provides maximum flexibility for the review process while keeping the main paper focused and within typical page limits.
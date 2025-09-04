# Figure Manifest for Paper 2 (PASE-Net)

## Main Paper Figures (6)

| Figure | File | Description | Data Source |
|--------|------|-------------|-------------|
| Fig. 1 | `fig1_system_architecture.pdf` | PASE-Net architecture | Conceptual diagram |
| Fig. 2 | `fig2_physics_modeling_new.pdf` | Physics-informed synthesis | Partial real SRV data |
| Fig. 3 | `fig3_calibration.pdf` | Calibration analysis | Real calibration data |
| Fig. 4 | `fig4_cross_domain.pdf` | Cross-domain performance | Real LOSO/LORO data |
| Fig. 5 | `fig5_label_efficiency.pdf` | Label efficiency | Real Sim2Real data |
| Fig. 6 | `fig6_interpretability.pdf` | Interpretability analysis | Visualization |

## Supplementary Figures (5)

| Figure | File | Description | Data Source |
|--------|------|-------------|-------------|
| Supp. S1 | `s1_cross_domain_multisubplot.pdf` | Extended cross-domain (4 subplots) | Real + analysis |
| Supp. S2 | `s2_label_efficiency_multisubplot.pdf` | Extended label efficiency (4 subplots) | Real + analysis |
| Supp. S3 | `s3_progressive_temporal.pdf` | Progressive temporal analysis | To be verified |
| Supp. S4 | `s4_ablation_noise_env.pdf` | Nuisance factor heatmaps | 135 real experiments |
| Supp. S5 | `s5_ablation_components.pdf` | Component ablation | To be verified |

## Caption Updates in Main Paper

### Original (with 4 subplots mentioned) → Simplified

1. **Fig. 3 (Calibration)**: Simplified caption, removed (a)(b)(c)(d) references
2. **Fig. 4 (Cross-Domain)**: Simplified caption, moved detailed analysis to Supp. S1
3. **Fig. 5 (Label Efficiency)**: Simplified caption, moved detailed analysis to Supp. S2

## Generation Scripts

### Main Figures
- `scr1_system_architecture.py`
- `scr2_physics_modeling.py`
- `scr3_cross_domain.py` → generates fig4
- `scr4_calibration.py` → generates fig3
- `scr5_label_efficiency.py`
- `scr6_interpretability.py`

### Supplementary Figures
- `generate_fig3_multisubplot.py` → generates s1
- `generate_fig5_multisubplot.py` → generates s2
- `generate_ablation_heatmap_real.py` → generates s4

## File Organization

```
plots/
├── Main paper figures (fig1-6)
├── Supplementary figures (s1-s5)
├── Generation scripts (scr*.py, generate*.py)
├── Verification scripts (verify*.py, debug*.py)
└── plots_backup/ (old/draft versions)
```

## Data Verification Status

| Figure | Real Data | Verified | Last Updated |
|--------|-----------|----------|--------------|
| Fig 1 | N/A | ✅ | - |
| Fig 2 | Partial | ⚠️ | 12:57 |
| Fig 3 | Yes | ✅ | 12:57 |
| Fig 4 | Yes | ✅ | 13:44 |
| Fig 5 | Yes | ✅ | 13:44 |
| Fig 6 | N/A | ✅ | 12:57 |
| S1 | Yes | ✅ | 14:10 |
| S2 | Yes | ✅ | 14:11 |
| S3 | TBD | ⚠️ | - |
| S4 | Yes | ✅ | 14:00 |
| S5 | TBD | ⚠️ | - |

---

**Last Updated**: 2024-12-04
**Status**: Ready for submission with main figures using real data
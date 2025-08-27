# Experiment Results Template

## Overall Performance

### Table 1: Main Results
| Model | F1 Score | Accuracy | Precision | Recall | Params | FLOPs | Latency |
|-------|----------|----------|-----------|--------|--------|-------|---------|
| CNN | | | | | 0.8M | 120M | |
| LSTM | | | | | 2.1M | 350M | |
| BiLSTM | | | | | 3.5M | 580M | |
| Conformer | | | | | 5.2M | 680M | |
| Enhanced | | | | | 1.2M | 180M | |
| **Exp1 (PINN)** | | | | | 1.0M | 150M | |
| **Exp2 (Mamba)** | | | | | 1.5M | 250M | |

### Table 2: Cross-Domain Results (CDAE)
| Model | LOSO F1 | LORO F1 | Domain Gap |
|-------|---------|---------|------------|
| Baseline | | | |
| Ours | | | |

### Table 3: Few-Shot Results (STEA)
| Model | 1% | 5% | 20% | 100% |
|-------|-----|-----|-----|------|
| Baseline | | | | |
| Ours | | | | |

## Per-Dataset Performance

### SignFi Results
- Samples: 276 gestures × 5 users
- F1 Score: 
- Confusion Matrix: [To be added]

### Widar Results
- Samples: 22 gestures × 17 users
- F1 Score:
- Confusion Matrix: [To be added]

### UT-HAR Results
- Samples: 7 activities × 6 subjects
- F1 Score:
- Confusion Matrix: [To be added]

## Ablation Studies

### Component Contribution
| Configuration | F1 Score | Δ |
|--------------|----------|---|
| Full Model | | - |
| w/o Component A | | |
| w/o Component B | | |
| w/o Component C | | |

## Computational Efficiency

### Edge Device Performance
| Device | Model | Latency | Throughput | Memory | Power |
|--------|-------|---------|------------|--------|-------|
| Jetson Xavier | | | | | |
| Raspberry Pi 4 | | | | | |
| Intel NUC | | | | | |

## Statistical Significance
- Paired t-test results: p < 0.05
- Confidence intervals: ±X.X%
- Effect size (Cohen's d): X.XX
Exp1: Multi-scale LSTM + Lite Attention + PINN

Objectives
- Assess multi-scale temporal encoding with lite attention vs baselines under CDAE/STEA.
- Integrate physics-informed loss (PINN) using soft constraints derived from channel dynamics priors.

Sub-experiments
- E1: Baseline LSTM vs multi-scale LSTM (no attention)
- E2: Add lite temporal attention (single-head additive)
- E3: Add SE channel attention
- E4: Add PINN regularization (λ schedule)
- E5: Ablations (scale count, hidden dims, attention heads)

Acceptance spec
- Reproducible runs (5 seeds), macro F1/ECE/NLL reported.
- Params within ±10% across comparisons; report Params and throughput.
- CDAE: LOSO/LORO gap ≤ 2% for Enhanced variant.
- STEA: 20% labels ≥ 98% of 100% labels for Enhanced variant.

Run
- See scripts/setup.sh and scripts/train.sh

Specification — Exp1

- Goals, metrics, and acceptance criteria: see README plus this spec for details.
- Errors & adjustments:
  - Convergence issues: lower LR to 5e-4, increase epochs by 50%.
  - Overfitting: enable dropout 0.2 in LSTM layers; early stopping on val F1.
  - Params mismatch >10%: adjust hidden_dims to match within ±10%.
  - Unstable ECE: add temperature scaling post-hoc; report ECE before/after.
- Reporting:
  - Save per-seed metrics to results/metrics.json; include Params via torchinfo.

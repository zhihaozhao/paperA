Specification — Exp2

- Goals: compare Mamba vs LSTM under CDAE/STEA at matched capacity.
- Errors & adjustments:
  - mamba-ssm install failure: fallback to GRU stub; record limitation.
  - Training instability: gradient clip at 1.0; reduce LR to 5e-4.
  - Params mismatch: tune hidden_dim to match baseline within ±10%.
- Reporting: per-seed F1/ECE/NLL; throughput via timed inference.

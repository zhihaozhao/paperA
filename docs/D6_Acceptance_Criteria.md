# D6 Acceptance Criteria — Robustness, Efficiency, and Summary

## Pass Criteria
- Robustness: LOSO/LORO aggregated macro_f1 ≥ planned thresholds; Enhanced ≥ baseline by ≥5% on average.
- Label Efficiency: 10–20% labels achieve ≥90% of full-supervision performance (macro_f1), with CI covering target.
- Calibration: ECE ≤ 0.15 after calibration at key ratios (1%, 5%, 100%).
- Efficiency: Report params/FLOPs/latency for Enhanced vs baseline (±10% capacity alignment) with clear advantage or parity.
- Reporting completeness: All plots/tables generated; docs updated with reproducible commands and seeds.

## Outputs
- results/metrics/*.csv and plots/*.pdf ready for paper
- docs/D6_Final_Summary.md summarizing conclusions and recommended deployment settings

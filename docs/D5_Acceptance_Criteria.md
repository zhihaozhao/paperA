# D5 Acceptance Criteria — Ablation and Mechanism Study

## Pass Criteria
- Coverage (D2): Each ablation variant × ≥3 seeds completed; summary table generated.
- Consistency: Enhanced (full) outperforms reduced variants by ≥5% macro_f1 on average (D2), or matched capacity shows no unfair advantage.
- Calibration: With-λ (or tuned) shows lower ECE than without-λ by ≥5% relative on D2; trend holds on D4 sample points.
- Real validation: On D4 ratios 1%/5%, fine_tune outperforms linear_probe by ≥5% macro_f1 on average over seeds [0,1].
- Reporting: Plots (ablation bars, reliability diagrams) and a consolidated table with mean±std and bootstrap 95% CI.

## Outputs
- results/ablation/*.json aggregated to a CSV/Markdown summary
- plots/ablation/*.pdf (bars, calibration, reliability)
- docs/D5_Acceptance_Report.md (short narrative + tables/fig refs)

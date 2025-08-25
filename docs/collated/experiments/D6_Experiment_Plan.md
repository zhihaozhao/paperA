# D6 Experiment Plan — Robustness, Efficiency, and Cross-Domain Summary

## Objective
- Summarize cross-domain robustness (D3) and Sim2Real efficiency (D4) with efficiency analysis to form publication-ready evidence.

## Scope
- Robustness: LOSO/LORO quick-set for Enhanced (and one baseline) with aggregated stats.
- Efficiency: Params, FLOPs, latency (CPU/GPU) for Enhanced vs baselines (capacity-aligned ±10%).
- Trustworthiness: Calibration summaries and reliability diagrams at key ratios (1%, 5%, 100%).

## Minimal Runs
- D3 LOSO/LORO: Enhanced on sampled folds (≥50% coverage) × seeds=[0,1]; baseline (cnn/bilstm) on a subset.
- D4: Use finalized label-efficiency runs (from D4) + a few extra seeds if needed for CI.

## Metrics & Artifacts
- macro_f1, falling_f1, ece, auprc_falling, mutual_misclass; params/FLOPs/latency.
- Plots: label-efficiency curves (mean±std/CI), calibration diagrams, confusion matrices.

## Deliverables
- docs/D6_Acceptance_Criteria.md (pass/fail checklist)
- plots/ (curve + calibration + confusion) and tables/ (efficiency, ablation summary)
- Finalized narrative: how much label is needed to reach ≥90–95% full-supervision.

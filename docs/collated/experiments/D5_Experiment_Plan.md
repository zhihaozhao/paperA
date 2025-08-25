# D5 Experiment Plan — Ablation and Mechanism Study

## Objective
- Identify which Enhanced components and training strategies drive performance/calibration.
- Provide minimal, high-signal ablations aligned with journal expectations, compute-efficient.

## Scope
- Synthetic (D2) for comprehensive/cheap coverage; Real (D4) for a few points validating transfer trends.

## Variants
- Architecture (Enhanced base):
  - Enhanced (full)
  - Enhanced−SE (remove Squeeze-Excitation)
  - Enhanced−Attn (remove temporal attention)
  - Enhanced−SE−Attn (≈ capacity-aligned CNN)
- Training/regularization:
  - With vs without logit-L2/λ (if applicable)
  - Input normalization: zscore vs center
- Transfer on REAL (D4): linear_probe vs fine_tune

## Minimal Grid
- D2 (Synthetic): Models=[Enhanced, −SE, −Attn, −SE−Attn]; Seeds=[0–4]; Difficulty grid=existing (or reduced if needed)
- D4 (Real, Sim2Real): Ratios=[0.01, 0.05]; Methods=[linear_probe, fine_tune]; Seeds=[0,1]

## Metrics
- macro_f1, falling_f1, ece, auprc_falling, mutual_misclass
- Params/FLOPs/latency (capacity alignment & efficiency)

## Commands (examples)
- D2 ablation (reuse D2 tooling):
```
python scripts/run_sweep_from_json.py --spec scripts/d2_spec.json --resume
# or: scripts/d2_spec_reduced.json (if provided)
```
- D4 ablation (Enhanced−SE placeholder; adapt model flag/checkpoint as needed):
```
$CKPT="E:\paperA\paperA\checkpoints"
python -m src.train_cross_domain --model enhanced --seed 0 --protocol sim2real --label_ratio 0.01 --transfer_method linear_probe --d2_model_path $CKPT --skip_synth_pretrain --benchmark_path benchmarks\WiFi-CSI-Sensing-Benchmark-main --files_per_activity 3 --class_weight inv_freq --input_norm zscore --batch_size 16 --lr 0.01 --min_epochs 50 --patience 5 --out results\d4\sim2real\abl_enhanced_minus_se_lp_0p01_s0.json
python -m src.train_cross_domain --model enhanced --seed 0 --protocol sim2real --label_ratio 0.01 --transfer_method fine_tune   --d2_model_path $CKPT --skip_synth_pretrain --benchmark_path benchmarks\WiFi-CSI-Sensing-Benchmark-main --files_per_activity 3 --class_weight inv_freq --input_norm zscore --batch_size 8 --lr 0.001 --min_epochs 50 --patience 8 --out results\d4\sim2real\abl_enhanced_minus_se_ft_0p01_s0.json
```

## Timeline (est.)
- Day 1: Prepare variants/checkpoints; D4 dry-run (1–2 configs)
- Day 2–3: D2 ablations (all seeds; reduced grid if needed)
- Day 4: D4 ablations (1%,5%; seeds 0–1)
- Day 5: Aggregate, CI/stats, plots, write-ups

## Deliverables
- Results JSONs + plots (ablation bars, calibration, reliability)
- docs/D5_Acceptance_Criteria.md (pass/fail) + ablation summary table

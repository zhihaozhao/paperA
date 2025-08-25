# Runbook: D1–D6 Commands and Acceptance

This runbook lists the minimal commands to run and validate each phase. All commands assume Windows PowerShell in project root.

## Global setup
- Activate env (example):
```powershell
conda activate py310
```
- Set D2 checkpoints path (adjust as needed):
```powershell
$CKPT="E:\paperA\paperA\checkpoints"  # or "$PWD\checkpoints"
```
- Resume behavior:
  - Python: add `--resume` to skip when `--out` already exists
  - Batch: `scripts\run_d4_loro.bat` will [SKIP] existing outputs automatically

---
## D1 — Baseline sanity and plots (quick)
- (Optional) Generate D1 plots if data/results exist:
```powershell
python scripts\plot_d1_bars.py
python scripts\plot_d1_overlap_scatter.py
```
- (Optional) Lambda acceptance checks:
```powershell
python verify_lambda_acceptance.py
```

---
## D2 — Synthetic sweep, stats, acceptance
- Full/pre-generated path:
```powershell
python scripts\pregenerate_d2_datasets.py --spec scripts\d2_spec.json
python scripts\run_sweep_from_json.py --spec scripts\d2_spec.json --resume
```
- Reduced spec (if provided):
```powershell
python scripts\run_sweep_from_json.py --spec scripts\d2_spec_reduced.json --resume
```
- Analysis / report:
```powershell
python scripts\generate_d2_analysis_report.py
python scripts\validate_d2_acceptance.py
```

---
## D3 — Cross-domain (LOSO / LORO)
- Quick LOSO/LORO (Windows):
```powershell
# Example LOSO for Enhanced
python -m src.train_cross_domain --protocol loso --model enhanced --seed 0 --benchmark_path benchmarks\WiFi-CSI-Sensing-Benchmark-main --files_per_activity 3 --class_weight inv_freq --out results\d3\loso_enhanced_s0.json

# Example LORO for Enhanced
python -m src.train_cross_domain --protocol loro --model enhanced --seed 0 --benchmark_path benchmarks\WiFi-CSI-Sensing-Benchmark-main --files_per_activity 3 --class_weight inv_freq --out results\d3\loro_enhanced_s0.json
```
- Batch helpers:
```powershell
scripts\run_d3_loso.ps1
scripts\run_d3_loro.bat
```
- Acceptance:
```powershell
python scripts\validate_d3_acceptance.py --root results --out_dir results\metrics --save_report docs\d3_acceptance_report.md
```

---
## D4 — Sim2Real label efficiency
- Minimal Enhanced + fine_tune sweep (recommended):
```powershell
$ratios=0.01,0.05,0.10,0.20,1.00; $seeds=0,1,2,3,4
foreach ($r in $ratios) {
  foreach ($s in $seeds) {
    python -m src.train_cross_domain --resume --model enhanced --seed $s --protocol sim2real --label_ratio $r --transfer_method fine_tune --d2_model_path $CKPT --skip_synth_pretrain --benchmark_path benchmarks\WiFi-CSI-Sensing-Benchmark-main --files_per_activity 3 --class_weight inv_freq --input_norm zscore --batch_size 8 --lr 0.001 --min_epochs 50 --patience 8 --out results\d4\sim2real\enhanced_s${s}_bs8_lr1e-3_me50_ft_${r}.json
  }
}
```
- Optional baselines:
```powershell
# linear_probe (fast baseline)
python -m src.train_cross_domain --resume --model enhanced --seed 0 --protocol sim2real --label_ratio 0.01 --transfer_method linear_probe --d2_model_path $CKPT --skip_synth_pretrain --benchmark_path benchmarks\WiFi-CSI-Sensing-Benchmark-main --files_per_activity 3 --class_weight inv_freq --input_norm zscore --batch_size 16 --lr 0.01 --min_epochs 50 --patience 5 --out results\d4\sim2real\enhanced_s0_bs16_lr1e-2_me50_lp_0p01.json

# zero_shot
python -m src.train_cross_domain --resume --model enhanced --seed 0 --protocol sim2real --label_ratio 0.01 --transfer_method zero_shot --d2_model_path $CKPT --skip_synth_pretrain --benchmark_path benchmarks\WiFi-CSI-Sensing-Benchmark-main --files_per_activity 3 --class_weight inv_freq --input_norm zscore --out results\d4\sim2real\enhanced_s0_zs_0p01.json
```
- Full-sweep driver (batch, now supports resume):
```powershell
# By default runs the full 560-config sweep. To run a subset:
$env:USE_ENV=1
$env:MODELS="enhanced"
$env:SEEDS="0 1 2 3 4"
$env:LABEL_RATIOS="0.01 0.05 0.10 0.20 1.00"
$env:TRANSFER_METHODS="fine_tune"
scripts\run_d4_loro.bat
```
- Acceptance / export:
```powershell
python scripts\validate_d4_acceptance.py --root results\d4 --out_dir results\metrics --save_report docs\d4_acceptance_report.md
python scripts\export_d4_curves.py --root results\d4 --out_dir results\metrics
```

---
## D5 — Ablation & Mechanism (docs-driven)
- Plan & Acceptance: `docs\D5_Experiment_Plan.md`, `docs\D5_Acceptance_Criteria.md`
- Execution:
  - D2 ablations via `scripts\run_sweep_from_json.py` (full or reduced)
  - D4 selected points (1%, 5%) using the D4 commands above; toggle architecture/normalization/regularization variants

---
## D6 — Robustness, Efficiency & Final Summary
- Plan & Acceptance: `docs\D6_Experiment_Plan.md`, `docs\D6_Acceptance_Criteria.md`
- Commands:
```powershell
# D3 quick-set (robustness) – see D3 section for commands
# Efficiency tables (example placeholders):
python scripts\export_summary.py  # if used to aggregate metrics
# Plot reliability/curves (see D4 export)
```

## Notes
- Resume best practices: always include `--resume` in Python; batch脚本会自动跳过已存在的输出。
- 若 `label_ratio=1.0`，已自动保留测试集样本，避免空测试导致的报错。

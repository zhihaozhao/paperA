# Experiment TODO (EN/CN) — WiFi CSI HAR Thesis

Branch: `thesis/phd-dissertation-chapter`
Date: 2025-08-25

## Overall Targets / 实验总体目标
- EN: Validate physics-guided synthetic data and cross-domain generalization; deliver robust, calibrated models with strong Sim2Real label efficiency.
- CN: 验证物理指导的合成数据与跨域泛化；产出鲁棒、可校准且具备高 Sim2Real 标签效率的模型。

## Specs / 规范
- **Input shape**: [B, T=128, F=52] (default)
- **Models**: Enhanced, CNN, BiLSTM, Conformer-lite; upcoming: PINN-LSTM + multi-scale windows, PINN-MAMBA
- **Seeds**: 8 per config (D2); ≥3 for CDAE; 3–5 for STEA per label ratio
- **Metrics**: Top-1, macro/micro F1, ECE, NLL, Brier; per-class F1; CI via bootstrap
- **Significance**: paired t-tests, Cohen’s d; report mean±std and 95% CI
- **Calibration**: temperature scaling; reliability diagrams (before/after)
- **Reproducibility**: log configs/commands; export CSVs; version figures

---

## D2 — Synthetic Robustness Validation / 合成鲁棒性验证

### Objectives / 目标
- EN: Show controllable trends vs noise/overlap/fading; Enhanced wins consistently; low calibration error; statistically reliable.
- CN: 呈现噪声/重叠度/衰落等维度的可控趋势；Enhanced 保持领先；校准误差低；统计可靠。

### TODOs
- Define 540 config grid: noise_std, class_overlap, fading/channel params, difficulty {low, mid, high}
- Run {Enhanced, CNN, BiLSTM, Conformer-lite} × 540 × 8 seeds
- Save predictions and logits for calibration; export metrics CSV
- Plot: overlap→error, noise→ECE, fading→Top-1; reliability diagrams
- Significance tables (pairwise tests, effect sizes)
- Ablations: SE on/off; capacity-matched baselines

### Scripts
- Shell: `bash scripts/run_synth_batch.py` or `bash scripts/run_d2_validation.bat` (Windows)
- Analysis: `python scripts/generate_d2_analysis_report.py`
- Plots: `python scripts/plot_reliability.sh`

---

## CDAE — Cross-Domain Adaptation Evaluation / 跨域适应评估（LOSO/LORO）

### Objectives / 目标
- EN: Verify LOSO/LORO generalization; quantify stability, report CI and significance; analyze domain gap.
- CN: 验证 LOSO/LORO 泛化；量化稳定性并报告 CI 与显著性；分析域间差距。

### TODOs
- Build LOSO/LORO splits; verify strict isolation
- Train/evaluate all models; export per-domain metrics, CIs
- Statistical tests: paired t-tests; effect sizes; coeff. of variation
- Optional: feature-space distances for domain gap

### Scripts
- Train: `python src/train_cross_domain.py --protocol loso|loro --model <name> --seeds 3`
- Validate: `python scripts/validate_d3_acceptance.py`
- Plots: `python scripts/plot_d3_folds_box.py`

---

## STEA — Sim2Real Transfer Efficiency / Sim2Real 标签效率

### Objectives / 目标
- EN: Demonstrate label efficiency with synthetic pretraining; high performance at 10–20% labels; calibration maintained.
- CN: 证明合成预训练的标签效率；10–20% 标签下达成高性能；保持良好校准。

### TODOs
- Pretrain on synthetic; fine-tune with {10%, 20%, 50%, 100%} real labels
- Compare: from-scratch vs synthetic-pretrain; (optional) self-supervised baseline
- Export label-efficiency curves; ECE/NLL vs label ratio

### Scripts
- Run: `bash scripts/run_sim2real.sh` or `pwsh scripts/run_sim2real.ps1`
- Plot: `python scripts/plot_d4_label_efficiency.py`
- Validate: `python scripts/validate_d4_acceptance.py`

---

## Calibration & Reliability / 校准与可靠性

### TODOs
- Temperature scaling on validation; report ECE/NLL before/after
- Reliability diagrams per model/dataset; per-class calibration (optional)
- Export CSV summary; baseline comparisons

### Scripts
- `bash scripts/plot_reliability.sh`
- `python src/calibration.py` (if applicable in pipeline)

---

## Reproducibility & Integration / 可复现与集成

### TODOs
- Standardize figure paths via `\graphicspath{}`; output to `Thesis/figures`
- Save run configs/commands and seeds; export metrics to `results/*.csv`
- Update `Thesis/appendices/results_tables.tex` with final tables
- Cross-reference figures/tables in `chapter_experiments_en.tex`

### Scripts & Utilities
- `python scripts/create_results_summary.py`
- `python scripts/export_summary.py`
- `python scripts/generate_paper_figures.py`

---

## PINN Extensions (Planned) / 物理先验扩展（计划）

### PINN LSTM + Multi-Scale Windows
- Loss: CE + λ_phys * PINN residual (e.g., smoothness/temporal PDE surrogate)
- Windows: {32, 64, 128} with shared backbone or heads; late fusion
- TODOs: implement residual module; lambda sweep; ablation by window sets

### PINN MAMBA
- Replace temporal module with Mamba block; add PINN residual regularizer
- TODOs: implement Mamba variant; match params; compare to BiLSTM/attn

---

## Deliverables / 交付物
- CSVs: metrics per config/domain/label-ratio; calibration summaries
- Figures: trend plots, reliability, boxplots, label-efficiency curves
- Tables: main results, ablations, significance
- Updated LaTeX: figures/tables integrated; references updated

---

## Quick Command Reference / 快速命令
```bash
# D2 batch (example)
python scripts/run_synth_batch.py --models enhanced cnn bilstm conformer_lite --seeds 0 1 2 3 4 5 6 7

# CDAE (example)
python src/train_cross_domain.py --protocol loso --model enhanced --seeds 0 1 2

# STEA (example)
bash scripts/run_sim2real.sh

# Plots & reports
python scripts/generate_d2_analysis_report.py
bash scripts/plot_reliability.sh
python scripts/plot_d4_label_efficiency.py
```
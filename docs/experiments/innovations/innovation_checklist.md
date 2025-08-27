### Innovation Checklist (Zero/Few-shot, Enhanced Model, Trustworthy Sim2Real)

- **Physics-guided CSI synthesis (parameterized, domain-randomized)**
  - Benchmarks/baselines: SenseFi datasets for evaluation; complements data-augmentation only approaches.
  - Evidence: D6 robustness; STEA efficiency (82.1% F1 @20% labels).
  - Citations: Yang et al., Patterns 2023; Goldsmith 2005.

- **PINN-like Enhanced model (CNN + SE + Temporal Attention)**
  - Benchmarks/baselines: CNN, BiLSTM, Conformer-lite (capacity-aligned).
  - Evidence: CDAE LOSO/LORO 83.0±0.1% F1; stability CV<0.2%.
  - Citations: SE-Nets (CVPR 2018); attention for time-series/video (TEA, TimeSformer, TFT, Informer).

- **Trustworthy evaluation: calibration (ECE/Brier/NLL) and selective deployment**
  - Benchmarks/baselines: report alongside macro-F1; temperature scaling.
  - Evidence: improved NLL and ECE on D6; Sim2Real zero-shot ECE characterization.
  - Citations: Guo et al. (ICML 2017).

- **Label-efficiency protocol (STEA) for Sim2Real**
  - Benchmarks/baselines: zero-shot, linear probe, fine-tune; label ratios 1–100%.
  - Evidence: 82.1% F1 @20% labels vs 83.3% full; diminishing returns beyond 20%.
  - Citations: SenseFi context; Sim2Real transfer (robotics dynamics randomization).

- **Cross-domain protocol (CDAE) for LOSO/LORO consistency**
  - Benchmarks/baselines: LOSO (subject), LORO (environment).
  - Evidence: Enhanced identical LOSO/LORO means; lower variance than baselines.
  - Citations: SenseFi (benchmarks), domain generalization in WiFi (FewSense, AirFi).

- **Ablation and interpretability**
  - Benchmarks/baselines: D2 nuisance sweeps (class overlap, env burst), D5/D6 trends.
  - Evidence: Enhanced robust across stressors; attribution aligns with propagation cues.
  - Citations: Grad-CAM, Integrated Gradients.
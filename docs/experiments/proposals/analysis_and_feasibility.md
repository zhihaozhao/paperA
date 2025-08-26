Analysis and Feasibility of Candidate Baselines and Directions (v0)

Evaluation axes
- Reproducibility (code available, datasets public, scripts stable)
- Metric comparability (macro F1, CDAE LOSO/LORO, STEA label ratios)
- Parameter observability (Params, FLOPs/throughput measurable from code)
- Alignment with our pipeline (CSI HAR tasks, SenseFi datasets, protocols)

SenseFi (Patterns 2023)
- Code: Available; multiple models and datasets unified.
- Metrics: Comparable; supports standard splits.
- Params/FLOPs: Measurable from model code.
- Feasibility: High. Direct baseline for our CDAE/STEA framing.

FewSense (arXiv 2022)
- Code: Likely available; few-shot domain generalization.
- Metrics: Need mapping to LOSO/LORO; may require custom loaders.
- Params: Measurable if PyTorch; throughput moderate.
- Feasibility: Medium. Useful for few-shot STEA variants.

AirFi (arXiv 2022)
- Code: Meta-learning framework likely heavier.
- Metrics: Domain generalization; map to our datasets.
- Params: Measurable; training complexity higher.
- Feasibility: Medium. Adds DG perspective.

ReWiS (arXiv 2022)
- Code: TBD; few-shot multi-antenna.
- Metrics: Match to SenseFi datasets uncertain.
- Feasibility: Medium. Worth partial replication if code exists.

CLNet (arXiv 2021)
- Code: Telecom CSI feedback; adaptation needed for HAR.
- Feasibility: Medium. Use for parameter efficiency studies.

DeepCSI (arXiv 2022)
- Code: Device ID; task mismatch.
- Feasibility: Medium-low. Limited for HAR directly.

Most feasible near-term directions
1) SenseFi-aligned CDAE/STEA sweeps with our Enhanced, CNN, BiLSTM, Conformer-lite.
2) FewSense-style few-shot STEA (1–5–20% labels) to stress transfer efficiency.
3) Parameter-efficiency sweep: multi-scale LSTM + lite attention vs CNN/LSTM baselines; measure Params/throughput at equal accuracy.

Risk notes
- Dataset licensing and preprocessing variance.
- Inconsistent protocol definitions; document assumptions per REPRO_PLAN.

Next steps
- Pin official repos/links; fill each REPRO_PLAN with concrete commands and dataset maps.
- Implement a unified metrics export to results/metrics.json across baselines.

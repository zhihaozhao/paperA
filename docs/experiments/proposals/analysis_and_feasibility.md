Analysis and Feasibility of Follow-up Directions

Scope
- Review of referenced baselines with emphasis on reproducibility and measurable parameters (Params, FLOPs/throughput, input size, attention type).
- Identify concrete, high-yield directions and rate feasibility.

Candidates (from refs)
- SenseFi benchmark (Patterns 2023) — official code available; multiple datasets.
- FewSense (arXiv 2022) — repo likely; few-shot protocols across domains.
- AirFi (arXiv 2022) — domain generalization with meta-learning.
- ReWiS (arXiv 2022) — few-shot multi-antenna; code TBD.
- CLNet (arXiv 2021) — lightweight CSI feedback; likely reproducible.
- DeepCSI (arXiv 2022) — device fingerprinting; code TBD.

Feasibility Ratings (initial)
- SenseFi: High — code + datasets; directly comparable.
- FewSense: Medium — domain configs may need adaptation.
- AirFi: Medium — meta-learning infra may be heavier.
- ReWiS: Medium — few-shot infra; data availability uncertain.
- CLNet: Medium — telecom focus; adapt to HAR.
- DeepCSI: Medium — task mismatch; adapt metrics.

Next actions
- Pin official repos/DOIs; verify licenses.
- For each, measure Params/throughput from code rather than paper if NR.
- Design comparative sweeps aligned to our CDAE/STEA protocols.

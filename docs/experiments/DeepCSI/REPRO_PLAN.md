DeepCSI — Reproducibility Plan

1. Repository and Commit
- Official paper key: see paper/refs.bib
- Official repo: [TBD after verification]
- Mirror/third-party: [TBD]

2. Environment
- Use Python 3.10+ and pinned requirements.
- Prefer conda or venv; see setup.sh.

3. Datasets and Mapping
- Benchmarks referenced in paper/main.tex (SenseFi datasets: UT-HAR, NTU-Fi-HAR, NTU-Fi-HumanID, Widar).
- Map dataset names and expected directory structures once repo is confirmed.

4. Training/Eval Commands (to adapt once repo is linked)
- Example: bash scripts/train.sh --config configs/baseline.yaml --output results/
- Seed sweep: 5 seeds; collect macro F1, ECE, NLL.

5. Expected Results (from paper)
- Metrics source: paper/refs.bib (key for this baseline).
- Note any protocol match/mismatch (LOSO/LORO/ID).

6. Repro Checklist
- [ ] Repo cloned and commit pinned
- [ ] Environment created and frozen
- [ ] Data preprocessed and cached
- [ ] Scripts run non-interactively
- [ ] Metrics captured to results/metrics.json

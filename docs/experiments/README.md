Experiments Workspace

This directory organizes external baselines, reproducibility notes, parameter/metric analysis, and our planned experiment designs. Each paper or experiment has its own subdirectory.

Subdirectories
- SenseFi/ — Benchmark library and repro notes
- FewSense/ — Cross-domain few-shot learning paper analysis and repro
- AirFi/ — Domain generalization/meta-learning paper analysis and repro
- ReWiS/ — Few-shot multi-antenna CSI learning repro notes
- DeepCSI/ — Device fingerprinting via deep feedback learning
- CLNet/ — Lightweight CSI feedback model
- EfficientFi/ — CSI compression for lightweight sensing
- GaitFi/ — Multimodal WiFi+vision identification
- proposals/ — Literature-driven follow-ups and feasibility assessment

Conventions
- Keep each subfolder self-contained: README.md, references.md, run.sh, requirements.txt, notes.md, results/.
- Do not fabricate metrics/parameters. If not reported, mark as NR; prefer code-based measurement in future updates.
- Use non-interactive scripts; pin dependencies where possible.

# WiFi HAR Experiments Documentation

## 📊 Project Overview

This repository contains comprehensive experiments for WiFi-based Human Activity Recognition using Channel State Information (CSI), featuring physics-informed learning and state-space models.

## 🚀 Quick Start

```bash
# 1. Setup environment
bash docs/experiments/environment_setup_claude4.1.sh
source /workspace/.venv_exp/bin/activate

# 2. Preprocess data
python docs/experiments/scripts/preprocess_data_claude4.1.py \
    --input_dir /path/to/raw/data \
    --output_dir /workspace/data/processed

# 3. Run experiments
bash docs/experiments/scripts/run_experiments_claude4.1.sh
```

## 📁 Repository Structure

```
docs/experiments/
├── innovations/                 # Innovation checklist and mapping
│   └── innovation_checklist_claude4.1.md
├── paper_drafts/               # Extended paper drafts (40-60k chars each)
│   ├── exp1_extended_claude4.1.tex  # Physics-informed LSTM (74k chars)
│   └── exp2_extended_claude4.1.tex  # Mamba SSM (77k chars)
├── baselines/                  # Baseline reproduction plans
│   ├── SenseFi/REPRO_PLAN_claude4.1.md
│   ├── FewSense/REPRO_PLAN_claude4.1.md
│   ├── AirFi/REPRO_PLAN_claude4.1.md
│   ├── ReWiS/REPRO_PLAN_claude4.1.md
│   ├── CLNet/REPRO_PLAN_claude4.1.md
│   ├── DeepCSI/REPRO_PLAN_claude4.1.md
│   ├── EfficientFi/REPRO_PLAN_claude4.1.md
│   └── GaitFi/REPRO_PLAN_claude4.1.md
├── exp1_multiscale_lstm_lite_attn_PINN/  # Experiment 1
├── exp2_mamba_replacement/                # Experiment 2
├── scripts/                    # Automation scripts
│   ├── run_experiments_claude4.1.sh
│   └── preprocess_data_claude4.1.py
├── bibliography/              # References management
│   ├── refs_claude4.1.json
│   └── refs_claude4.1.csv
├── results_template_claude4.1.md     # Results collection template
├── roadmap_claude4.1.md             # Research roadmap
└── commit_analysis_claude4.1.md     # Git history analysis
```

## 🧪 Experiments

### Exp1: Physics-Informed Multi-Scale LSTM
- **Innovation**: Incorporates electromagnetic propagation constraints
- **Architecture**: Multi-scale LSTM + Lightweight Attention + PINN
- **Target**: 85% F1 score with physics interpretability

### Exp2: Mamba State-Space Models
- **Innovation**: First application of Mamba SSM to WiFi sensing
- **Architecture**: Selective state-space with linear complexity
- **Target**: 80% F1 with 3x throughput improvement

## 📊 Evaluation Protocols

### CDAE (Cross-Domain Adaptation Evaluation)
- **LOSO**: Leave-One-Subject-Out
- **LORO**: Leave-One-Room-Out

### STEA (Sample-Efficient Transfer Adaptation)
- 1%, 5%, 20% labeled data scenarios
- Pre-training with physics constraints

## 🎯 Performance Targets

| Model | F1 Score | Params | Throughput | Edge Latency |
|-------|----------|--------|------------|--------------|
| Enhanced (Baseline) | 83.0% | 1.2M | 1000 sps | 105ms |
| Exp1 (Physics-LSTM) | 85.0% | 1.0M | 1200 sps | 85ms |
| Exp2 (Mamba) | 80.0% | 1.5M | 2400 sps | 42ms |

## 📈 Current Progress

- ✅ Documentation: 98% complete
- ⏳ Implementation: 10% (scaffolds ready)
- ⏳ Evaluation: 5% (protocols designed)
- ❌ Deployment: 0% (pending)

## 🛠️ Tools & Dependencies

- **Framework**: PyTorch 1.13+
- **State-Space**: Mamba SSM, Triton
- **Physics**: torch-geometric
- **Deployment**: ONNX, TensorRT
- **Hardware**: CUDA 11.8+

## 📚 Key References

1. SenseFi (Patterns 2023) - Benchmark framework
2. Mamba (NeurIPS 2023) - State-space models
3. Physics-Informed NNs (JCP 2019) - Physics constraints

## 🚦 Next Steps

1. **Immediate**: Complete Exp1/Exp2 implementations
2. **Short-term**: Run baseline comparisons
3. **Medium-term**: Cross-dataset validation
4. **Long-term**: Paper submission preparation

## 📞 Contact

For questions about this research, please open an issue or contact [email].

## 📄 License

This project is for research purposes. See LICENSE for details.

---
*Last updated: December 2024*
*All files use claude4.1 suffix for version tracking*
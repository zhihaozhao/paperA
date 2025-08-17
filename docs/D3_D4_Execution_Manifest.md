# D3 & D4 Experiments Execution Manifest

## Overview
Comprehensive execution guide for D3 (Cross-Domain) and D4 (Sim2Real) experiments following successful D2 validation with 540 synthetic configurations.

**Target**: Top-tier journals (IoTJ/TMC/IMWUT) with trustworthy evaluation framework

---

## 📁 File Structure & Components

### Core Implementation Files
```
src/
├── data_real.py          ✅ Real data loading with LOSO/LORO splits
├── sim2real.py           ✅ Sim2Real transfer learning framework  
├── train_cross_domain.py ✅ Cross-domain training script
└── run_sim2real_single.py ✅ Single Sim2Real experiment runner
```

### Execution Scripts
```
scripts/
├── run_d3_loso.sh        ✅ D3 LOSO experiments (4 models × 5 seeds)
├── run_d3_loro.sh        ✅ D3 LORO experiments (4 models × 5 seeds)  
├── run_d4_sim2real.sh    ✅ D4 Sim2Real sweep (560 configurations)
├── validate_d3_acceptance.py ✅ D3 validation & reporting
└── validate_d4_acceptance.py ✅ D4 validation & reporting
```

### Documentation
```
docs/
├── D3_D4_Experiment_Plans.md    ✅ Detailed experimental design
└── D3_D4_Execution_Manifest.md  ✅ This execution guide
```

---

## 🚀 Quick Start Commands

### Prerequisites Check
```bash
# 1. Verify benchmark dataset
ls -la benchmarks/WiFi-CSI-Sensing-Benchmark-main/

# 2. Check D2 pre-trained models (for D4)
ls -la checkpoints/d2/

# 3. Verify environment
python -c "import torch, numpy, sklearn; print('✅ Dependencies OK')"
```

### D3 Execution (Cross-Domain)
```bash
# 1. Run LOSO experiments (~20 configs, 4 models × 5 seeds)
bash scripts/run_d3_loso.sh

# 2. Run LORO experiments (~20 configs, 4 models × 5 seeds)  
bash scripts/run_d3_loro.sh

# 3. Validate results
python scripts/validate_d3_acceptance.py --protocol loso
python scripts/validate_d3_acceptance.py --protocol loro
```

### D4 Execution (Sim2Real)
```bash
# 1. Run Sim2Real label efficiency sweep (~560 configs)
bash scripts/run_d4_sim2real.sh

# 2. Validate results
python scripts/validate_d4_acceptance.py

# 3. Generate visualizations (optional)
python scripts/plot_sim2real_curves.py
```

---

## 📊 Experimental Configurations

### D3: Cross-Domain Generalization
| Component | Configuration |
|-----------|---------------|
| **Models** | enhanced, cnn, bilstm, conformer_lite |
| **Protocols** | LOSO (Leave-One-Subject-Out), LORO (Leave-One-Room-Out) |
| **Seeds** | 0, 1, 2, 3, 4 |
| **Dataset** | WiFi-CSI-Sensing-Benchmark (real data) |
| **Total Configs** | ~200-400 (depends on benchmark subjects/rooms) |

### D4: Sim2Real Label Efficiency  
| Component | Configuration |
|-----------|---------------|
| **Models** | enhanced, cnn, bilstm, conformer_lite |
| **Label Ratios** | 1%, 5%, 10%, 15%, 20%, 50%, 100% |
| **Transfer Methods** | zero_shot, linear_probe, fine_tune, temp_scale |
| **Seeds** | 0, 1, 2, 3, 4 |
| **Total Configs** | 560 (4 models × 7 ratios × 4 methods × 5 seeds) |

---

## 🎯 Success Criteria

### D3 Acceptance Criteria
- ✅ **Performance**: Falling F1 ≥ 0.75, Macro F1 ≥ 0.80 (cross-domain avg)
- ✅ **Calibration**: ECE ≤ 0.15 after temperature scaling
- ✅ **Robustness**: Enhanced model outperforms baselines by ≥5% Falling F1
- ✅ **Coverage**: ≥90% of LOSO/LORO folds complete successfully
- ✅ **Statistical**: Bootstrap 95% CI for all metrics

### D4 Acceptance Criteria  
- ✅ **Zero-shot**: Falling F1 ≥ 0.60, Macro F1 ≥ 0.70
- ✅ **Label Efficiency**: 10-20% real labels achieve ≥90-95% full performance
- ✅ **Transfer Gain**: Fine-tuning improves over zero-shot by ≥15% Falling F1
- ✅ **Calibration**: Sim2Real ECE gap ≤ 0.10 after temperature scaling
- ✅ **Coverage**: All label efficiency points with ≥3 successful seeds

---

## ⚡ Performance Optimization

### Parallel Execution (Recommended)
```bash
# D3: Run LOSO and LORO in parallel
bash scripts/run_d3_loso.sh &
bash scripts/run_d3_loro.sh &
wait

# D4: Use GNU parallel for massive speedup
seq 0 4 | parallel -j 4 "
  for model in enhanced cnn bilstm conformer_lite; do
    for ratio in 0.01 0.05 0.10 0.15 0.20 0.50 1.00; do
      for method in zero_shot linear_probe fine_tune temp_scale; do
        python -m src.run_sim2real_single --model \$model --transfer_method \$method --label_ratio \$ratio --seed {} --benchmark_path benchmarks/WiFi-CSI-Sensing-Benchmark-main --d2_models_dir checkpoints/d2 --output_file results/d4/sim2real/sim2real_\${model}_\${method}_ratio\${ratio}_seed{}.json
      done
    done
  done
"
```

### Resource Requirements
| Experiment | Time Estimate | GPU Memory | Storage |
|------------|---------------|------------|---------|
| **D3 LOSO** | 4-8 hours | 4-8 GB | 1-2 GB |
| **D3 LORO** | 3-6 hours | 4-8 GB | 1-2 GB |
| **D4 Sim2Real** | 8-16 hours | 4-8 GB | 3-5 GB |

---

## 🔍 Results Organization

### D3 Output Structure
```
results/d3/
├── loso/
│   ├── loso_enhanced_seed0.json
│   ├── loso_enhanced_seed1.json
│   ├── ...
│   └── d3_loso_validation_report.md
└── loro/
    ├── loro_enhanced_seed0.json
    ├── loro_enhanced_seed1.json
    ├── ...
    └── d3_loro_validation_report.md
```

### D4 Output Structure
```
results/d4/
└── sim2real/
    ├── sim2real_enhanced_zero_shot_ratio0.01_seed0.json
    ├── sim2real_enhanced_linear_probe_ratio0.10_seed1.json
    ├── ...
    ├── d4_sim2real_validation_report.md
    └── label_efficiency_plots/
```

---

## 🧪 Validation & Quality Control

### Automated Validation
```bash
# D3 Validation
python scripts/validate_d3_acceptance.py --protocol loso
python scripts/validate_d3_acceptance.py --protocol loro

# D4 Validation  
python scripts/validate_d4_acceptance.py

# Combined validation
python scripts/validate_all_experiments.py  # Create if needed
```

### Manual Quality Checks
1. **Data Integrity**: Verify benchmark dataset loading
2. **Model Loading**: Confirm D2 pre-trained models available
3. **Metrics Consistency**: Check metric calculations across experiments
4. **Statistical Significance**: Verify bootstrap confidence intervals

---

## 📈 Expected Outcomes & Deliverables

### D3 Deliverables
- [ ] LOSO cross-validation results (per-subject analysis)
- [ ] LORO cross-validation results (per-environment analysis)  
- [ ] Cross-domain robustness comparison table
- [ ] Statistical significance tests
- [ ] Publication-ready cross-domain evaluation

### D4 Deliverables
- [ ] Zero-shot Sim2Real baseline performance
- [ ] Label efficiency curves (1% → 100% real data)
- [ ] Transfer learning method comparison
- [ ] Optimal adaptation strategy identification
- [ ] Practical deployment guidelines

### Combined Impact
- [ ] Complete trustworthy ML evaluation framework
- [ ] Strong experimental evidence for top-tier journal submission
- [ ] Reproducible cross-domain and Sim2Real protocols
- [ ] Open-source benchmark for WiFi CSI HAR community

---

## 🐛 Troubleshooting Guide

### Common Issues & Solutions

#### D3 Issues
```bash
# Issue: Benchmark dataset not found
# Solution: Check dataset path and structure
python -c "from src.data_real import BenchmarkCSIDataset; 
           b = BenchmarkCSIDataset(); 
           X,y,s,r,m = b.load_wifi_csi_benchmark(); 
           print(f'Loaded: {len(y)} samples, {m[\"n_subjects\"]} subjects')"

# Issue: LOSO/LORO splits fail
# Solution: Verify subject/room metadata
python -c "import numpy as np; 
           from src.data_real import BenchmarkCSIDataset;
           b = BenchmarkCSIDataset();
           _,_,subjects,rooms,_ = b.load_wifi_csi_benchmark();
           print(f'Subjects: {np.unique(subjects)}, Rooms: {np.unique(rooms)}')"
```

#### D4 Issues  
```bash
# Issue: Pre-trained models not found
# Solution: Check D2 model checkpoints
find checkpoints/ -name "*.pth" | head -5

# Issue: Transfer learning fails
# Solution: Test individual components
python -c "from src.sim2real import Sim2RealEvaluator; 
           eval = Sim2RealEvaluator(); 
           print('✅ Sim2Real evaluator initialized')"
```

#### General Issues
```bash
# Memory issues: Reduce batch size
export BATCH_SIZE=32

# Slow execution: Use subset for testing
export TEST_MODE=1  # Enable if scripts support test mode

# Dependency issues: Reinstall environment
pip install -r requirements.txt
```

---

## 📋 Execution Checklist

### Pre-Execution Setup
- [ ] Verify benchmark dataset in `benchmarks/WiFi-CSI-Sensing-Benchmark-main/`
- [ ] Confirm D2 pre-trained models in `checkpoints/d2/`
- [ ] Check Python environment and dependencies
- [ ] Create results directories: `mkdir -p results/{d3,d4}`

### D3 Execution
- [ ] Run LOSO experiments: `bash scripts/run_d3_loso.sh`
- [ ] Run LORO experiments: `bash scripts/run_d3_loro.sh`  
- [ ] Validate LOSO: `python scripts/validate_d3_acceptance.py --protocol loso`
- [ ] Validate LORO: `python scripts/validate_d3_acceptance.py --protocol loro`
- [ ] Generate D3 summary reports

### D4 Execution
- [ ] Run Sim2Real sweep: `bash scripts/run_d4_sim2real.sh`
- [ ] Validate results: `python scripts/validate_d4_acceptance.py`
- [ ] Generate label efficiency plots
- [ ] Create D4 summary reports

### Post-Execution
- [ ] Merge results into unified analysis
- [ ] Update paper with D3/D4 findings
- [ ] Prepare supplementary materials
- [ ] Create publication-ready figures and tables

---

## 🏆 Publication Integration

### Paper Sections Enhanced
1. **Method**: Cross-domain evaluation protocols (D3)
2. **Experiments**: Sim2Real analysis with label efficiency (D4)
3. **Results**: Combined D1+D2+D3+D4 comprehensive evaluation
4. **Discussion**: Practical deployment insights from D4

### Key Figures to Generate
- [ ] D3: Cross-domain performance comparison (LOSO vs LORO)
- [ ] D4: Label efficiency curves (all models, all methods)
- [ ] Combined: Comprehensive evaluation framework overview
- [ ] Supplementary: Per-subject/room detailed analysis

---

## 🎯 Success Metrics Summary

| Experiment | Key Metric | Target | Impact |
|------------|------------|--------|---------|
| **D3 LOSO** | Cross-subject Falling F1 | ≥0.75 | Subject generalization |
| **D3 LORO** | Cross-room Falling F1 | ≥0.75 | Environment robustness |
| **D4 Zero-shot** | Sim2Real Falling F1 | ≥0.60 | Transfer baseline |
| **D4 Efficiency** | 10% labels → 90% performance | ≥0.90 | Practical deployment |

**Combined Goal**: Establish complete trustworthy evaluation framework for WiFi CSI HAR with cross-domain and Sim2Real validation protocols suitable for top-tier journal publication.

---

## 🔄 Version Control & Results Management

### Recommended Git Workflow
```bash
# 1. Create experimental branches
git checkout -b exp/d3-cross-domain
git checkout -b exp/d4-sim2real

# 2. Run experiments and commit results
git add results/d3/ docs/D3_D4_*
git commit -m "Add D3 cross-domain experiment results"

# 3. Create tags for major milestones
git tag d3-loso-complete
git tag d3-loro-complete  
git tag d4-sim2real-complete

# 4. Merge to main after validation
git checkout main
git merge exp/d3-cross-domain
git merge exp/d4-sim2real
```

### Results Backup Strategy
```bash
# Create compressed archives for long-term storage
tar -czf d3_results_$(date +%Y%m%d).tar.gz results/d3/
tar -czf d4_results_$(date +%Y%m%d).tar.gz results/d4/

# Upload to cloud storage or backup location
# rsync -av results/ backup_server:/path/to/backup/
```

---

## 📞 Support & Next Steps

### If Issues Arise
1. **Check Prerequisites**: Dataset availability, model checkpoints, dependencies
2. **Run Subset First**: Test with single model/seed before full sweep
3. **Examine Logs**: Check detailed error messages and validation reports
4. **Contact Support**: Provide specific error messages and context

### After Successful Completion
1. **Paper Integration**: Update manuscript with D3/D4 findings
2. **Supplementary Material**: Prepare detailed experimental appendix
3. **Code Release**: Prepare reproducible research package
4. **Journal Submission**: Target IoTJ/TMC with complete evaluation framework

---

**Status**: ✅ Implementation Complete - Ready for Execution
**Next Action**: Run `bash scripts/run_d3_loso.sh` to begin D3 experiments
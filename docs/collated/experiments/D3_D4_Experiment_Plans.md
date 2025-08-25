# D3 & D4 Experiment Plans - Cross-Domain & Sim2Real Evaluation

## Background
After successful D2 validation with 540 synthetic configurations, we proceed to real-world evaluation phases targeting top-tier journals (IoTJ/TMC/IMWUT) with focus on:
- Cross-domain generalization (LOSO/LORO)  
- Sim2Real label efficiency (10-20% labels → ≥90-95% performance)
- Trustworthy evaluation with calibration metrics

---

## D3 Experiments: Cross-Domain Generalization

### Objective
Validate model robustness across subjects and environments using real WiFi CSI benchmark data with Leave-One-Subject-Out (LOSO) and Leave-One-Room-Out (LORO) protocols.

### D3.1 Configuration
**Dataset**: benchmarks/WiFi-CSI-Sensing-Benchmark-main
**Models**: enhanced, cnn, bilstm, conformer_lite (capacity-matched ±10%)
**Protocol**: LOSO (Leave-One-Subject-Out) & LORO (Leave-One-Room-Out)
**Seeds**: 0, 1, 2, 3, 4 (consistency with D2)
**Metrics**: macro_f1, falling_f1, ece, auprc_falling, mutual_misclass
**CLI Params (train_cross_domain.py)**:
- `--protocol {loso,loro,sim2real}`: 协议选择
- `--files_per_activity INT`: 每类读取的文件数（平衡/速度）
- `--class_weight {none,inv_freq}`: 损失加权方案（类别不平衡）
- `--loso_all_folds`: 跑完整 LOSO 全折并聚合
- 其他：`--epochs, --batch_size, --lr, --benchmark_path`

### D3/D4 验收与导出命令

#### D3 验收
```powershell
python scripts\validate_d3_acceptance.py --root results --out_dir results\metrics --save_report docs\d3_acceptance_report.md
```

#### D2 验收（参考）
```powershell
python scripts\validate_d2_acceptance.py --results-dir results_gpu\d2 --save-report docs\d2_acceptance_report.md
```

#### D4 运行（Windows 总控）
```cmd
scripts\run_d3_d4_windows.bat
```

#### D4 验收与曲线导出
```powershell
python scripts\validate_d4_acceptance.py --root results\d4 --out_dir results\metrics --save_report docs\d4_acceptance_report.md
python scripts\export_d4_curves.py --root results\d4 --out_dir results\metrics
```

### D3.2 Experimental Grid
```
Total Configurations: 4 models × 2 protocols × N_subjects × 5 seeds
Expected: ~200-400 experiments (depending on benchmark subjects/rooms)

Protocol 1: LOSO Cross-Validation
- For each subject i in {0, 1, ..., N_subjects-1}:
  - Train on all subjects except i
  - Test on subject i
  - Report per-subject results + aggregate stats

Protocol 2: LORO Cross-Validation  
- For each room j in {0, 1, ..., N_rooms-1}:
  - Train on all rooms except j
  - Test on room j
  - Report per-room results + aggregate stats
```

### D3.3 Success Criteria
- **Performance**: Falling F1 ≥ 0.75 (cross-domain avg), Macro F1 ≥ 0.80
- **Calibration**: ECE ≤ 0.15 after temperature scaling
- **Robustness**: Enhanced model outperforms baselines by ≥5% Falling F1
- **Coverage**: ≥90% of LOSO/LORO folds complete successfully
- **Statistical**: Bootstrap 95% CI for all main metrics

---

## D4 Experiments: Sim2Real Label Efficiency  

### Objective
Evaluate synthetic-to-real transfer learning with minimal real data requirements, targeting 10-20% labeled real data to achieve ≥90-95% of full-supervision performance.

### D4.1 Configuration
**Source**: D2 pre-trained models (enhanced, cnn, bilstm, conformer_lite)
**Target**: Real WiFi CSI benchmark data
**Protocol**: Progressive label efficiency (1%, 5%, 10%, 15%, 20%, 50%, 100%)
**Seeds**: 0, 1, 2, 3, 4
**Transfer Methods**: Zero-shot, Fine-tuning, Linear Probing, Temperature Scaling

### D4.2 Experimental Design
```
Label Efficiency Sweep:
- real_label_ratios: [0.01, 0.05, 0.10, 0.15, 0.20, 0.50, 1.00]
- transfer_methods: ["zero_shot", "linear_probe", "fine_tune", "temp_scale"]
- Total: 4 models × 7 ratios × 4 methods × 5 seeds = 560 configurations

Transfer Learning Protocols:
1. Zero-shot: Direct evaluation of synthetic-trained model on real data
2. Linear Probe: Freeze backbone, train only classifier on real data
3. Fine-tune: End-to-end fine-tuning with low learning rate
4. Temperature Scaling: Calibration-only adaptation on small real subset
```

### D4.3 Success Criteria  
- **Label Efficiency**: 10-20% real labels achieve ≥90% of full-supervision performance
- **Zero-shot Baseline**: Falling F1 ≥ 0.60, Macro F1 ≥ 0.70  
- **Transfer Gain**: Fine-tuning improves over zero-shot by ≥15% Falling F1
- **Calibration**: Sim2Real ECE gap ≤ 0.10 after temperature scaling
- **Coverage**: All label efficiency points with ≥3 successful seeds

---

## D3/D4 验收命令

### D3 验收
```powershell
python scripts\validate_d3_acceptance.py --root results --out_dir results\metrics --save_report docs\d3_acceptance_report.md
```

### D2 验收（参考）
```powershell
python scripts\validate_d2_acceptance.py --results-dir results_gpu\d2 --save-report docs\d2_acceptance_report.md
```

### D4 运行（Windows 总控）
```cmd
scripts\run_d3_d4_windows.bat
```

## Implementation Requirements

### Code Modifications Needed

#### 1. Real Data Integration (`src/data_real.py`)
```python
# Extend current minimal implementation to:
class BenchmarkCSIDataset:
    def load_wifi_csi_benchmark(self, path="benchmarks/WiFi-CSI-Sensing-Benchmark-main"):
        # Load benchmark dataset with subject/room metadata
        # Return X, y, subjects, rooms, metadata
        
    def create_loso_splits(self, subjects):
        # Generate LOSO cross-validation splits
        
    def create_loro_splits(self, rooms):  
        # Generate LORO cross-validation splits
```

#### 2. Sim2Real Framework (`src/sim2real.py`)
```python
class Sim2RealEvaluator:
    def zero_shot_evaluation(self, synth_model, real_loader):
        # Direct evaluation without adaptation
        
    def linear_probe_adaptation(self, synth_model, real_loader, label_ratio):
        # Freeze backbone, train classifier only
        
    def fine_tune_adaptation(self, synth_model, real_loader, label_ratio):
        # End-to-end fine-tuning with low LR
        
    def temperature_calibration(self, model, real_val_loader):
        # Calibration-only adaptation
```

#### 3. Cross-Domain Training Script (`src/train_cross_domain.py`)
```python
# New script for LOSO/LORO experiments
def run_loso_experiment(model_name, benchmark_path, seed):
    # Main LOSO experimental loop
    
def run_loro_experiment(model_name, benchmark_path, seed):
    # Main LORO experimental loop
```

### Scripts to Create

#### D3 Scripts
1. `scripts/run_d3_loso.sh` - LOSO cross-validation runner
2. `scripts/run_d3_loro.sh` - LORO cross-validation runner  
3. `scripts/validate_d3_acceptance.py` - D3 results validation
4. `scripts/prepare_benchmark_data.py` - Benchmark dataset preprocessing

#### D4 Scripts
1. `scripts/run_d4_sim2real.sh` - Sim2Real label efficiency runner
2. `scripts/validate_d4_acceptance.py` - D4 results validation
3. `scripts/plot_sim2real_curves.py` - Label efficiency visualization
4. `scripts/export_d4_summary.py` - D4 results summarization

---

## Execution Timeline

### Phase 1: Implementation (Days 1-3)
- [ ] Implement real data loading with LOSO/LORO splits
- [ ] Create Sim2Real evaluation framework
- [ ] Develop validation scripts
- [ ] Test on subset to verify implementation

### Phase 2: D3 Execution (Days 4-6)  
- [ ] Run LOSO experiments (~200 configs)
- [ ] Run LORO experiments (~200 configs)
- [ ] Validate D3 acceptance criteria
- [ ] Generate D3 summary reports

### Phase 3: D4 Execution (Days 7-10)
- [ ] Run Sim2Real label efficiency sweep (~560 configs)
- [ ] Validate D4 acceptance criteria  
- [ ] Generate label efficiency plots and analysis
- [ ] Create publication-ready D4 summary

### Phase 4: Integration (Days 11-12)
- [ ] Merge results into unified analysis
- [ ] Update paper with D3/D4 findings
- [ ] Prepare supplementary materials
- [ ] Create final experimental manifest

---

## Expected Outcomes

### D3 Deliverables
- LOSO/LORO performance benchmarks for 4 models
- Cross-domain robustness analysis 
- Subject/environment-specific performance insights
- Statistical significance tests across domains

### D4 Deliverables  
- Sim2Real transfer learning baseline (zero-shot)
- Label efficiency curves (1% → 100% real data)
- Optimal transfer method identification
- Practical deployment guidelines (minimal labeling requirements)

### Combined Impact
- Complete trustworthy evaluation framework
- Strong baselines for top-tier journal submission
- Reproducible cross-domain and Sim2Real protocols
- Publication-ready experimental evidence
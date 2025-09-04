# 📊 Submission Strategy - Paper 2: PASE-Net Architecture

## Paper Information
- **Title**: Physics-Informed PASE-Net Architecture for WiFi CSI Human Activity Recognition
- **Focus**: Novel architecture with attention mechanisms, calibration, interpretability
- **Key Contribution**: 83% consistent LOSO/LORO performance, 99% ECE reduction

## Target Journals (Priority Order)

### Tier 1: IEEE Transactions on Mobile Computing (TMC) ✅ READY
- **Impact Factor**: 7.9
- **Review Time**: 6-8 months
- **Acceptance Rate**: ~25%
- **Why Good Fit**:
  - Architecture innovation valued
  - Strong experimental validation
  - Cross-domain expertise
- **Status**: **READY FOR SUBMISSION**
  - Manuscript prepared
  - Real data integrated
  - Tag: TMC_v1.1

### Tier 2: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
- **Impact Factor**: 23.6
- **Review Time**: 8-12 months
- **Acceptance Rate**: ~15%
- **Why Good Fit**:
  - Novel architecture contribution
  - Strong theoretical foundation
  - Attention mechanism innovation
- **Requirements**:
  - Exceptional novelty required
  - Extended theoretical analysis
  - Comparison with 10+ baselines

### Tier 3: IEEE Transactions on Neural Networks and Learning Systems (TNNLS)
- **Impact Factor**: 10.4
- **Review Time**: 6-9 months
- **Acceptance Rate**: ~20%
- **Why Good Fit**:
  - Deep learning focus
  - Attention mechanisms
  - Calibration methods
- **Requirements**:
  - Neural architecture focus
  - Ablation studies
  - Convergence analysis

## Key Selling Points

### Unique Achievements
1. **Perfect Consistency**: 83.0% on both LOSO and LORO (unprecedented!)
2. **Best Calibration**: ECE 0.094→0.001 (99% reduction)
3. **Fall Detection**: >99% on specific types
4. **Complete Package**: Architecture + Calibration + Interpretability

### For TMC
1. **Mobile Sensing**: WiFi CSI for HAR
2. **Novel Architecture**: SE + Temporal attention
3. **Real Validation**: WiFi-CSI-Sensing-Benchmark
4. **Practical Impact**: Deployment ready

### For TPAMI
1. **Theoretical Contribution**: Physics-informed design
2. **Novel Attention**: Dual attention mechanism
3. **Interpretability**: Attribution analysis
4. **Generalization**: Cross-domain consistency

### For TNNLS
1. **Architecture Innovation**: PASE-Net design
2. **Learning Dynamics**: Attention evolution
3. **Calibration Theory**: Temperature scaling analysis
4. **Ablation Studies**: Component analysis

## Submission Timeline

### Immediate Action (TMC)
1. **Week 1**: Final review and compile PDF
2. **Week 1**: Submit to TMC
3. **Month 2-3**: Prepare for reviews
4. **Month 6**: Expected decision

### Backup Plan
1. **If TMC Reject**: Revise for TNNLS
2. **If Minor Issues**: Consider IoTJ
3. **If Major Issues**: Sensors or IEEE Access

## Required Modifications per Journal

### TMC Version ✅ CURRENT
- Balanced approach
- Strong experiments
- Practical focus
- 14 pages

### TPAMI Version (If Needed)
- Add theoretical proofs
- Extended related work (3+ pages)
- 10+ baseline comparisons
- Mathematical framework

### TNNLS Version (If Needed)
- Focus on neural architecture
- Add convergence analysis
- Extended ablation studies
- Learning dynamics analysis

## Strengths & Weaknesses

### Strengths
- ✅ Real experimental data (668+ experiments)
- ✅ Consistent cross-domain (83% both protocols)
- ✅ Exceptional calibration (0.001 ECE)
- ✅ Complete reproducibility package

### Potential Concerns
- ⚠️ Title length (consider shortening)
- ⚠️ Conformer LOSO issues (documented)
- ⚠️ High synthetic performance (explained)

## Review Response Strategy

### Likely Criticisms
1. **"Why 99% on synthetic?"**
   - Response: Expected for controlled generation, real data shows 83%

2. **"Conformer LOSO failure?"**
   - Response: Architecture-specific issue, LORO works well

3. **"Limited to WiFi CSI?"**
   - Response: Principles applicable to other time-series

## Action Items

### Immediate (For TMC)
- [x] Data extraction complete
- [x] Figures updated with real data
- [x] Text consistency verified
- [ ] Final PDF compilation
- [ ] Cover letter personalization
- [ ] Online submission

### Future Improvements
- [ ] Add more recent references (2024)
- [ ] Include SignFi dataset results
- [ ] Extract real attention weights
- [ ] Create demo video
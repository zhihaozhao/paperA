# 🗺️ Master Navigation - All Papers

## 📚 Paper Portfolio Overview

| Paper | Title | Status | Target | Priority |
|-------|-------|--------|--------|----------|
| **Paper 1** | [Sim2Real Approach](#paper-1-sim2real) | 🟡 Needs Review | IoTJ/TMC | Medium |
| **Paper 2** | [PASE-Net Architecture](#paper-2-pase-net) | ✅ Ready | TMC | **HIGH** |
| **Paper 3** | [Zero-Shot Learning](#paper-3-zero-shot) | 🔴 In Progress | TKDE/PR | Low |

---

## 📁 Directory Structure

```
paper/
├── paper1_sim2real/          # Synthetic data generation
├── paper2_pase_net/          # Architecture innovation ✅
├── paper3_zero_shot/         # Zero-shot learning
└── common_resources/         # Shared resources
```

---

## Paper 1: Sim2Real {#paper-1-sim2real}

### 📍 Location
[`paper1_sim2real/`](paper1_sim2real/)

### 📄 Main Files
- **Manuscript**: [`main.tex`](paper1_sim2real/manuscript/main.tex)
- **Strategy**: [`SUBMISSION_STRATEGY.md`](paper1_sim2real/SUBMISSION_STRATEGY.md)

### 🎯 Target Journals
1. **IEEE IoT Journal** (IoTJ) - 1st choice
   - [Cover Letter](paper1_sim2real/submissions/IoTJ/cover_letter.md)
   - Impact Factor: 10.6
   - Review: 3-4 months

2. **IEEE TMC** - 2nd choice
   - [Cover Letter](paper1_sim2real/submissions/TMC/cover_letter.md)
   - Impact Factor: 7.9
   - Review: 6-8 months

3. **Sensors MDPI** - Backup
   - [Cover Letter](paper1_sim2real/submissions/Sensors/cover_letter.md)
   - Fast review: 1-2 months

### 📊 Key Results
- Label efficiency: 82% with 20% labels
- Sim2Real transfer validated
- 668+ synthetic experiments

### ⚠️ Status
- **Manuscript**: Needs formatting check
- **Experiments**: Complete
- **Cover Letters**: Need update

[↑ Back to top](#-master-navigation---all-papers)

---

## Paper 2: PASE-Net {#paper-2-pase-net}

### 📍 Location
[`paper2_pase_net/`](paper2_pase_net/)

### 📄 Main Files
- **Manuscript**: [`enhanced_claude_v1.tex`](paper2_pase_net/manuscript/enhanced_claude_v1.tex) ✅
- **Strategy**: [`SUBMISSION_STRATEGY.md`](paper2_pase_net/SUBMISSION_STRATEGY.md)
- **Supplementary**: [`SUPPLEMENTARY_MATERIALS.tex`](paper2_pase_net/manuscript/SUPPLEMENTARY_MATERIALS.tex)

### 🎯 Target Journals
1. **IEEE TMC** - Ready to submit! ✅
   - [Cover Letter](paper2_pase_net/submissions/TMC/cover_letter.md)
   - All data verified
   - Tag: TMC_v1.1

2. **IEEE TPAMI** - If TMC rejects
   - [Cover Letter](paper2_pase_net/submissions/TPAMI/cover_letter.md)
   - Needs theory enhancement

3. **IEEE TNNLS** - Alternative
   - [Cover Letter](paper2_pase_net/submissions/TNNLS/cover_letter.md)
   - Neural network focus

### 📊 Key Results (All Real Data!)
- **LOSO/LORO**: 83.0% (perfectly consistent!)
- **Calibration**: ECE 0.094→0.001 (99% reduction)
- **Fall Detection**: >99% for specific types
- **Real Dataset**: WiFi-CSI-Sensing-Benchmark

### ✅ Ready for Submission
- **Data**: All extracted and verified
- **Figures**: Updated with real data
- **Text**: Consistent with results
- **Git**: Tagged as TMC_v1.1

### 📋 Quick Actions
```bash
# Compile PDF
cd paper2_pase_net/manuscript
pdflatex enhanced_claude_v1.tex
bibtex enhanced_claude_v1
pdflatex enhanced_claude_v1.tex
pdflatex enhanced_claude_v1.tex
```

[↑ Back to top](#-master-navigation---all-papers)

---

## Paper 3: Zero-Shot {#paper-3-zero-shot}

### 📍 Location
[`paper3_zero_shot/`](paper3_zero_shot/)

### 📄 Main Files
- **Manuscript**: [`zeroshot.tex`](paper3_zero_shot/manuscript/zeroshot.tex)
- **Strategy**: [`SUBMISSION_STRATEGY.md`](paper3_zero_shot/SUBMISSION_STRATEGY.md)

### 🎯 Target Journals
1. **IEEE TKDE** - Knowledge transfer focus
   - Impact Factor: 8.9
   - Review: 4-6 months

2. **Pattern Recognition** - Signal patterns
   - Impact Factor: 8.0
   - Review: 3-5 months

3. **IEEE Access** - Fast publication
   - Open Access
   - Review: 1-2 months

### 📊 Status
- **Manuscript**: Needs completion
- **Experiments**: Need verification
- **Strategy**: Planned

### ⚠️ Required Actions
- [ ] Complete manuscript review
- [ ] Verify experimental results
- [ ] Update references (2023-2024)
- [ ] Prepare cover letters

[↑ Back to top](#-master-navigation---all-papers)

---

## 🎯 Priority Actions

### Immediate (Paper 2 - PASE-Net)
1. ✅ Compile final PDF
2. ✅ Submit to TMC
3. ✅ Save confirmation

### Short-term (Paper 1 - Sim2Real)
1. Review manuscript formatting
2. Update cover letters
3. Prepare for IoTJ submission

### Medium-term (Paper 3 - Zero-Shot)
1. Complete manuscript
2. Run missing experiments
3. Choose target journal

---

## 📊 Submission Timeline

```
2024 Q4:
- Week 1: Submit Paper 2 to TMC ✅
- Week 2-3: Prepare Paper 1 for IoTJ
- Week 4: Review Paper 3 status

2025 Q1:
- January: Submit Paper 1
- February: Paper 3 preparation
- March: Submit Paper 3
```

---

## 🔧 Common Resources

### Shared Data
- [`common_resources/experimental_data/`](common_resources/experimental_data/)
  - WiFi-CSI-Sensing-Benchmark
  - 668+ experimental results

### Templates
- [`common_resources/templates/`](common_resources/templates/)
  - IEEE format
  - Cover letter templates

### Utilities
- [`common_resources/utilities/`](common_resources/utilities/)
  - Data extraction scripts
  - Figure generation tools

---

## 📚 Quick References

### Git Information
- **Repository**: https://github.com/zhihaozhao/paperA
- **Branch**: `feat/enhanced-model-and-sweep`
- **Latest Tag**: `TMC_v1.1`

### Compilation Commands
```bash
# For any paper
cd paper[N]_*/manuscript/
pdflatex [main_file].tex
bibtex [main_file]
pdflatex [main_file].tex
pdflatex [main_file].tex
```

### Submission Portals
- **TMC**: https://mc.manuscriptcentral.com/tmc-cs
- **IoTJ**: https://mc.manuscriptcentral.com/iot
- **TPAMI**: https://mc.manuscriptcentral.com/tpami-cs
- **IEEE Access**: https://ieeeaccess.ieee.org/

---

## ✅ Checklist for Each Submission

### Before Submission
- [ ] Manuscript compiled without errors
- [ ] All figures included and referenced
- [ ] Page limit compliance
- [ ] References complete and formatted
- [ ] Cover letter personalized
- [ ] Supplementary materials prepared
- [ ] Co-author approval obtained

### During Submission
- [ ] Create/login to journal system
- [ ] Upload all required files
- [ ] Fill in metadata correctly
- [ ] Select appropriate keywords
- [ ] Suggest reviewers (if required)
- [ ] Confirm submission

### After Submission
- [ ] Save confirmation email
- [ ] Note manuscript ID
- [ ] Set reminder for follow-up
- [ ] Prepare for reviewer responses

---

**Last Updated**: 2024-12-04
**Next Review**: After Paper 2 TMC submission
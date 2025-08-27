# Comprehensive Research Roadmap

## Executive Summary

This roadmap outlines the strategic plan for advancing our WiFi CSI-based human activity recognition research, targeting top-tier venues with physics-informed synthetic data generation and enhanced architectures.

## Current Status (December 2024)

### Completed ✅
- **Innovation Checklist**: Comprehensive mapping of innovations to benchmarks
- **Baseline Reproductions**: REPRO_PLAN documents for SenseFi, FewSense, AirFi, ReWiS
- **Paper Drafts**: 10-page skeletons for Exp1 (Physics-LSTM) and Exp2 (Mamba)
- **Bibliography**: Extracted 29 papers with metadata and metrics
- **3D Figures**: Enhanced visualizations for main paper

### In Progress 🔄
- Exp1: Multi-scale LSTM + PINN implementation (stub ready)
- Exp2: Mamba SSM replacement (stub ready)
- Baseline reproductions with actual data
- Comprehensive ablation studies

### Pending ⏳
- Cross-dataset validation (SignFi, Widar)
- User study for synthetic data quality
- Hardware profiling and optimization
- Statistical significance testing

---

## Target Venues and Timeline

### Tier 1: Machine Learning Conferences

#### NeurIPS 2025
- **Submission**: May 2025
- **Focus**: Physics-informed learning + synthetic data generation
- **Paper**: Enhanced main paper with full experiments
- **Requirements**:
  - [ ] Complete ablation studies (by March 2025)
  - [ ] Statistical significance testing (by April 2025)
  - [ ] Cross-dataset validation (by April 2025)
  - [ ] Camera-ready prep (by May 2025)

#### ICML 2025
- **Submission**: January 2025
- **Focus**: Sample-efficient learning with physics constraints
- **Paper**: Exp1 (Multi-scale LSTM + PINN)
- **Requirements**:
  - [ ] Complete Exp1 implementation (by Dec 2024)
  - [ ] Baseline comparisons (by Jan 2025)
  - [ ] Physics loss validation (by Jan 2025)

#### ICLR 2025
- **Submission**: September 2024 (missed) → Target ICLR 2026
- **Focus**: State-space models for sensing
- **Paper**: Exp2 (Mamba replacement)
- **Timeline**: September 2025 submission

### Tier 2: Systems and Sensing Conferences

#### MobiCom 2025
- **Submission**: August 2025
- **Focus**: Edge deployment and real-time performance
- **Paper**: System implementation paper
- **Requirements**:
  - [ ] Hardware profiling complete
  - [ ] Edge device optimization
  - [ ] Real-world deployment study

#### SenSys 2025
- **Submission**: April 2025
- **Focus**: Cross-domain WiFi sensing
- **Paper**: Enhanced architecture with deployment
- **Requirements**:
  - [ ] Multi-environment testing
  - [ ] Robustness evaluation
  - [ ] System integration demo

### Tier 3: AI/Applied Conferences

#### AAAI 2025
- **Submission**: August 2024 (missed) → Target AAAI 2026
- **Focus**: Applications and deployment
- **Timeline**: August 2025 submission

#### IJCAI 2025
- **Submission**: January 2025
- **Focus**: Trustworthy AI for sensing
- **Paper**: Calibration and uncertainty work
- **Requirements**:
  - [ ] Complete trustworthy evaluation
  - [ ] OOD detection experiments
  - [ ] Calibration analysis

---

## Priority Task List (Q1 2025)

### Week 1 (Current)
1. ✅ Innovation checklist with benchmark mapping
2. ✅ Baseline REPRO_PLANs (4 completed)
3. ✅ Paper draft skeletons (Exp1, Exp2)
4. ✅ Bibliography extraction
5. 🔄 This roadmap

### Week 2
1. [ ] Complete Exp1 implementation (beyond stub)
2. [ ] Complete Exp2 implementation (beyond stub)
3. [ ] Run baseline reproductions (SenseFi, FewSense)
4. [ ] Initial performance benchmarking
5. [ ] Setup continuous integration

### Week 3
1. [ ] Ablation studies (SE module, attention, physics)
2. [ ] Cross-domain evaluation (CDAE protocol)
3. [ ] Few-shot experiments (STEA protocol)
4. [ ] Statistical significance testing
5. [ ] Generate result tables and plots

### Week 4
1. [ ] Cross-dataset validation (SignFi, Widar)
2. [ ] Hardware profiling and optimization
3. [ ] Edge deployment testing
4. [ ] User study design and execution
5. [ ] Paper writing and polishing

---

## Technical Milestones

### Milestone 1: Baseline Superiority (Jan 2025)
- [ ] Enhanced model > 83% F1 on SenseFi
- [ ] Physics-LSTM > 85% F1 target
- [ ] Mamba > 80% F1 with 3x throughput
- [ ] All results statistically significant (p < 0.05)

### Milestone 2: Physics Validation (Feb 2025)
- [ ] Synthetic data statistical validation
- [ ] Physics loss convergence proof
- [ ] Ablation showing +15% over random
- [ ] User study confirming realism

### Milestone 3: Deployment Ready (Mar 2025)
- [ ] Edge device < 50ms latency
- [ ] Model compression to < 5MB
- [ ] Real-time demo application
- [ ] Docker containerization

### Milestone 4: Paper Submission (Apr-May 2025)
- [ ] Main paper to NeurIPS/ICML
- [ ] System paper to SenSys
- [ ] Workshop papers ready
- [ ] Supplementary materials complete

---

## Risk Mitigation

### Technical Risks
1. **Physics model accuracy**
   - Mitigation: Extensive validation against real data
   - Backup: Hybrid physics + learned model

2. **Mamba implementation challenges**
   - Mitigation: Collaborate with original authors
   - Backup: Use simplified SSM variant

3. **Cross-dataset generalization**
   - Mitigation: Domain adaptation techniques
   - Backup: Dataset-specific fine-tuning

### Timeline Risks
1. **Exp1/Exp2 implementation delays**
   - Mitigation: Parallel development tracks
   - Backup: Focus on one experiment

2. **Baseline reproduction issues**
   - Mitigation: Contact original authors
   - Backup: Use reported numbers

3. **Review/revision time**
   - Mitigation: Start writing early
   - Backup: Target later venues

---

## Resource Requirements

### Compute Resources
- **GPU Hours**: 500 hours for full experiments
- **Storage**: 100GB for datasets and checkpoints
- **Memory**: 32GB GPU for Mamba training

### Human Resources
- **Lead researcher**: Full-time
- **Collaborators**: 2-3 for experiments
- **Domain expert**: Physics consultation
- **Writing support**: Paper preparation

### External Dependencies
- **Datasets**: SignFi, Widar access confirmed
- **Hardware**: Edge devices for deployment
- **Software**: PyTorch, CUDA, Mamba library

---

## Success Metrics

### Research Impact
- [ ] 3+ top-tier publications
- [ ] 100+ citations within 2 years
- [ ] Open-source release with 500+ stars
- [ ] Industry adoption/interest

### Technical Achievement
- [ ] State-of-the-art performance (>85% F1)
- [ ] 80% data reduction demonstrated
- [ ] Real-time edge deployment (<50ms)
- [ ] Cross-domain robustness proven

### Community Contribution
- [ ] Reproducible benchmark released
- [ ] Tutorial/workshop organized
- [ ] Code and models open-sourced
- [ ] Documentation and guides published

---

## Long-term Vision (2025-2027)

### Year 1 (2025)
- Establish physics-informed WiFi sensing
- Deploy in 2-3 real environments
- Publish foundational papers

### Year 2 (2026)
- Extend to mmWave and UWB
- Multi-modal sensor fusion
- Commercial partnerships

### Year 3 (2027)
- Industry standard for WiFi HAR
- Healthcare and eldercare applications
- Startup or technology transfer

---

## Action Items (Immediate)

### This Week
- [x] Complete innovation checklist
- [x] Update baseline REPRO_PLANs
- [x] Create paper drafts
- [x] Extract bibliography
- [x] Draft this roadmap
- [ ] Begin Exp1 implementation
- [ ] Setup experiment tracking

### Next Week
- [ ] Complete Exp1/Exp2 beyond stubs
- [ ] Run first baseline reproduction
- [ ] Setup CI/CD pipeline
- [ ] Begin ablation studies
- [ ] Draft introduction sections

### This Month
- [ ] All experiments running
- [ ] Initial results collected
- [ ] Paper drafts expanded
- [ ] Submission strategy finalized
- [ ] Collaborations established

---

## Communication Plan

### Internal
- Weekly progress meetings
- Shared experiment tracking
- Version-controlled documentation
- Slack/Discord for quick updates

### External
- Conference submissions
- ArXiv preprints
- Twitter/social media updates
- Blog posts on key findings

---

## Budget Allocation

### Estimated Costs
- **Compute**: $2,000 (cloud GPU)
- **Conference travel**: $5,000 x 2
- **Hardware**: $3,000 (edge devices)
- **Publication fees**: $2,000
- **Total**: ~$17,000

### Funding Sources
- Research grant (primary)
- Industry collaboration
- Conference scholarships
- Department support

---

## Conclusion

This roadmap provides a clear path from current experimental state to top-tier publications. The combination of physics-informed learning, efficient architectures, and comprehensive evaluation positions our work for significant impact in the WiFi sensing community.

**Next Steps**:
1. Complete current week's tasks
2. Begin Exp1/Exp2 implementation
3. Establish collaboration channels
4. Start paper writing in parallel

**Success Factors**:
- Rigorous experimental validation
- Clear innovation narrative
- Timely execution
- Strong empirical results

The roadmap will be updated monthly based on progress and new opportunities.
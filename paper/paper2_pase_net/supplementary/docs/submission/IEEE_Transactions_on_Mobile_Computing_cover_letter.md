# Cover Letter - IEEE Transactions on Mobile Computing

Dear Editor-in-Chief,

We are pleased to submit our manuscript "Physics-Guided Synthetic WiFi CSI Data Generation for Trustworthy Human Activity Recognition: A Sim2Real Approach" for consideration in IEEE Transactions on Mobile Computing.

## Executive Summary

This manuscript presents groundbreaking research in mobile and pervasive computing, addressing the critical data scarcity challenge that limits practical deployment of WiFi-based human activity recognition systems. We introduce a novel physics-guided synthetic data generation framework combined with Sim2Real transfer learning, achieving near-fully-supervised performance with 80% fewer labeled samples.

## Core Technical Contributions

Our work advances the state-of-the-art in mobile computing through:

1. **Physics-Based Mobile Sensing**: First systematic integration of electromagnetic propagation physics into synthetic CSI data generation, modeling multipath effects, Doppler shifts, and device mobility patterns essential for mobile computing scenarios.

2. **Efficient Mobile Learning**: Breakthrough in sample efficiency—82.1% F1 score with only 20% real data—critical for mobile devices with limited storage and battery constraints.

3. **Robust Cross-Environment Performance**: Identical 83.0% performance across subject and room variations, essential for mobile applications transitioning between diverse environments.

4. **Mobile-Optimized Architecture**: Enhanced CNN with attention mechanisms designed for efficient inference on mobile processors, achieving 3.2 GFLOPs computational complexity suitable for edge deployment.

## Alignment with TMC Scope

This work directly addresses TMC's core areas:

- **Wireless Communication and Mobile Networks**: Deep understanding of WiFi CSI characteristics and channel modeling
- **Mobile Sensing and Context Awareness**: Novel approaches to device-free sensing using existing WiFi infrastructure
- **Machine Learning for Mobile Systems**: Specialized architectures and training strategies for resource-constrained mobile environments
- **Performance Evaluation**: Comprehensive benchmarking across 540+ configurations with rigorous statistical validation

## Technical Depth and Rigor

The manuscript provides:
- Mathematical formulation of physics-guided CSI synthesis
- Detailed architectural specifications with complexity analysis
- Three systematic evaluation protocols (SRV, CDAE, STEA) totaling 636 experimental configurations
- Statistical significance testing with p-values and effect sizes
- Comparison against 11 state-of-the-art models from SenseFi benchmark

## Mobile Computing Impact

Our approach enables practical deployment of WiFi sensing on mobile devices by:
- Reducing training data requirements by 80%, enabling on-device personalization
- Achieving consistent performance across environmental transitions
- Providing calibrated confidence scores for battery-efficient selective processing
- Supporting incremental learning with minimal labeled data

## Innovation Beyond State-of-the-Art

Unlike existing work that assumes abundant labeled data (e.g., SenseFi benchmark), we:
- Pioneer Sim2Real transfer for WiFi sensing
- Demonstrate physics-informed architectural design without explicit PDE constraints
- Achieve unprecedented cross-domain consistency (CV<0.2%)
- Provide trustworthy predictions with ECE=0.0072 calibration error

## Reproducibility and Community Impact

We commit to:
- Releasing source code and synthetic data generation framework
- Providing pre-trained models for community use
- Detailed implementation specifications for reproduction
- Benchmark protocols for fair comparison

## Ethical Considerations

The research involves no human subjects data collection, focusing entirely on synthetic data generation and public benchmark datasets. Privacy-preserving nature of WiFi sensing is emphasized throughout.

## Declaration of Originality

This manuscript represents original work not under consideration elsewhere. All co-authors have reviewed and approved the submission. We declare no conflicts of interest.

We believe this work makes fundamental contributions to mobile and pervasive computing, enabling practical deployment of ubiquitous sensing applications. The combination of theoretical innovation, practical impact, and rigorous evaluation makes it highly suitable for TMC's audience.

Thank you for considering our submission.

Sincerely,

[Authors]

## Recommended Reviewers

1. Prof. Romit Roy Choudhury (UIUC) - Mobile sensing and wireless systems
2. Prof. Shwetak Patel (University of Washington) - Ubiquitous computing and sensing
3. Prof. Hae Young Noh (Stanford) - Infrastructure sensing and mobile computing

## Keywords

Mobile Computing, WiFi Sensing, Channel State Information, Sim2Real Transfer Learning, Physics-Guided Synthesis, Cross-Domain Adaptation, Edge Intelligence
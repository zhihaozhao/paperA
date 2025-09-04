# 📋 Paper Organization Plan

## Current Papers Identified

### Paper 1: Sim2Real Approach
- **File**: `main.tex`
- **Title**: "Physics-Guided Synthetic WiFi CSI Data Generation for Trustworthy Human Activity Recognition: A Sim2Real Approach"
- **Focus**: Synthetic data generation and Sim2Real transfer
- **Target Journals**: IoTJ, TMC, Sensors

### Paper 2: PASE-Net Architecture (Enhanced)
- **File**: `enhanced/enhanced_claude_v1.tex`
- **Title**: "Physics-Informed PASE-Net Architecture for WiFi CSI Human Activity Recognition"
- **Focus**: Novel architecture with attention mechanisms
- **Target Journals**: TMC, TPAMI, TNNLS

### Paper 3: Zero-Shot Learning
- **Directory**: `zero/`
- **Files**: `zeroshot.tex`, `zeroshot_claude4.1opus.tex`
- **Focus**: Zero-shot learning approach
- **Target Journals**: TKDE, PR, IEEE Access

## Proposed New Structure

```
paper/
├── paper1_sim2real/
│   ├── manuscript/
│   │   ├── main.tex
│   │   ├── refs.bib
│   │   └── figures/
│   ├── submissions/
│   │   ├── TMC/
│   │   │   ├── cover_letter.md
│   │   │   └── submission_checklist.md
│   │   ├── IoTJ/
│   │   │   ├── cover_letter.md
│   │   │   └── submission_checklist.md
│   │   └── Sensors/
│   │       ├── cover_letter.md
│   │       └── submission_checklist.md
│   └── supplementary/
│       ├── data/
│       ├── scripts/
│       └── docs/
│
├── paper2_pase_net/
│   ├── manuscript/
│   │   ├── enhanced_claude_v1.tex
│   │   ├── enhanced_refs.bib
│   │   └── SUPPLEMENTARY_MATERIALS.tex
│   ├── submissions/
│   │   ├── TMC/
│   │   ├── TPAMI/
│   │   └── TNNLS/
│   └── supplementary/
│       ├── data/
│       ├── scripts/
│       └── docs/
│
├── paper3_zero_shot/
│   ├── manuscript/
│   │   ├── zeroshot.tex
│   │   ├── zero_refs.bib
│   │   └── figures/
│   ├── submissions/
│   │   ├── TKDE/
│   │   ├── PR/
│   │   └── IEEE_Access/
│   └── supplementary/
│       ├── data/
│       ├── scripts/
│       └── docs/
│
└── common_resources/
    ├── templates/
    ├── experimental_data/
    └── utilities/
```

## Benefits of This Structure

1. **Clear Separation**: Each paper has its own complete directory
2. **Multiple Submissions**: Easy to manage different journal submissions
3. **Version Control**: Can track changes per paper/journal
4. **Reusability**: Common resources shared across papers
5. **Organization**: Supplementary materials organized per paper
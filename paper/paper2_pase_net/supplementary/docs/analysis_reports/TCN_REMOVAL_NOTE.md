# TCN Removal from PASE-Net Paper

## Issue Identified
TCN (Temporal Convolutional Network) was mentioned only in Table 1 but:
- Not discussed in the methodology section
- Not analyzed in the results discussion
- Not compared in the text
- Appears to be inconsistent with actual experiments

## Resolution
Removed TCN from the paper entirely:

### 1. Table 1 (Line 300)
**Removed**: TCN row with results (80.5±0.9 LOSO, 79.8±1.1 LORO, etc.)

### 2. Abbreviations Table (Line 573)
**Removed**: "TCN & Temporal Convolutional Network"

## Current Baseline Models
The paper now consistently compares with:
1. **CNN** - Basic convolutional baseline
2. **BiLSTM** - Recurrent baseline
3. **Conformer** - Attention-based baseline

These three baselines are:
- Properly introduced in the methodology
- Discussed in the results
- Analyzed in comparisons throughout the text

## Justification
- TCN was likely added as an additional baseline but never fully integrated
- Its removal makes the paper more consistent
- The remaining baselines provide sufficient comparison:
  - CNN: Non-temporal baseline
  - BiLSTM: Sequential/recurrent approach
  - Conformer: Modern attention-based approach

## Alternative (If TCN experiments exist)
If you actually have TCN experimental results and want to keep them:

1. Add TCN introduction in Section II (Related Work):
```latex
Temporal Convolutional Networks (TCNs) have emerged as an alternative 
to RNNs for sequence modeling, using dilated convolutions to capture 
long-range dependencies...
```

2. Add TCN discussion in results (Section V):
```latex
TCN achieves intermediate performance (80.5% LOSO, 79.8% LORO), 
outperforming CNN but falling short of BiLSTM and PASE-Net...
```

3. Keep the table entry and abbreviation

## Recommendation
The current solution (removing TCN) is cleaner and more consistent with the paper's narrative, which focuses on comparing against CNN (spatial), BiLSTM (temporal), and Conformer (attention) baselines.
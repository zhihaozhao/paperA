# PASE-Net Paper Figure Coverage Analysis

## Current Figure Distribution

### Main Experimental Sections and Their Figures

#### 1. **System Overview** ✅
- **Figure 1** (`fig:system_architecture`): System architecture overview
- Location: Line 62-64
- Status: Active

#### 2. **Physics Modeling** ✅
- **Figure 2** (`fig:physics_modeling`): Physics-based modeling illustration
- Location: Line 272-274
- Referenced: Line 216
- Status: Active

#### 3. **Calibration Results (SRV Protocol)** ✅
- **Figure 3** (`fig:calibration`): Calibration and SRV results
- Location: Line 279-281
- Referenced: Line 309
- Status: Active

#### 4. **Cross-Domain Adaptation (CDAE Protocol)** ✅
- **Figure 4** (`fig:cross_domain`): LOSO/LORO cross-domain results
- Location: Line 327-332
- Referenced: Line 323
- Status: Active

#### 5. **Label Efficiency (STEA Protocol)** ✅
- **Figure 5** (`fig:label_efficiency`): Sim2Real transfer efficiency
- Location: Line 334-339
- Referenced: Line 362
- Status: Active

#### 6. **Progressive Temporal Analysis** ⚠️
- **Figure 6** (`fig:progressive_temporal`): Temporal granularity analysis
- Location: Line 341-346
- Referenced: Line 350
- Status: **Now Active** (was commented)
- **Necessity**: QUESTIONABLE - This is supplementary analysis, not a main protocol

#### 7. **Nuisance Factor Analysis** ⚠️
- **Figure 7** (`fig:nuisance_factors`): Environmental robustness heatmap
- Location: Line 385-390
- Referenced: Line 394
- Status: **Now Active** (was commented)
- **Necessity**: QUESTIONABLE - Partially redundant with SRV results in Figure 3

#### 8. **Component Ablation** ⚠️
- **Figure 8** (`fig:component_analysis`): Component-wise performance
- Location: Line 406-411
- Referenced: Line 413
- Status: **Now Active** (was commented)
- **Necessity**: QUESTIONABLE - Could be moved to supplementary or table

#### 9. **Interpretability** ✅
- **Figure 9** (`fig:interpretability`): Attribution analysis
- Location: Line 467-469
- Referenced: Line 472
- Status: Active

## Analysis Summary

### Core Protocol Coverage (Required)
✅ **SRV Protocol**: Figure 3 (Calibration)
✅ **CDAE Protocol**: Figure 4 (Cross-domain)
✅ **STEA Protocol**: Figure 5 (Label efficiency)

### Additional Analyses (Optional)
⚠️ **Progressive Temporal**: Figure 6 - Supplementary analysis
⚠️ **Nuisance Factors**: Figure 7 - Detailed ablation (partially redundant)
⚠️ **Component Analysis**: Figure 8 - Could be table instead

## Recommendation

### Option 1: Keep All Figures (Current State)
**Pros:**
- Comprehensive visual support
- All references work
- More detailed analysis

**Cons:**
- 9 figures may be excessive for journal submission
- Some redundancy between figures
- May exceed page limits

### Option 2: Optimize to Core Figures (Recommended)
**Keep Active:**
1. System Architecture (Fig 1)
2. Physics Modeling (Fig 2) 
3. Calibration/SRV (Fig 3)
4. Cross-Domain/CDAE (Fig 4)
5. Label Efficiency/STEA (Fig 5)
6. Interpretability (Fig 9)

**Move to Supplementary or Remove:**
- Progressive Temporal (Fig 6) → Supplementary
- Nuisance Factors (Fig 7) → Merge with Fig 3 or Supplementary
- Component Analysis (Fig 8) → Convert to table or Supplementary

### Option 3: Strategic Consolidation
**Merge Figures:**
- Combine Figures 3 & 7 into comprehensive "Robustness Analysis"
- Combine Figures 6 & 8 into "Ablation Studies"
- Result: 7 total figures (more reasonable)

## Decision Factors

### Journal Requirements:
- **IEEE TPAMI**: Typically allows 8-10 figures
- **IEEE TMC**: Prefers 6-8 figures
- **Nature MI**: Strict limit of 6-8 display items

### Current Status:
- 9 figures total
- 3 were previously commented (likely for space reasons)
- Text still references all figures

## Recommended Action

**For TMC Submission:**
1. Comment out Figures 6, 7, 8 again
2. Move their content to supplementary materials
3. Update text to say "see supplementary materials" 
4. Keep core 6 figures for main paper

OR

Keep all 9 if journal allows, but be prepared to move 3 to supplementary during revision.

## Code to Re-comment if Needed

```latex
% Comment out these three figures again if space is limited:
% Lines 341-346: Progressive Temporal
% Lines 385-390: Nuisance Factors  
% Lines 406-411: Component Analysis

% Update references to:
% "detailed analysis in supplementary materials shows..."
```
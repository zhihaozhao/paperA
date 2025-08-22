# Statistical Meta-Analysis Scripts for Fruit-Picking Robot Literature

## 🎯 **Overview**
Complete pipeline for transforming literature review into data-driven meta-analysis with high-order statistical figures for journal submission.

---

## 📁 **Directory Structure**
```
scripts/
├── README.md                           # This file
├── run_complete_analysis.py            # Master pipeline script
├── data_extraction/
│   └── literature_data_extractor.py    # Extract data from LaTeX tables
├── statistical_analysis/
│   └── meta_analysis_experiments.py    # Statistical experiments & tests
└── figure_generation/
    ├── create_meta_analysis_figures.py # Python high-order figures
    └── motion_control_analysis.m       # MATLAB supplementary analysis
```

---

## 🚀 **Quick Start**

### **Option 1: Complete Pipeline (Recommended)**
```bash
cd /workspace/benchmarks
python scripts/run_complete_analysis.py --tex-file FP_2025_SN-APPLIED-SCIENCES/FP_2025_SN-APPLIED-SCIENCES_v1.tex
```

### **Option 2: Step-by-Step Execution**
```bash
# Step 1: Extract data
python scripts/data_extraction/literature_data_extractor.py

# Step 2: Statistical analysis  
python scripts/statistical_analysis/meta_analysis_experiments.py

# Step 3: Generate figures
python scripts/figure_generation/create_meta_analysis_figures.py

# Step 4: MATLAB analysis (optional)
matlab -batch "motion_control_analysis"
```

---

## 📊 **Pipeline Outputs**

### **📈 Generated Files:**
- `fruit_picking_literature_data.csv` - Structured dataset (137 studies)
- `meta_analysis_results.json` - Statistical analysis results
- `fig_meta_analysis_visual_detection.png/.pdf` - Visual detection meta-analysis
- `fig_motion_control_analysis.png/.pdf` - Motion control analysis
- `fig_technology_roadmap.png/.pdf` - Technology evolution & projections
- `fig_comprehensive_dashboard.png/.pdf` - Complete dashboard
- `integration_report.md` - LaTeX integration instructions

### **📊 Statistical Experiments Performed:**
1. **Performance Distribution Analysis** - Algorithm family comparisons
2. **Correlation Analysis** - Metric relationships and PCA
3. **Temporal Trend Analysis** - Technology evolution over time
4. **Environmental Impact Assessment** - Performance across conditions
5. **Technology Adoption Patterns** - Algorithm lifecycle analysis

---

## 🔬 **Scientific Value**

### **✅ Original Contributions:**
- First comprehensive meta-analysis of fruit-picking robot performance (2015-2024)
- Quantitative evidence for algorithm selection guidelines
- Statistical validation of performance claims
- Evidence-based technology roadmap with projections
- High-order figures suitable for top-tier journals

### **✅ Journal Integration:**
- **SN Applied Sciences**: Single-column figures (0.9\textwidth)
- **IEEE Access**: Double-column figures (0.95\textwidth)  
- **RAS**: Technical emphasis with statistical rigor
- **Discover Robotics**: Innovation focus with trend analysis

---

## 💻 **Requirements**

### **Python Dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### **MATLAB Requirements (Optional):**
- MATLAB R2020a or newer
- Statistics and Machine Learning Toolbox
- Curve Fitting Toolbox (recommended)

---

## 🔧 **Usage Examples**

### **1. Extract Data Only:**
```python
from data_extraction.literature_data_extractor import LiteratureDataExtractor

extractor = LiteratureDataExtractor('paper.tex')
data = extractor.extract_all_data()
extractor.save_data('output.csv')
```

### **2. Generate Specific Figure:**
```python
from figure_generation.create_meta_analysis_figures import MetaAnalysisFigureGenerator

generator = MetaAnalysisFigureGenerator('data.csv', 'results.json')
generator.create_figure_4_visual_detection_analysis()
```

### **3. Run Statistical Tests:**
```python
from statistical_analysis.meta_analysis_experiments import MetaAnalysisExperiments

analyzer = MetaAnalysisExperiments('data.csv')
results = analyzer.run_all_experiments()
```

---

## 📋 **Data Schema**

### **Extracted Metrics:**
- **Performance**: accuracy, f1_score, map_score, success_rate
- **Speed**: speed_ms, speed_fps, cycle_time, processing_time  
- **Complexity**: model_size, epochs, training_images
- **Context**: algorithm_family, environment, fruit_type, year

### **Statistical Results:**
- **Distributions**: Mean, std, median, quartiles per algorithm family
- **Correlations**: Pearson/Spearman correlations with p-values
- **Trends**: Linear/polynomial fits with R² and projections
- **ANOVA**: Between-group differences with effect sizes
- **Clustering**: Algorithm groupings based on performance

---

## 🎨 **Figure Specifications**

### **Publication Quality:**
- **Resolution**: 300 DPI for all outputs
- **Formats**: PNG (for viewing), PDF (for LaTeX), EPS (compatibility)
- **Style**: Publication-ready with proper fonts and colors
- **Size**: Optimized for journal layouts (single/double column)

### **Figure Content:**
- **Figure 4**: Visual detection meta-analysis (4 panels)
- **Figure 5**: Motion control statistical analysis (6 panels)
- **Figure 6**: Technology roadmap & projections (4 panels)
- **Figure 7**: Comprehensive meta-analysis dashboard (6 panels)

---

## 🚨 **Important Notes**

### **⚠️ Data Quality:**
- Extraction depends on consistent LaTeX table formatting
- Manual validation recommended for critical metrics
- Missing data handled gracefully with NaN values

### **⚠️ Statistical Validity:**
- All tests include appropriate significance thresholds
- Effect sizes calculated for practical significance
- Multiple comparison corrections applied where appropriate

### **⚠️ Figure Integration:**
- Test compilation after figure integration
- Adjust figure sizes based on journal requirements
- Ensure figure quality meets journal standards

---

## 📧 **Support**
For issues or questions:
1. Check console output for detailed error messages
2. Validate input LaTeX file format and content
3. Ensure all dependencies are installed
4. Refer to integration_report.md for LaTeX usage

This pipeline transforms literature surveys into high-impact, data-driven scientific contributions suitable for top-tier journal submission.
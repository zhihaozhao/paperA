# WiFi-CSI Sensing Core Repository

This is the main code repository for the WiFi CSI sensing research project. It contains the core algorithms, experimental scripts, and development tools.

## 🏗️ Multi-Repository Architecture

This project now uses a **multi-repository architecture** to avoid file coupling and branch conflicts:

### 📊 [WiFi-CSI-Sensing-Results](./repos/WiFi-CSI-Sensing-Results/)
- **Purpose**: Experimental results, logs, and generated visualizations
- **Content**: `results/`, `results_gpu/`, `tables/`, plots and analysis data
- **Usage**: Data storage and analysis, supports large files

### 📝 [WiFi-CSI-Journal-Paper](./repos/WiFi-CSI-Journal-Paper/)  
- **Purpose**: Journal paper LaTeX sources and references
- **Content**: `paper/`, `references/`, submission materials
- **Target**: IoTJ, TMC, IMWUT (top-tier IoT/mobile computing journals)

### 🎓 [WiFi-CSI-PhD-Thesis](./repos/WiFi-CSI-PhD-Thesis/)
- **Purpose**: PhD dissertation LaTeX sources
- **Content**: `论文/`, chapters, appendices, and defense materials
- **Usage**: Thesis writing and committee submissions

### 💻 WiFi-CSI-Sensing-Core (This Repository)
- **Purpose**: Core algorithms, scripts, and development tools
- **Content**: `src/`, `scripts/`, `eval/`, configuration files

## 📁 Repository Structure

```
├── src/                 # Core algorithm implementations
│   ├── models.py       # Enhanced CNN+SE+Attention models
│   ├── data_*.py       # Data loading and preprocessing
│   ├── evaluate.py     # Evaluation framework
│   └── calibration.py  # Trustworthy evaluation tools
├── scripts/            # Experimental and analysis scripts
│   ├── run_*.sh       # Training and evaluation scripts
│   ├── analyze_*.py   # Result analysis tools
│   └── sweep_*.py     # Hyperparameter optimization
├── eval/              # Evaluation tools and benchmarks
├── benchmarks/        # External benchmark frameworks
├── docs/              # Documentation and guides
└── env.yml           # Environment configuration
```

## 🚀 Quick Start

### Environment Setup
```bash
# Create conda environment
conda env create -f env.yml
conda activate paperA

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments
```bash
# Basic model training
./scripts/run_train.sh

# Cross-domain evaluation (LOSO/LORO)
./scripts/run_real_loso.sh

# Sim2Real label efficiency analysis
./scripts/run_sim2real.sh

# Generate analysis and figures
python scripts/analyze_d3_d4_for_figures.py
```

## 🔬 Key Features

### Enhanced Model Architecture
- **CNN+SE+Attention**: Lightweight temporal attention with squeeze-and-excitation
- **Capacity-matched comparisons**: Fair evaluation against CNN, BiLSTM, TCN, Conformer-lite
- **Multi-scale feature extraction**: Optimized for WiFi CSI data characteristics

### Trustworthy Evaluation Framework
- **Calibration metrics**: ECE, NLL for reliability assessment
- **Cross-domain validation**: LOSO (Leave-One-Subject-Out), LORO (Leave-One-Room-Out)
- **Statistical significance**: Proper error bars and hypothesis testing

### Label Efficiency Analysis
- **Sim2Real transfer**: 10-20% labels achieving 90-95% performance
- **Few-shot learning**: Domain adaptation with minimal supervision
- **Temperature scaling**: Calibrated probability outputs

## 📊 Related Repositories

Access related materials:
```bash
# View experimental results
cd repos/WiFi-CSI-Sensing-Results/

# Work on journal paper
cd repos/WiFi-CSI-Journal-Paper/

# PhD thesis writing
cd repos/WiFi-CSI-PhD-Thesis/
```

## 🏆 Research Contributions

1. **Enhanced architecture** with attention mechanisms for WiFi CSI
2. **Comprehensive calibration** framework for trustworthy predictions  
3. **Cross-domain generalization** validation (LOSO/LORO protocols)
4. **Sim2Real efficiency** analysis with minimal labeled data
5. **Capacity-matched benchmarks** for fair model comparisons

## 📖 Publications

- **Journal Paper**: Submitted to IoTJ/TMC/IMWUT (see journal paper repo)
- **PhD Thesis**: "Deep Learning Approaches for Trustworthy WiFi CSI Sensing" (see thesis repo)

## 🛠️ Development

### Branch Strategy
- `feat/enhanced-model-and-sweep`: Main development branch
- `master`: Stable releases
- Feature branches: `feat/feature-name`

### Contributing
1. Create feature branch from `feat/enhanced-model-and-sweep`
2. Implement changes in appropriate repository
3. Update tests and documentation
4. Submit pull request

## 📧 Contact

For questions about this research:
- Core algorithms: This repository
- Experimental results: WiFi-CSI-Sensing-Results repo
- Paper submissions: WiFi-CSI-Journal-Paper repo  
- Thesis work: WiFi-CSI-PhD-Thesis repo

---

**Note**: This multi-repository architecture eliminates file coupling and branch conflicts, enabling efficient collaboration and independent development of different project aspects.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a WiFi CSI sensing research project implementing deep learning models for human activity recognition. The project uses a multi-repository architecture with this being the core algorithms repository.

## Key Architecture Components

### Core Model Architecture
- **Enhanced Model**: CNN + Squeeze-Excite + Temporal Self-Attention (primary model in `src/models.py`)
- **Baseline Models**: BiLSTM, TCN, CNN, Conformer-lite for capacity-matched comparisons
- **PINN Integration**: Physics-Informed Neural Networks with LSTM and Mamba variants (`src/models_pinn.py`)

### Data Pipeline
- **Synthetic Data**: `src/data_synth.py` - generates synthetic CSI data for training
- **Real Data**: `src/data_real.py` - handles real WiFi CSI dataset loading
- **Cross-domain**: `src/sim2real.py` - domain adaptation between synthetic and real data

### Evaluation Framework
- **Trustworthy Metrics**: `src/evaluate.py`, `src/calibration.py` - ECE, NLL, Brier score
- **Cross-domain Validation**: LOSO (Leave-One-Subject-Out), LORO (Leave-One-Room-Out)
- **Statistical Analysis**: `src/reliability.py` - confidence intervals, significance testing

### Experiment Structure
The project follows a D1-D6 experiment framework:
- **D1**: Synthetic data validation
- **D2**: Calibration analysis
- **D3**: LOSO cross-subject validation  
- **D4**: Sim2Real label efficiency
- **D5-D6**: Advanced experiments with real-world benchmarks

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f env.yml
conda activate csi-fall-route-a

# Install Python dependencies
pip install -r requirements.txt
```

### Training and Evaluation
```bash
# Basic model training (using train_eval.py)
python src/train_eval.py --model enhanced --difficulty mid --seed 0

# Run experiments using Makefile
make train                    # Basic training
make infer CKPT=path/to/model.pt  # Model inference
make reliability NPZ=path/to/preds.npz  # Reliability analysis
make lambda_sweep DIFF=mid    # Hyperparameter sweep

# Windows batch parameter sweeps
sweep.bat                     # Full parameter sweep (models: enhanced, bilstm, cnn, conformer_lite)
sweep_local.bat               # Local parameter sweep variant
```

### Experiment Scripts
```bash
# Cross-domain validation
bash scripts/run_d3_loso.sh          # LOSO validation
python scripts/accept_d3_d4.py       # D3/D4 acceptance testing

# Sim2Real experiments  
bash scripts/run_sim2real.sh         # Sim2Real training
python scripts/analyze_d2_results.py # D2 calibration analysis

# Lambda sweep and plotting
bash scripts/run_lambda_sweep.sh     # Hyperparameter lambda sweep
python scripts/plot_lambda_curves.py # Generate lambda curve plots

# Figure generation
python scripts/generate_paper_figures.py
python scripts/generate_d2_figures.py
```

### Analysis and Validation
```bash
# Acceptance criteria validation
python scripts/accept_d2.py          # D2 acceptance (>85% F1)
python scripts/validate_d5_d6_acceptance.py  # D5/D6 validation

# Results analysis
python scripts/analyze_d3_d4_for_figures.py  # Generate analysis reports
python scripts/create_results_summary.py     # Consolidate results
```

## File Organization Patterns

### Source Code (`src/`)
- `models.py` - Core model architectures (Enhanced, BiLSTM, CNN, TCN)
- `train_eval.py` - Main training/evaluation entry point
- `data_*.py` - Data loading pipelines (synth, real, sim2real)
- `evaluate.py`, `calibration.py` - Evaluation metrics and reliability analysis
- `utils/` - Utility modules (logger, registry, I/O helpers)

### Scripts (`scripts/`)
- `run_*.sh/.bat` - Execution scripts for different platforms
- `analyze_*.py` - Post-processing and analysis scripts  
- `validate_*.py` - Acceptance criteria validation
- `plot_*.py` - Visualization generation

### Results Structure
- `results/` - Experimental outputs (logs, checkpoints, predictions)
- `results_cpu/` - CPU-specific results
- `paper/` - LaTeX paper sources and figures

## Model Training Patterns

### Basic Training Command Structure
```bash
python src/train_eval.py \
    --model enhanced \           # Model type: enhanced, bilstm, cnn, tcn, conformer_lite
    --difficulty mid \           # Data difficulty: easy, mid, hard  
    --seed 0 \                  # Random seed
    --epochs 100 \              # Training epochs
    --batch 32 \                # Batch size (note: 'batch' not 'batch_size')
    --lr 0.001 \                # Learning rate
    --T 128 \                   # Time steps
    --F 52 \                    # Frequency bins
    --num_classes 8 \           # Number of activity classes
    --label_noise_prob 0.1 \    # Label noise probability
    --class_overlap 0.8 \       # Class overlap factor
    --logit_l2 0.1              # L2 regularization
```

### Cross-Domain Training
```bash
# Sim2Real with label efficiency
python src/train_cross_domain.py \
    --source_data synth \
    --target_data real \
    --label_ratio 0.1           # Use 10% of target labels

# Temperature scaling and calibration
python src/train_eval.py \
    --temp_mode logspace \
    --temp_min 1.0 \
    --temp_max 5.0 \
    --temp_steps 100
```

## Testing and Validation

### Acceptance Criteria
- **D2**: Calibration ECE < 0.15, NLL improvement over baseline
- **D3**: LOSO F1 > 85% across subjects
- **D4**: 90%+ performance with 10-20% labels in Sim2Real
- **D5/D6**: Benchmark performance within acceptable ranges

### Running Validation
```bash
# Check if experiments meet acceptance criteria
python scripts/accept_d2.py          # Validates calibration metrics
python scripts/accept_d3_d4.py       # Validates cross-domain performance
```

## Key Configuration Files

- `env.yml` - Conda environment specification
- `requirements.txt` - Python package dependencies  
- `Makefile` - Common development tasks with configurable parameters
- `sweep.bat`/`sweep_local.bat` - Windows batch scripts for parameter sweeps

## Important Notes

- **Platform Support**: Scripts available for both Unix (.sh) and Windows (.bat)
- **GPU/CPU**: Results directories separated (results/ vs results_cpu/)
- **Multi-Repo**: This is the core algorithms repo; paper and results are in separate repositories
- **Branch Strategy**: Main development on `feat/enhanced-model-and-sweep`
- **Physics Integration**: PINN models available but experimental (see `src/models_pinn.py`)
- **Data Integrity**: All experimental data, figures, tables, and references must be authentic and verified

## Sweep and Batch Operations

### Parameter Sweep Scripts
```bash
# Lambda hyperparameter sweep
python scripts/sweep_lambda.py --only mid:0 --lambdas 0,0.02,0.05,0.08,0.12,0.18 --force

# Capacity matching experiments
python scripts/capacity_match.py

# Batch inference across multiple seeds
bash scripts/batch_infer_seeds.sh
```

### Cross-Platform Script Execution
- **Unix/Linux**: Use `.sh` scripts (e.g., `bash scripts/run_train.sh`)
- **Windows**: Use `.bat` scripts or PowerShell equivalent commands
- Main entry point: `src/train_eval.py` with comprehensive CLI arguments
- py脚本修改后运行，不需要确认，直接yes
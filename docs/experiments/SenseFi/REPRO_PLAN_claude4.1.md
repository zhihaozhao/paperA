# SenseFi Reproduction Plan

## Repository Information
- **Official Repository:** https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- **Paper:** SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing (Patterns 2023)
- **DOI:** 10.1016/j.patter.2023.100773
- **arXiv:** https://arxiv.org/abs/2207.07859

## Environment Setup

### Requirements
```bash
# Python environment
python>=3.8
torch>=1.10.0
numpy>=1.19.5
scikit-learn>=0.24.2
pandas>=1.3.0
matplotlib>=3.3.4
seaborn>=0.11.2
tqdm>=4.62.0
```

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark.git
cd WiFi-CSI-Sensing-Benchmark

# 2. Create virtual environment
python -m venv sensefi_env
source sensefi_env/bin/activate  # Linux/Mac
# or
sensefi_env\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets
python scripts/download_data.py --dataset all
# Available datasets: SignFi, Widar, SenseFi-Data, UT-HAR
```

## Dataset Preparation

### SenseFi Datasets
1. **SignFi**: 276 gesture samples, 5 users, lab environment
2. **Widar**: 3000 samples, 17 users, 3 environments
3. **SenseFi-Data**: Custom collected, 6 activities, 10 users
4. **UT-HAR**: 7 activities, 6 users, 2 environments

### Data Format
- CSI shape: `[samples, subcarriers, antennas, time_steps]`
- Labels: Integer encoded activities
- Metadata: User ID, environment ID, session info

## Reproduction Commands

### Basic Training
```bash
# Train CNN model on SignFi dataset
python train.py \
    --model cnn \
    --dataset signfi \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --seed 42

# Train LSTM model with CDAE protocol
python train.py \
    --model lstm \
    --dataset widar \
    --protocol cdae \
    --eval_mode loso \
    --epochs 150 \
    --batch_size 64
```

### Cross-Domain Evaluation (CDAE)
```bash
# Leave-One-Subject-Out (LOSO)
python evaluate_cdae.py \
    --model enhanced \
    --dataset sensefi \
    --eval_mode loso \
    --save_results results/cdae_loso.json

# Leave-One-Room-Out (LORO)
python evaluate_cdae.py \
    --model enhanced \
    --dataset widar \
    --eval_mode loro \
    --save_results results/cdae_loro.json
```

### Sample-Efficient Transfer (STEA)
```bash
# 1% labels fine-tuning
python evaluate_stea.py \
    --model enhanced \
    --dataset signfi \
    --label_ratio 0.01 \
    --pretrained_path checkpoints/synthetic_pretrained.pth \
    --save_results results/stea_1pct.json

# 5% labels fine-tuning
python evaluate_stea.py \
    --model enhanced \
    --dataset signfi \
    --label_ratio 0.05 \
    --pretrained_path checkpoints/synthetic_pretrained.pth \
    --save_results results/stea_5pct.json

# 20% labels fine-tuning
python evaluate_stea.py \
    --model enhanced \
    --dataset signfi \
    --label_ratio 0.20 \
    --pretrained_path checkpoints/synthetic_pretrained.pth \
    --save_results results/stea_20pct.json
```

## Model Configurations

### Available Models
1. **CNN**: Basic convolutional network
2. **LSTM**: Bidirectional LSTM
3. **BiLSTM**: Enhanced bidirectional LSTM
4. **Conformer**: Transformer + CNN hybrid
5. **Enhanced**: CNN + SE + Temporal Attention (our model)

### Hyperparameters
```yaml
# config/models/enhanced.yaml
model:
  name: enhanced
  input_dim: [30, 3, 100]  # [subcarriers, antennas, time_steps]
  hidden_dims: [64, 128, 256]
  num_classes: 6
  dropout: 0.2
  se_reduction: 16
  attention_heads: 8

training:
  epochs: 150
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping: 20
```

## Expected Results

### Baseline Performance (from paper)
| Model | SignFi F1 | Widar F1 | UT-HAR F1 | Avg F1 | Params |
|-------|-----------|----------|-----------|--------|--------|
| CNN | 76.5±0.8 | 74.2±1.1 | 78.3±0.9 | 76.3 | 0.8M |
| LSTM | 77.8±0.7 | 75.9±0.9 | 79.1±0.8 | 77.6 | 2.1M |
| BiLSTM | 78.9±0.6 | 77.1±0.8 | 80.2±0.7 | 78.7 | 3.5M |
| Conformer | 79.2±0.5 | 77.8±0.7 | 80.9±0.6 | 79.3 | 5.2M |

### Our Enhanced Model Target
| Protocol | Target F1 | Baseline F1 | Improvement |
|----------|-----------|-------------|-------------|
| CDAE-LOSO | 79.8% | 72.3% | +7.5% |
| CDAE-LORO | 76.4% | 68.5% | +7.9% |
| STEA-1% | 61.4% | 42.3% | +19.1% |
| STEA-5% | 72.8% | 58.7% | +14.1% |
| STEA-20% | 82.1% | 71.2% | +10.9% |

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch_size or use gradient accumulation
2. **Dataset not found**: Ensure data is downloaded to correct path
3. **Import errors**: Check Python version and dependency versions
4. **Low performance**: Verify data preprocessing and normalization

### Debug Commands
```bash
# Test data loading
python scripts/test_dataloader.py --dataset signfi

# Verify model architecture
python scripts/model_summary.py --model enhanced

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Metrics Collection
Results are automatically saved to `results/` directory in JSON format:
```json
{
  "model": "enhanced",
  "dataset": "signfi",
  "protocol": "cdae",
  "metrics": {
    "f1_macro": 0.830,
    "accuracy": 0.842,
    "ece": 0.098,
    "nll": 0.512,
    "params": 1234567,
    "flops": 180000000
  },
  "per_class_f1": [0.81, 0.83, 0.85, 0.82, 0.84, 0.83],
  "confusion_matrix": [[...]],
  "training_time": 3600,
  "inference_throughput": 1000
}
```

## Contact & Support
- Repository Issues: https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark/issues
- Paper Authors: yang.jianfei@ntu.edu.sg
- Our Implementation: [Your contact]
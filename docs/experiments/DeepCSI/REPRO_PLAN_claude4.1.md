# DeepCSI Reproduction Plan

## Repository Information
- **Official Repository:** https://github.com/francescamen/DeepCSI (tentative - verify)
- **Paper:** DeepCSI: Rethinking Wi-Fi Radio Fingerprinting Through MU-MIMO CSI Feedback Deep Learning
- **arXiv:** https://arxiv.org/abs/2204.07614
- **Year:** 2022
- **Authors:** Francesca Meneghello, Michele Rossi, Francesco Restuccia

## Environment Setup

### Requirements
```bash
# Python environment
python>=3.8
tensorflow>=2.6.0  # or torch>=1.9.0
numpy>=1.19.0
scikit-learn>=0.24.0
pandas>=1.3.0
matplotlib>=3.3.0
```

### Installation Steps
```bash
# 1. Clone repository (when available)
git clone https://github.com/francescamen/DeepCSI.git  # Verify URL
cd DeepCSI

# 2. Create virtual environment
python -m venv deepcsi_env
source deepcsi_env/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### MU-MIMO CSI Data
- **Format**: Multi-user MIMO CSI feedback
- **Shape**: `[batch, users, antennas, subcarriers]`
- **Task**: Device identification (98% accuracy reported)

## Reproduction Commands

### Training for Device ID
```bash
# Train DeepCSI for device identification
python train_device_id.py \
    --model deepcsi \
    --dataset mu_mimo_csi \
    --num_devices 50 \
    --epochs 200 \
    --batch_size 64
```

### Adaptation for HAR
```bash
# Adapt for human activity recognition
python adapt_for_har.py \
    --pretrained checkpoints/deepcsi_device.pth \
    --task activity_recognition \
    --num_classes 6 \
    --fine_tune_epochs 50
```

## Expected Results

### Original Task (Device ID)
| Metric | Value |
|--------|-------|
| Accuracy | 98% |
| F1 Score | 0.97 |
| Devices | 50 |

### HAR Adaptation
- Transfer learning from device ID features
- Expected F1: 70-75% (domain shift)
- Requires architectural modifications

## Notes
- Primary focus on device fingerprinting
- MU-MIMO features may help multi-person HAR
- Limited direct applicability to single-person HAR
- Useful for studying CSI feature representations
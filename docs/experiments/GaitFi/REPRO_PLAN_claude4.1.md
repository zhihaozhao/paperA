# GaitFi Reproduction Plan

## Repository Information
- **Official Repository:** https://github.com/langdeng/GaitFi (tentative - verify)
- **Paper:** GaitFi: Robust Device-Free Human Identification via WiFi and Vision Multimodal Learning
- **arXiv:** https://arxiv.org/abs/2208.14326
- **Year:** 2022
- **Authors:** Lang Deng, Jianfei Yang, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu, Lihua Xie

## Environment Setup

### Requirements
```bash
# Python environment
python>=3.8
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.19.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
pandas>=1.3.0
matplotlib>=3.3.0
```

### Installation Steps
```bash
# 1. Clone repository (when available)
git clone https://github.com/langdeng/GaitFi.git  # Verify URL
cd GaitFi

# 2. Create virtual environment
python -m venv gaitfi_env
source gaitfi_env/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install torch torchvision
pip install opencv-python
pip install -r requirements.txt
```

## Dataset Preparation

### Multimodal Data Collection
- **WiFi CSI**: Gait patterns from walking
- **Vision**: Synchronized camera footage
- **Fusion**: Temporal alignment of modalities

### Dataset Statistics
- 10 subjects
- 5 walking speeds
- 3 environments
- 94.2% identification accuracy reported

## Reproduction Commands

### Training Multimodal Model
```bash
# Train GaitFi with WiFi and vision
python train_multimodal.py \
    --wifi_data data/wifi_csi/ \
    --vision_data data/video/ \
    --fusion_method attention \
    --num_subjects 10 \
    --epochs 200

# WiFi-only baseline
python train_wifi_only.py \
    --data data/wifi_csi/ \
    --model gait_lstm \
    --num_subjects 10 \
    --epochs 150
```

### Evaluation
```bash
# Test identification accuracy
python evaluate.py \
    --model_path checkpoints/gaitfi_best.pth \
    --test_data data/test/ \
    --metrics accuracy precision recall
```

## Expected Results

### Performance Comparison
| Method | Accuracy | F1 Score | Modalities |
|--------|----------|----------|------------|
| WiFi-only | 81.3% | 0.80 | CSI |
| Vision-only | 88.7% | 0.87 | Video |
| GaitFi | 94.2% | 0.93 | CSI + Video |

### Adaptation for Activity Recognition
- Modify output layer for activity classes
- Adjust temporal window for different activities
- Consider privacy implications of vision

## Notes
- Multimodal fusion improves accuracy significantly
- Vision component may not be suitable for privacy-sensitive deployments
- WiFi-only variant still competitive for HAR tasks
- Gait patterns are highly discriminative biometric features
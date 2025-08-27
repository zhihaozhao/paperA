# CLNet Reproduction Plan

## Repository Information
- **Official Repository:** https://github.com/sijieaaa/CLNet (tentative - verify)
- **Paper:** CLNet: Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback
- **arXiv:** https://arxiv.org/abs/2102.07507
- **Year:** 2021
- **Authors:** Sijie Xu, Xiang Liu, Xiaoming Tao

## Environment Setup

### Requirements
```bash
# Python environment
python>=3.7
torch>=1.8.0
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.2.0
h5py>=2.10.0
```

### Installation Steps
```bash
# 1. Clone repository (when available)
git clone https://github.com/sijieaaa/CLNet.git  # Verify URL
cd CLNet

# 2. Create virtual environment
python -m venv clnet_env
source clnet_env/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install torch torchvision
pip install -r requirements.txt
```

## Dataset Preparation

### CSI Feedback Dataset
- **Format**: Complex-valued CSI matrices
- **Shape**: `[batch, 2, H, W]` (real and imaginary parts)
- **Preprocessing**: Normalization, complex-to-real conversion

## Reproduction Commands

### Training
```bash
# Train CLNet model
python train.py \
    --model clnet \
    --dataset csi_feedback \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

### Evaluation
```bash
# Test model performance
python test.py \
    --model_path checkpoints/clnet_best.pth \
    --dataset csi_feedback \
    --metrics nmse cosine
```

## Expected Results

### Performance Metrics
| Model | NMSE | Cosine Similarity | Params | FLOPs |
|-------|------|-------------------|--------|-------|
| CLNet | -17.3 dB | 0.95 | 1.8M | 450M |

### Adaptation for WiFi HAR
- Modify input layer for CSI dimensions
- Adjust output for activity classification
- Add temporal processing layers

## Notes
- Original work focuses on CSI feedback, needs adaptation for HAR
- Complex-valued processing may improve WiFi sensing
- Lightweight design suitable for edge deployment
# EfficientFi Reproduction Plan

## Repository Information
- **Official Repository:** https://github.com/xyanchen/EfficientFi (tentative - verify)
- **Paper:** EfficientFi: Towards Large-Scale Lightweight WiFi Sensing via CSI Compression
- **arXiv:** https://arxiv.org/abs/2204.04138
- **Year:** 2022
- **Authors:** Jianfei Yang, Xinyan Chen, Han Zou, Dazhuo Wang, Qianwen Xu, Lihua Xie

## Environment Setup

### Requirements
```bash
# Python environment
python>=3.8
torch>=1.10.0
numpy>=1.19.0
scikit-learn>=0.24.0
pandas>=1.3.0
matplotlib>=3.3.0
tensorly>=0.6.0  # For tensor decomposition
```

### Installation Steps
```bash
# 1. Clone repository (when available)
git clone https://github.com/xyanchen/EfficientFi.git  # Verify URL
cd EfficientFi

# 2. Create virtual environment
python -m venv efficientfi_env
source efficientfi_env/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install torch torchvision
pip install tensorly
pip install -r requirements.txt
```

## Dataset Preparation

### Compressed CSI Data
- **Compression Ratio**: 1784× reported
- **Method**: Tensor decomposition + learned compression
- **Format**: Compressed CSI tensors

## Reproduction Commands

### Training Compression Model
```bash
# Train EfficientFi compression
python train_compression.py \
    --model efficientfi \
    --compression_ratio 1784 \
    --dataset signfi \
    --epochs 150 \
    --batch_size 32
```

### HAR with Compressed CSI
```bash
# Train HAR on compressed data
python train_har_compressed.py \
    --compression_model checkpoints/efficientfi_comp.pth \
    --har_model lstm \
    --dataset signfi \
    --compressed_dim 16 \
    --epochs 100
```

### Evaluation
```bash
# Test compression-recognition pipeline
python evaluate.py \
    --compression_model checkpoints/efficientfi_comp.pth \
    --har_model checkpoints/har_compressed.pth \
    --test_data data/signfi_test.h5 \
    --metrics accuracy f1 compression_ratio
```

## Expected Results

### Compression Performance
| Metric | Value |
|--------|-------|
| Compression Ratio | 1784× |
| Reconstruction Error | < 5% |
| Storage Reduction | 99.94% |

### HAR Performance
| Model | Original CSI | Compressed CSI | Degradation |
|-------|-------------|----------------|-------------|
| CNN | 76.5% | 73.2% | -3.3% |
| LSTM | 77.8% | 74.5% | -3.3% |
| Enhanced | 83.0% | 79.1% | -3.9% |

## Key Features

### Compression Techniques
1. **Tensor Decomposition**: CP/Tucker decomposition
2. **Learned Quantization**: Trainable quantization levels
3. **Adaptive Compression**: Activity-aware compression rates
4. **Progressive Compression**: Multi-level compression

### Benefits
- Massive storage reduction (99.94%)
- Enables large-scale deployment
- Privacy preservation through compression
- Edge-friendly processing

## Implementation Details

```python
# Compression pipeline
def efficientfi_compress(csi_data):
    # Step 1: Tensor decomposition
    factors = tucker_decomposition(csi_data, ranks=[8, 8, 4])
    
    # Step 2: Quantization
    quantized = learned_quantization(factors, bits=4)
    
    # Step 3: Entropy coding
    compressed = entropy_encode(quantized)
    
    return compressed
```

## Notes
- Trade-off between compression and accuracy
- Suitable for bandwidth-constrained scenarios
- Can be combined with other baselines
- Orthogonal to model architecture improvements
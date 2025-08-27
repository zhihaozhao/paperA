# FewSense Reproduction Plan

## Repository Information
- **Official Repository:** https://github.com/arguslab/FewSense (tentative - verify)
- **Paper:** FewSense: Towards a Scalable and Cross-Domain Wi-Fi Sensing System Using Few-Shot Learning
- **arXiv:** https://arxiv.org/abs/2203.02014
- **Year:** 2022

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
learn2learn>=0.1.6  # For meta-learning
higher>=0.2.1  # For MAML implementation
```

### Installation Steps
```bash
# 1. Clone repository (when available)
git clone https://github.com/arguslab/FewSense.git  # Verify URL
cd FewSense

# 2. Create virtual environment
python -m venv fewsense_env
source fewsense_env/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install torch torchvision
pip install learn2learn higher
pip install -r requirements.txt

# 4. Download datasets
# FewSense uses SignFi, Widar, and Wiar datasets
python scripts/download_data.py --dataset all
```

## Dataset Preparation

### Supported Datasets
1. **SignFi**: 276 gesture instances, 5 users, 1 environment
2. **Widar**: 3000 samples, 17 users, 3 environments  
3. **Wiar**: Custom dataset for cross-domain evaluation

### Few-Shot Task Construction
```python
# Task sampling for N-way K-shot learning
# Default: 5-way 1-shot, 5-way 5-shot
task_config = {
    'n_way': 5,  # Number of classes per task
    'k_shot': 1,  # Support samples per class
    'q_query': 15,  # Query samples per class
    'num_tasks': 1000  # Total number of tasks
}
```

## Reproduction Commands

### Meta-Learning Training
```bash
# Train MAML-based few-shot model
python train_maml.py \
    --dataset signfi \
    --n_way 5 \
    --k_shot 1 \
    --inner_lr 0.01 \
    --meta_lr 0.001 \
    --num_iterations 10000 \
    --adaptation_steps 5

# Train ProtoNet baseline
python train_protonet.py \
    --dataset widar \
    --n_way 5 \
    --k_shot 5 \
    --embedding_dim 128 \
    --num_episodes 20000
```

### Cross-Domain Few-Shot Evaluation
```bash
# Source: SignFi, Target: Widar (1-shot)
python evaluate_cross_domain.py \
    --source_dataset signfi \
    --target_dataset widar \
    --model maml \
    --k_shot 1 \
    --num_test_tasks 600 \
    --checkpoint checkpoints/maml_signfi.pth

# Source: Widar, Target: Wiar (5-shot)
python evaluate_cross_domain.py \
    --source_dataset widar \
    --target_dataset wiar \
    --model maml \
    --k_shot 5 \
    --num_test_tasks 600 \
    --checkpoint checkpoints/maml_widar.pth
```

### Few-Shot Transfer Learning
```bash
# 1% labels (approximately 1-shot per class)
python few_shot_transfer.py \
    --pretrained_model checkpoints/maml_base.pth \
    --target_dataset signfi \
    --label_ratio 0.01 \
    --adaptation_steps 10 \
    --save_results results/fewsense_1pct.json

# 5% labels  
python few_shot_transfer.py \
    --pretrained_model checkpoints/maml_base.pth \
    --target_dataset signfi \
    --label_ratio 0.05 \
    --adaptation_steps 20 \
    --save_results results/fewsense_5pct.json

# 20% labels
python few_shot_transfer.py \
    --pretrained_model checkpoints/maml_base.pth \
    --target_dataset signfi \
    --label_ratio 0.20 \
    --adaptation_steps 50 \
    --save_results results/fewsense_20pct.json
```

## Model Configurations

### MAML Configuration
```yaml
# config/maml.yaml
model:
  name: maml_cnn
  input_shape: [30, 3, 100]  # CSI dimensions
  hidden_dims: [32, 64, 128]
  output_dim: 128  # Embedding dimension

meta_learning:
  inner_lr: 0.01
  meta_lr: 0.001
  adaptation_steps: 5
  first_order: false  # Use second-order gradients

task:
  n_way: 5
  k_shot: 1
  q_query: 15
  num_tasks_train: 10000
  num_tasks_test: 600
```

### ProtoNet Configuration
```yaml
# config/protonet.yaml
model:
  name: protonet_cnn
  input_shape: [30, 3, 100]
  embedding_dim: 128
  hidden_dims: [64, 128, 256]

training:
  num_episodes: 20000
  learning_rate: 0.001
  weight_decay: 0.0001
```

## Expected Results

### Few-Shot Learning Performance (from paper)
| Method | SignFi→Widar | Widar→Wiar | Avg 1-shot | Avg 5-shot |
|--------|--------------|------------|------------|------------|
| ProtoNet | 45.2% | 42.8% | 44.0% | 58.3% |
| MAML | 48.9% | 46.3% | 47.6% | 62.1% |
| FewSense | 52.4% | 49.8% | 51.1% | 65.7% |

### Comparison with Our Method
| Label Ratio | FewSense | Our STEA | Improvement |
|-------------|----------|----------|-------------|
| 1% | 48.5% | 61.4% | +12.9% |
| 5% | 63.2% | 72.8% | +9.6% |
| 20% | 73.8% | 82.1% | +8.3% |

## Implementation Details

### Data Augmentation
```python
# FewSense-specific augmentations
augmentations = {
    'time_shift': 0.1,  # Random time shifting
    'amplitude_scale': 0.2,  # Amplitude scaling
    'phase_shift': np.pi/6,  # Phase perturbation
    'noise_level': 0.05  # Additive Gaussian noise
}
```

### Meta-Learning Algorithm
```python
# Simplified MAML inner loop
def maml_inner_loop(model, support_x, support_y, inner_lr):
    # Clone model for task-specific adaptation
    task_model = copy.deepcopy(model)
    
    # Inner loop optimization
    for _ in range(adaptation_steps):
        loss = cross_entropy(task_model(support_x), support_y)
        grads = torch.autograd.grad(loss, task_model.parameters())
        
        # Update task-specific parameters
        for param, grad in zip(task_model.parameters(), grads):
            param.data -= inner_lr * grad
    
    return task_model
```

## Troubleshooting

### Common Issues
1. **learn2learn import error**: Install with `pip install learn2learn`
2. **CUDA memory issues**: Reduce batch size or use gradient checkpointing
3. **Slow meta-learning**: Use first-order approximation (FOMAML)
4. **Poor few-shot performance**: Increase inner loop steps

### Debug Commands
```bash
# Test few-shot task sampling
python scripts/test_task_sampling.py --n_way 5 --k_shot 1

# Verify MAML implementation
python scripts/test_maml.py --debug

# Check dataset statistics
python scripts/dataset_stats.py --dataset signfi
```

## Metrics Collection
```json
{
  "method": "fewsense",
  "source_dataset": "signfi",
  "target_dataset": "widar",
  "n_way": 5,
  "k_shot": 1,
  "metrics": {
    "accuracy": 0.524,
    "f1_macro": 0.511,
    "confidence_interval": 0.023,
    "adaptation_steps": 5,
    "meta_lr": 0.001
  },
  "per_task_accuracy": [0.52, 0.48, 0.55, ...],
  "training_time": 7200,
  "inference_time_per_task": 0.15
}
```

## Notes & Limitations
- Repository URL needs verification (check with authors)
- Wiar dataset may not be publicly available
- Meta-learning requires significant compute for training
- Results sensitive to task sampling strategy

## Contact & Support
- Paper Authors: kun.wang@[institution].edu (verify)
- Implementation Issues: [Create GitHub issue when repo available]
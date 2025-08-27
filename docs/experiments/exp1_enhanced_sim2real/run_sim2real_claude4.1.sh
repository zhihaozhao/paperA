#!/bin/bash
# Exp1: Enhanced Model Sim2Real Training Script
# Author: Claude 4.1
# Date: December 2024

set -e

# Configuration
EXPERIMENT_NAME="exp1_enhanced_sim2real"
DATA_DIR="data"
RESULTS_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LOG_FILE="${RESULTS_DIR}/training.log"

# Create directories
mkdir -p ${RESULTS_DIR}
mkdir -p ${CHECKPOINT_DIR}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${LOG_FILE}
}

# Phase 1: Pretrain on Simulation Data
run_phase1_simulation() {
    log "========================================="
    log "Phase 1: Pretraining on Simulation Data"
    log "========================================="
    
    python3 << 'EOF'
import sys
sys.path.append('docs/experiments/exp1_enhanced_sim2real')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model_claude4 import EnhancedSim2RealModel, Sim2RealTrainer
import json

# Configuration
config = {
    "batch_size": 32,
    "sim_epochs": 50,
    "learning_rate": 0.001,
    "num_classes": 6,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"Using device: {config['device']}")

# Generate synthetic simulation data
def generate_sim_data(num_samples=1000):
    """Generate synthetic CSI data for simulation"""
    np.random.seed(42)
    
    # CSI data: [samples, channels, freq, time]
    data = np.random.randn(num_samples, 3, 114, 500).astype(np.float32)
    
    # Add activity-specific patterns
    labels = np.random.randint(0, config["num_classes"], num_samples)
    for i in range(num_samples):
        activity = labels[i]
        # Add activity-specific frequency pattern
        data[i, :, activity*10:(activity+1)*10, :] += 2.0
        # Add temporal pattern
        data[i, :, :, ::activity+1] += 1.5
    
    return data, labels

# Create datasets
print("Generating simulation data...")
train_data, train_labels = generate_sim_data(1000)
val_data, val_labels = generate_sim_data(200)

# Create data loaders
train_dataset = TensorDataset(
    torch.from_numpy(train_data),
    torch.from_numpy(train_labels)
)
val_dataset = TensorDataset(
    torch.from_numpy(val_data),
    torch.from_numpy(val_labels)
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Create model
print("Creating Enhanced Sim2Real model...")
model = EnhancedSim2RealModel(
    input_shape=(3, 114, 500),
    num_classes=config["num_classes"],
    use_domain_adaptation=True
)

# Create trainer
trainer = Sim2RealTrainer(model, device=config["device"])

# Phase 1: Pretrain on simulation
print("\nStarting Phase 1: Simulation Pretraining...")
trainer.pretrain_on_simulation(train_loader, val_loader, num_epochs=config["sim_epochs"])

# Save pretrained model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': trainer.sim_optimizer.state_dict(),
    'history': trainer.history,
    'config': config
}, 'checkpoints/exp1_enhanced_sim2real/pretrained_sim.pth')

print("Phase 1 completed. Model saved.")

# Save training history
with open('results/exp1_enhanced_sim2real/phase1_history.json', 'w') as f:
    json.dump(trainer.history, f, indent=2)

# Print final results
final_acc = trainer.history['sim_val']['acc'][-1] if trainer.history['sim_val']['acc'] else 0
print(f"\nFinal Simulation Validation Accuracy: {final_acc:.2f}%")
EOF
    
    log "Phase 1 completed successfully"
}

# Phase 2: Few-shot Adaptation with Real Data
run_phase2_adaptation() {
    log "========================================="
    log "Phase 2: Few-shot Real Data Adaptation"
    log "========================================="
    
    python3 << 'EOF'
import sys
sys.path.append('docs/experiments/exp1_enhanced_sim2real')
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from model_claude4 import EnhancedSim2RealModel, Sim2RealTrainer
import json

# Configuration
config = {
    "batch_size": 16,
    "adapt_epochs": 20,
    "learning_rate": 0.0001,
    "few_shot_samples": 50,  # Few samples per class
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"Using device: {config['device']}")

# Generate synthetic real data (simulating real-world data)
def generate_real_data(num_samples=500):
    """Generate synthetic 'real' CSI data with domain shift"""
    np.random.seed(123)
    
    # CSI data with different noise characteristics
    data = np.random.randn(num_samples, 3, 114, 500).astype(np.float32) * 1.2
    
    # Add domain shift: different noise, slight pattern changes
    labels = np.random.randint(0, 6, num_samples)
    for i in range(num_samples):
        activity = labels[i]
        # Different pattern strength (domain shift)
        data[i, :, activity*10:(activity+1)*10, :] += 1.5 + np.random.randn() * 0.3
        # Add realistic noise
        data[i] += np.random.randn(3, 114, 500) * 0.1
        # Add environmental effects
        data[i, :, :, :] *= (0.8 + np.random.rand() * 0.4)
    
    return data, labels

# Load pretrained model
print("Loading pretrained model...")
checkpoint = torch.load('checkpoints/exp1_enhanced_sim2real/pretrained_sim.pth', 
                       map_location=config["device"])

model = EnhancedSim2RealModel(
    input_shape=(3, 114, 500),
    num_classes=6,
    use_domain_adaptation=True
)
model.load_state_dict(checkpoint['model_state_dict'])

# Create trainer
trainer = Sim2RealTrainer(model, device=config["device"])

# Generate real data
print("Generating 'real' data with domain shift...")
real_train_data, real_train_labels = generate_real_data(300)
real_val_data, real_val_labels = generate_real_data(100)

# Create few-shot subset
print(f"Creating few-shot dataset with {config['few_shot_samples']} samples...")
indices = np.random.choice(len(real_train_data), config['few_shot_samples'], replace=False)
few_shot_data = real_train_data[indices]
few_shot_labels = real_train_labels[indices]

# Create data loaders
few_shot_dataset = TensorDataset(
    torch.from_numpy(few_shot_data),
    torch.from_numpy(few_shot_labels)
)
val_dataset = TensorDataset(
    torch.from_numpy(real_val_data),
    torch.from_numpy(real_val_labels)
)

few_shot_loader = DataLoader(few_shot_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Phase 2: Adapt with few-shot real data
print("\nStarting Phase 2: Few-shot Adaptation...")
trainer.adapt_with_real_data(
    few_shot_loader, 
    val_loader, 
    num_epochs=config["adapt_epochs"],
    few_shot=True
)

# Save adapted model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': trainer.adapt_optimizer.state_dict(),
    'history': trainer.history,
    'config': config
}, 'checkpoints/exp1_enhanced_sim2real/adapted_real.pth')

print("Phase 2 completed. Adapted model saved.")

# Save adaptation history
with open('results/exp1_enhanced_sim2real/phase2_history.json', 'w') as f:
    json.dump(trainer.history, f, indent=2)

# Print adaptation results
print("\n" + "="*50)
print("Sim2Real Transfer Results:")
print("="*50)
if checkpoint['history']['sim_val']['acc']:
    print(f"Simulation Val Acc: {checkpoint['history']['sim_val']['acc'][-1]:.2f}%")
if trainer.history['real_val']['acc']:
    print(f"Real Val Acc (after adaptation): {trainer.history['real_val']['acc'][-1]:.2f}%")
    print(f"Adaptation Improvement: {trainer.history['real_val']['acc'][-1] - trainer.history['real_val']['acc'][0]:.2f}%")
EOF
    
    log "Phase 2 completed successfully"
}

# Phase 3: Evaluation and Analysis
run_phase3_evaluation() {
    log "========================================="
    log "Phase 3: Evaluation and Analysis"
    log "========================================="
    
    python3 << 'EOF'
import sys
sys.path.append('docs/experiments/exp1_enhanced_sim2real')
import torch
import numpy as np
from model_claude4 import EnhancedSim2RealModel
import json
import matplotlib.pyplot as plt

# Load models
print("Loading models for comparison...")

# Load pretrained (sim only) model
sim_checkpoint = torch.load('checkpoints/exp1_enhanced_sim2real/pretrained_sim.pth',
                           map_location='cpu')

# Load adapted model
adapted_checkpoint = torch.load('checkpoints/exp1_enhanced_sim2real/adapted_real.pth',
                               map_location='cpu')

# Analysis
results = {
    "simulation_performance": {
        "final_train_acc": sim_checkpoint['history']['sim_train']['acc'][-1] if sim_checkpoint['history']['sim_train']['acc'] else 0,
        "final_val_acc": sim_checkpoint['history']['sim_val']['acc'][-1] if sim_checkpoint['history']['sim_val']['acc'] else 0,
        "total_epochs": len(sim_checkpoint['history']['sim_train']['acc'])
    },
    "adaptation_performance": {
        "initial_real_acc": adapted_checkpoint['history']['real_val']['acc'][0] if adapted_checkpoint['history']['real_val']['acc'] else 0,
        "final_real_acc": adapted_checkpoint['history']['real_val']['acc'][-1] if adapted_checkpoint['history']['real_val']['acc'] else 0,
        "improvement": 0,
        "adaptation_epochs": len(adapted_checkpoint['history']['real_train']['acc'])
    },
    "model_statistics": {
        "total_parameters": sum(p.numel() for p in EnhancedSim2RealModel().parameters()),
        "domain_adapter_params": sum(p.numel() for name, p in EnhancedSim2RealModel().named_parameters() if 'domain' in name)
    },
    "sim2real_gap": {
        "before_adaptation": 0,
        "after_adaptation": 0,
        "gap_reduction": 0
    }
}

# Calculate improvements
if adapted_checkpoint['history']['real_val']['acc']:
    results["adaptation_performance"]["improvement"] = \
        results["adaptation_performance"]["final_real_acc"] - \
        results["adaptation_performance"]["initial_real_acc"]

# Calculate sim2real gap
results["sim2real_gap"]["before_adaptation"] = \
    results["simulation_performance"]["final_val_acc"] - \
    results["adaptation_performance"]["initial_real_acc"]
    
results["sim2real_gap"]["after_adaptation"] = \
    results["simulation_performance"]["final_val_acc"] - \
    results["adaptation_performance"]["final_real_acc"]
    
results["sim2real_gap"]["gap_reduction"] = \
    results["sim2real_gap"]["before_adaptation"] - \
    results["sim2real_gap"]["after_adaptation"]

# Save results
with open('results/exp1_enhanced_sim2real/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*60)
print("EXP1: Enhanced Sim2Real Model - Evaluation Summary")
print("="*60)
print(f"Simulation Performance:")
print(f"  Final Validation Accuracy: {results['simulation_performance']['final_val_acc']:.2f}%")
print(f"\nReal Data Adaptation:")
print(f"  Initial Accuracy: {results['adaptation_performance']['initial_real_acc']:.2f}%")
print(f"  Final Accuracy: {results['adaptation_performance']['final_real_acc']:.2f}%")
print(f"  Improvement: +{results['adaptation_performance']['improvement']:.2f}%")
print(f"\nSim2Real Gap:")
print(f"  Before Adaptation: {results['sim2real_gap']['before_adaptation']:.2f}%")
print(f"  After Adaptation: {results['sim2real_gap']['after_adaptation']:.2f}%")
print(f"  Gap Reduction: {results['sim2real_gap']['gap_reduction']:.2f}%")
print(f"\nModel Complexity:")
print(f"  Total Parameters: {results['model_statistics']['total_parameters']:,}")
print(f"  Domain Adapter Parameters: {results['model_statistics']['domain_adapter_params']:,}")
print("="*60)
EOF
    
    log "Phase 3 evaluation completed"
}

# Main execution
main() {
    log "Starting Exp1: Enhanced Sim2Real Training Pipeline"
    log "=================================================="
    
    # Phase 1: Simulation pretraining
    run_phase1_simulation
    
    # Phase 2: Real data adaptation
    run_phase2_adaptation
    
    # Phase 3: Evaluation
    run_phase3_evaluation
    
    log "=================================================="
    log "Exp1 Enhanced Sim2Real Pipeline Completed Successfully!"
    log "Results saved to: ${RESULTS_DIR}"
    log "Checkpoints saved to: ${CHECKPOINT_DIR}"
}

# Run if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
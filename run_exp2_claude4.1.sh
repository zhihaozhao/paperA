#!/bin/bash
# Run Exp2: Mamba State-Space Model

echo "Starting Experiment 2: Mamba State-Space Model"
echo "==============================================="

# Default parameters
DATASET=${1:-"ntu-fi-har"}
EPOCHS=${2:-50}
BATCH_SIZE=${3:-32}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"

# Run experiment
python3 docs/experiments/main_experiment_claude4.1.py \
    --experiment exp2 \
    --dataset $DATASET \
    --data_path ./Data \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 0.001 \
    --evaluate_cdae \
    --evaluate_stea \
    --save_dir ./experiments/exp2

echo "Experiment 2 completed!"

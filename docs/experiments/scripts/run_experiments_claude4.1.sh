#!/bin/bash
# Comprehensive Experiment Running Script

# Environment Setup
source /workspace/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace:$PYTHONPATH

# Paths
DATA_DIR="/workspace/data"
CHECKPOINT_DIR="/workspace/checkpoints"
RESULTS_DIR="/workspace/results"
LOG_DIR="/workspace/logs"

# Create directories
mkdir -p $CHECKPOINT_DIR $RESULTS_DIR $LOG_DIR

# Experiment Configuration
SEED=42
EPOCHS=150
BATCH_SIZE=32
LR=0.001

echo "Starting experiments at $(date)"

# ============== Baseline Experiments ==============

# CNN Baseline
echo "Running CNN baseline..."
python train.py \
    --model cnn \
    --dataset sensefi \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --save_dir $CHECKPOINT_DIR/cnn \
    --log_dir $LOG_DIR/cnn \
    2>&1 | tee $LOG_DIR/cnn.log

# LSTM Baseline
echo "Running LSTM baseline..."
python train.py \
    --model lstm \
    --dataset sensefi \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --save_dir $CHECKPOINT_DIR/lstm \
    --log_dir $LOG_DIR/lstm \
    2>&1 | tee $LOG_DIR/lstm.log

# Enhanced Baseline
echo "Running Enhanced baseline..."
python train.py \
    --model enhanced \
    --dataset sensefi \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --save_dir $CHECKPOINT_DIR/enhanced \
    --log_dir $LOG_DIR/enhanced \
    2>&1 | tee $LOG_DIR/enhanced.log

# ============== Our Methods ==============

# Exp1: Physics-Informed Multi-Scale LSTM
echo "Running Exp1 (Physics-Informed)..."
python train_physics_informed.py \
    --model multiscale_lstm_pinn \
    --dataset sensefi \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --physics_weight 0.1 \
    --seed $SEED \
    --save_dir $CHECKPOINT_DIR/exp1 \
    --log_dir $LOG_DIR/exp1 \
    2>&1 | tee $LOG_DIR/exp1.log

# Exp2: Mamba SSM
echo "Running Exp2 (Mamba)..."
python train_mamba.py \
    --model mamba_csi \
    --dataset sensefi \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model 256 \
    --d_state 16 \
    --seed $SEED \
    --save_dir $CHECKPOINT_DIR/exp2 \
    --log_dir $LOG_DIR/exp2 \
    2>&1 | tee $LOG_DIR/exp2.log

# ============== Evaluation Protocols ==============

# CDAE - Leave One Subject Out
echo "Running LOSO evaluation..."
python evaluate_cdae.py \
    --protocol loso \
    --models cnn lstm enhanced exp1 exp2 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --results_dir $RESULTS_DIR \
    2>&1 | tee $LOG_DIR/loso.log

# CDAE - Leave One Room Out
echo "Running LORO evaluation..."
python evaluate_cdae.py \
    --protocol loro \
    --models cnn lstm enhanced exp1 exp2 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --results_dir $RESULTS_DIR \
    2>&1 | tee $LOG_DIR/loro.log

# STEA - Few Shot Learning
for ratio in 0.01 0.05 0.20; do
    echo "Running STEA with $ratio labels..."
    python evaluate_stea.py \
        --label_ratio $ratio \
        --models cnn lstm enhanced exp1 exp2 \
        --checkpoint_dir $CHECKPOINT_DIR \
        --results_dir $RESULTS_DIR \
        2>&1 | tee $LOG_DIR/stea_${ratio}.log
done

# ============== Ablation Studies ==============

echo "Running ablation studies..."
python run_ablations.py \
    --model exp1 \
    --checkpoint $CHECKPOINT_DIR/exp1/best.pth \
    --results_dir $RESULTS_DIR/ablations \
    2>&1 | tee $LOG_DIR/ablations.log

# ============== Computational Efficiency ==============

echo "Measuring inference efficiency..."
python measure_efficiency.py \
    --models cnn lstm enhanced exp1 exp2 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --device cuda \
    --batch_sizes 1 8 32 128 \
    --sequence_lengths 100 500 1000 5000 \
    --results_dir $RESULTS_DIR/efficiency \
    2>&1 | tee $LOG_DIR/efficiency.log

# ============== Generate Results Tables ==============

echo "Generating result tables and plots..."
python generate_results.py \
    --results_dir $RESULTS_DIR \
    --output_dir $RESULTS_DIR/tables \
    --format latex markdown \
    2>&1 | tee $LOG_DIR/results.log

echo "Experiments completed at $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOG_DIR"
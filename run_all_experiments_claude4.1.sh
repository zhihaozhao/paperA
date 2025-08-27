#!/bin/bash
# Run all experiments on all datasets

DATASETS=("ntu-fi-har" "ut-har" "widar")
EXPERIMENTS=("exp1" "exp2")

echo "Running all experiments on all datasets"
echo "======================================="

for dataset in "${DATASETS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        echo ""
        echo "Running $exp on $dataset..."
        echo "----------------------------"
        
        python3 docs/experiments/main_experiment_claude4.1.py \
            --experiment $exp \
            --dataset $dataset \
            --data_path ./Data \
            --epochs 30 \
            --batch_size 32 \
            --lr 0.001 \
            --save_dir ./experiments/$exp/$dataset
        
        echo "Completed $exp on $dataset"
        sleep 2
    done
done

echo ""
echo "All experiments completed!"
echo "Results saved in ./experiments/"

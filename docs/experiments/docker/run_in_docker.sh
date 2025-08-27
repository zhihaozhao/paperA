#!/bin/bash
# Run experiments in Docker container
# Author: Claude 4.1
# Date: December 2024

set -e

# Default values
MODEL="exp1_sim2real"
EPOCHS=20
GPU_ID=0
BATCH_SIZE=32
INTERACTIVE=false
JUPYTER=false

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run WiFi CSI HAR experiments in Docker container

OPTIONS:
    -m, --model MODEL        Model to run (exp1_sim2real, exp2_pinn_loss, exp3_pinn_lstm, exp4_mamba)
    -e, --epochs EPOCHS      Number of training epochs (default: 20)
    -b, --batch-size SIZE    Batch size (default: 32)
    -g, --gpu GPU_ID         GPU device ID (default: 0, use -1 for CPU)
    -i, --interactive        Run in interactive mode
    -j, --jupyter            Start Jupyter Lab server
    --all                    Run all experiments
    -h, --help              Show this help message

EXAMPLES:
    # Run Exp1 for 50 epochs
    $0 --model exp1_sim2real --epochs 50

    # Run Exp2 on GPU 1 with batch size 64
    $0 -m exp2_pinn_loss -g 1 -b 64

    # Start interactive session
    $0 --interactive

    # Start Jupyter Lab
    $0 --jupyter

    # Run all experiments
    $0 --all
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -j|--jupyter)
            JUPYTER=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if Docker image exists
if ! docker images | grep -q "wifi-csi-har"; then
    echo -e "${YELLOW}Docker image not found. Building...${NC}"
    bash build_docker.sh
fi

# Set GPU options
if [ "$GPU_ID" -eq -1 ]; then
    GPU_OPTIONS=""
    CUDA_VISIBLE_DEVICES=""
else
    GPU_OPTIONS="--gpus all"
    CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Base Docker run command
DOCKER_RUN="docker run ${GPU_OPTIONS} --rm \
    -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    -e PYTHONUNBUFFERED=1 \
    -v $(pwd)/..:/workspace \
    -v $(pwd)/../../data:/data \
    -v $(pwd)/../../results:/results \
    -v $(pwd)/../../checkpoints:/checkpoints"

# Run based on mode
if [ "$JUPYTER" = true ]; then
    echo -e "${GREEN}Starting Jupyter Lab server...${NC}"
    ${DOCKER_RUN} \
        -p 8888:8888 \
        -it \
        wifi-csi-har:latest \
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        
elif [ "$INTERACTIVE" = true ]; then
    echo -e "${GREEN}Starting interactive session...${NC}"
    ${DOCKER_RUN} \
        -it \
        wifi-csi-har:latest \
        bash
        
elif [ "$RUN_ALL" = true ]; then
    echo -e "${GREEN}Running all experiments...${NC}"
    
    for model in exp1_sim2real exp2_pinn_loss exp3_pinn_lstm exp4_mamba; do
        echo -e "${GREEN}Running ${model}...${NC}"
        ${DOCKER_RUN} \
            wifi-csi-har:latest \
            unified_experiment_runner_claude4.1.py \
            --model ${model} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE}
    done
    
    # Generate comparison
    echo -e "${GREEN}Generating comparison report...${NC}"
    ${DOCKER_RUN} \
        wifi-csi-har:latest \
        unified_experiment_runner_claude4.1.py \
        --model exp1_sim2real \
        --compare
        
else
    echo -e "${GREEN}Running experiment: ${MODEL}${NC}"
    echo "Configuration:"
    echo "  Model: ${MODEL}"
    echo "  Epochs: ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  GPU: ${GPU_ID}"
    
    ${DOCKER_RUN} \
        wifi-csi-har:latest \
        unified_experiment_runner_claude4.1.py \
        --model ${MODEL} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE}
fi

echo -e "${GREEN}Done!${NC}"
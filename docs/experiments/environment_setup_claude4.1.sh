#!/bin/bash
# Environment Setup Script for WiFi HAR Experiments

echo "Setting up environment for WiFi HAR experiments..."

# Python version check
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required, found $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv /workspace/.venv_exp
source /workspace/.venv_exp/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 11.8)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "Installing requirements..."
pip install -r /workspace/docs/experiments/requirements_claude4.1.txt

# Install Mamba (if available)
echo "Attempting to install Mamba SSM..."
pip install mamba-ssm || echo "Mamba SSM not available, install from source if needed"

# Setup Jupyter kernel
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name wifi_har --display-name "WiFi HAR"

# Download pre-trained models (if needed)
echo "Creating model directories..."
mkdir -p /workspace/models/pretrained
mkdir -p /workspace/data/raw
mkdir -p /workspace/data/processed
mkdir -p /workspace/results
mkdir -p /workspace/checkpoints
mkdir -p /workspace/logs

# Set environment variables
echo "Setting environment variables..."
export PYTHONPATH=/workspace:$PYTHONPATH
export DATA_DIR=/workspace/data
export MODEL_DIR=/workspace/models
export RESULTS_DIR=/workspace/results

# Save environment variables to file
cat > /workspace/.env << EOF
PYTHONPATH=/workspace:\$PYTHONPATH
DATA_DIR=/workspace/data
MODEL_DIR=/workspace/models
RESULTS_DIR=/workspace/results
CUDA_VISIBLE_DEVICES=0
EOF

echo "Environment setup complete!"
echo "To activate: source /workspace/.venv_exp/bin/activate"
echo "Environment variables saved to /workspace/.env"
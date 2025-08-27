#!/bin/bash

# Setup Python virtual environment for experiments

echo "Setting up Python virtual environment..."

# Create virtual environment
python3 -m venv venv_claude4.1

# Activate virtual environment
source venv_claude4.1/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn matplotlib seaborn tqdm einops h5py pandas
pip install wandb bibtexparser

echo "Virtual environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv_claude4.1/bin/activate"
echo ""
echo "Then you can run experiments with:"
echo "  python docs/experiments/main_experiment_claude4.1.py --help"
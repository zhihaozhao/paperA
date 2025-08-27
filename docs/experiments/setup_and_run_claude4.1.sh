#!/bin/bash

# Setup and run experiments for WiFi CSI HAR
# This script prepares the data structure and runs experiments

echo "================================================"
echo "WiFi CSI HAR Experiment Setup and Execution"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in correct directory
if [ ! -d "docs/experiments" ]; then
    echo -e "${RED}Error: Please run this script from the workspace root directory${NC}"
    exit 1
fi

# Step 1: Create data directory structure
echo -e "\n${YELLOW}Step 1: Setting up data directory structure...${NC}"
mkdir -p Data/NTU-Fi_HAR/{train_amp,test_amp}
mkdir -p Data/NTU-Fi-HumanID/{train_amp,test_amp}
mkdir -p Data/UT_HAR/{data,label}
mkdir -p Data/Widardata/{train,test}

echo -e "${GREEN}✓ Data directory structure created${NC}"
tree Data/ -d -L 2

# Step 2: Check for benchmark data
echo -e "\n${YELLOW}Step 2: Checking for benchmark data...${NC}"
if [ -d "benchmark_data_claude4.1" ]; then
    echo -e "${GREEN}✓ Benchmark repository found${NC}"
else
    echo "Cloning benchmark repository..."
    git clone https://github.com/zhihaozhao/WiFi-CSI-Sensing-Benchmark.git benchmark_data_claude4.1
fi

# Step 3: Install Python dependencies
echo -e "\n${YELLOW}Step 3: Installing Python dependencies...${NC}"
pip install -q torch torchvision numpy scipy scikit-learn matplotlib seaborn tqdm einops h5py pandas

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Create experiment directories
echo -e "\n${YELLOW}Step 4: Creating experiment output directories...${NC}"
mkdir -p experiments/{exp1,exp2}/{checkpoints,results,logs}
mkdir -p experiments/figures
mkdir -p experiments/ablation_studies

echo -e "${GREEN}✓ Experiment directories created${NC}"

# Step 5: Create quick run scripts
echo -e "\n${YELLOW}Step 5: Creating quick run scripts...${NC}"

# Create run_exp1.sh
cat > run_exp1_claude4.1.sh << 'EOF'
#!/bin/bash
# Run Exp1: Physics-Informed Multi-Scale LSTM

echo "Starting Experiment 1: Physics-Informed Multi-Scale LSTM"
echo "========================================================="

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
    --experiment exp1 \
    --dataset $DATASET \
    --data_path ./Data \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 0.001 \
    --evaluate_cdae \
    --evaluate_stea \
    --save_dir ./experiments/exp1

echo "Experiment 1 completed!"
EOF

# Create run_exp2.sh
cat > run_exp2_claude4.1.sh << 'EOF'
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
EOF

# Create run_all_experiments.sh
cat > run_all_experiments_claude4.1.sh << 'EOF'
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
EOF

chmod +x run_exp1_claude4.1.sh
chmod +x run_exp2_claude4.1.sh
chmod +x run_all_experiments_claude4.1.sh

echo -e "${GREEN}✓ Quick run scripts created${NC}"

# Step 6: Create sample data generator (for testing without real data)
echo -e "\n${YELLOW}Step 6: Creating sample data generator...${NC}"

cat > generate_sample_data_claude4.1.py << 'EOF'
"""
Generate sample data for testing experiments
"""
import numpy as np
import scipy.io as sio
from pathlib import Path
import os

def generate_ntu_fi_sample(data_dir, num_classes=6, samples_per_class=10):
    """Generate sample NTU-Fi format data"""
    print(f"Generating NTU-Fi sample data in {data_dir}")
    
    for mode in ['train_amp', 'test_amp']:
        mode_dir = Path(data_dir) / mode
        
        for class_id in range(num_classes):
            class_dir = mode_dir / f'class_{class_id}'
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for sample_id in range(samples_per_class):
                # Generate random CSI data
                # Shape: [342, 2000] (3 antennas * 114 subcarriers, 2000 time steps)
                csi_data = np.random.randn(342, 2000) * 10 + 42.3199
                
                # Save as .mat file
                mat_data = {'CSIamp': csi_data}
                sio.savemat(class_dir / f'sample_{sample_id}.mat', mat_data)
    
    print(f"  Created {num_classes} classes with {samples_per_class} samples each")

def generate_ut_har_sample(data_dir):
    """Generate sample UT-HAR format data"""
    print(f"Generating UT-HAR sample data in {data_dir}")
    
    for mode in ['train', 'val', 'test']:
        # Generate random data
        num_samples = 100 if mode == 'train' else 50
        data = np.random.randn(num_samples, 250, 90)
        labels = np.random.randint(0, 7, num_samples)
        
        # Save as .csv (actually numpy arrays)
        data_path = Path(data_dir) / 'data' / f'{mode}_data.csv'
        label_path = Path(data_dir) / 'label' / f'{mode}_label.csv'
        
        data_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(data_path, 'wb') as f:
            np.save(f, data)
        with open(label_path, 'wb') as f:
            np.save(f, labels)
    
    print(f"  Created train/val/test splits")

def generate_widar_sample(data_dir, num_classes=6, samples_per_class=10):
    """Generate sample Widar format data"""
    print(f"Generating Widar sample data in {data_dir}")
    
    for mode in ['train', 'test']:
        mode_dir = Path(data_dir) / mode
        
        for class_id in range(num_classes):
            class_dir = mode_dir / f'gesture_{class_id}'
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for sample_id in range(samples_per_class):
                # Generate random BVP data
                # Shape: [22*400] = 8800 values
                data = np.random.randn(8800) * 0.0119 + 0.0025
                
                # Save as CSV
                csv_path = class_dir / f'sample_{sample_id}.csv'
                np.savetxt(csv_path, data.reshape(-1, 1), delimiter=',')
    
    print(f"  Created {num_classes} classes with {samples_per_class} samples each")

if __name__ == "__main__":
    print("Generating sample data for testing...")
    print("=" * 50)
    
    # Generate samples for each dataset
    generate_ntu_fi_sample('Data/NTU-Fi_HAR')
    generate_ut_har_sample('Data/UT_HAR')
    generate_widar_sample('Data/Widardata')
    
    print("\nSample data generation complete!")
    print("You can now run experiments with this test data.")
EOF

python3 generate_sample_data_claude4.1.py

echo -e "${GREEN}✓ Sample data generated${NC}"

# Step 7: Display summary and instructions
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"

echo -e "\n${YELLOW}Quick Start Commands:${NC}"
echo "1. Run Exp1 on NTU-Fi HAR:"
echo "   ./run_exp1_claude4.1.sh ntu-fi-har 50 32"
echo ""
echo "2. Run Exp2 on UT-HAR:"
echo "   ./run_exp2_claude4.1.sh ut-har 50 32"
echo ""
echo "3. Run all experiments:"
echo "   ./run_all_experiments_claude4.1.sh"
echo ""
echo "4. Run with custom parameters:"
echo "   python3 docs/experiments/main_experiment_claude4.1.py --help"

echo -e "\n${YELLOW}Data Information:${NC}"
echo "- Sample data has been generated for testing"
echo "- To use real data, download from:"
echo "  https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt"
echo "- Place downloaded data in the ./Data directory"

echo -e "\n${YELLOW}Output Locations:${NC}"
echo "- Model checkpoints: ./experiments/{exp1,exp2}/checkpoints/"
echo "- Results: ./experiments/{exp1,exp2}/results/"
echo "- Figures: ./experiments/figures/"

echo -e "\n${GREEN}Ready to run experiments!${NC}"
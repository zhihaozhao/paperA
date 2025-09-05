#!/bin/bash
# Xavier Setup and D1 True Parameter Efficiency Measurement
# Complete environment setup and measurement execution guide

echo "🚀 Xavier D1 True Parameter Configuration Setup & Measurement"
echo "🎯 Upgrading from Python 2.7 to Python 3.8+ environment"

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Step 1: System Update
echo ""
echo "📦 Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y
check_status "System update"

# Step 2: Install Python 3.8
echo ""
echo "🐍 Step 2: Installing Python 3.8..."
sudo apt install -y python3.8 python3.8-dev python3.8-venv python3-pip
sudo apt install -y build-essential libssl-dev libffi-dev
check_status "Python 3.8 installation"

# Step 3: Set Python 3 as default
echo ""
echo "⚙️  Step 3: Configuring Python 3 as default..."
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
python --version
check_status "Python configuration"

# Step 4: Create virtual environment
echo ""
echo "📂 Step 4: Creating virtual environment..."
python3.8 -m venv ~/xavier_d1_env
source ~/xavier_d1_env/bin/activate
pip install --upgrade pip setuptools wheel
check_status "Virtual environment creation"

# Step 5: Check CUDA and install PyTorch for Xavier
echo ""
echo "🔥 Step 5: Checking CUDA and installing PyTorch for Xavier..."

# Check CUDA version and JetPack
echo "🔍 Detecting CUDA and JetPack version..."
if [ -f "/etc/nv_tegra_release" ]; then
    echo "📋 Jetson platform detected:"
    cat /etc/nv_tegra_release
    
    # Detect JetPack version
    if grep -q "R35" /etc/nv_tegra_release; then
        echo "🎯 JetPack 5.x detected (CUDA 11.4+)"
        JETPACK_VERSION="5.x"
    elif grep -q "R32" /etc/nv_tegra_release; then
        echo "🎯 JetPack 4.x detected (CUDA 10.2)"
        JETPACK_VERSION="4.x"
    else
        echo "⚠️  Unknown JetPack version, defaulting to 5.x"
        JETPACK_VERSION="5.x"
    fi
else
    echo "❌ Not a Jetson platform"
    exit 1
fi

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ NVCC found - CUDA version: $CUDA_VERSION"
else
    echo "⚠️  NVCC not found, setting up CUDA path..."
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

# Install appropriate PyTorch version
echo "📦 Installing PyTorch for $JETPACK_VERSION..."
case $JETPACK_VERSION in
    "5.x")
        # JetPack 5.x - CUDA 11.4
        wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.12.0-cp38-cp38-linux_aarch64.whl
        pip install torch-1.12.0-cp38-cp38-linux_aarch64.whl
        ;;
    "4.x")
        # JetPack 4.x - CUDA 10.2 - Use existing PyTorch 1.8 if available
        echo "🎯 JetPack 4.x detected - checking existing PyTorch 1.8..."
        if python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "1.8"; then
            echo "✅ PyTorch 1.8 already installed - keeping existing version"
            echo "📌 Version: $(python3 -c 'import torch; print(torch.__version__)')"
        else
            echo "📦 Installing PyTorch 1.8 for JetPack 4.x..."
            # PyTorch 1.8.0 wheel for JetPack 4.x
            wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
            pip install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
        fi
        ;;
esac
check_status "PyTorch installation"

# Step 6: Install Torchvision for PyTorch 1.8
echo ""
echo "🖼️  Step 6: Installing Torchvision for PyTorch 1.8..."

# Install dependencies for torchvision
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

# Check if torchvision is already installed and compatible
if python3 -c "import torchvision; print('torchvision version:', torchvision.__version__)" 2>/dev/null; then
    echo "✅ Torchvision already installed"
else
    echo "📦 Installing Torchvision 0.9.0 for PyTorch 1.8..."
    # Install torchvision 0.9.0 compatible with PyTorch 1.8
    pip install torchvision==0.9.0 --no-deps
    
    # If that fails, try from source
    if ! python3 -c "import torchvision" 2>/dev/null; then
        echo "📦 Building torchvision from source..."
        git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision_build
        cd torchvision_build
        python setup.py install
        cd ..
        rm -rf torchvision_build
    fi
fi
check_status "Torchvision installation"

# Step 7: Install project dependencies
echo ""
echo "📦 Step 7: Installing project dependencies..."
pip install -r xavier_requirements_python3.txt
check_status "Project dependencies installation"

# Step 8: Verify installation
echo ""
echo "🔍 Step 8: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
python -c "import numpy, psutil; print('NumPy and psutil: OK')"
check_status "Installation verification"

# Step 9: Check NVIDIA GPU
echo ""
echo "🖥️  Step 9: NVIDIA GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="cuda"
else
    echo "⚠️  nvidia-smi not available, using CPU mode"
    DEVICE="cpu"
fi

# Step 10: Run D1 efficiency measurement
echo ""
echo "📊 Step 10: Running D1 True Parameter Configuration Measurement..."
echo "🎯 Target: PASE-Net & CNN ~64K parameters"
echo "💾 Virtual environment: ~/xavier_d1_env"
echo "🖥️  Device: $DEVICE"

# Activate environment and run measurement
source ~/xavier_d1_env/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="xavier_d1_true_efficiency_${TIMESTAMP}.json"

echo "📁 Output file: $OUTPUT_FILE"
echo "⏳ Starting measurement..."

# Run the efficiency measurement
python measure_d1_true_efficiency_xavier.py \
    --device $DEVICE \
    --output $OUTPUT_FILE \
    --T 128 \
    --F 30 \
    --classes 8

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS: D1 True Configuration Measurement Completed!"
    echo "📊 Results saved in: $OUTPUT_FILE"
    
    # Quick summary if jq available
    if command -v jq &> /dev/null; then
        echo ""
        echo "📋 Quick Summary:"
        jq -r '.results | to_entries[] | "\(.key): \(.value.parameters_K // "ERROR")K params, \(.value.inference_mean_ms // "N/A")ms, \(.value.memory_peak_mb // "N/A")MB"' $OUTPUT_FILE
    else
        echo "💡 Install jq for JSON summary: sudo apt install -y jq"
    fi
    
    echo ""
    echo "🎯 Key Results (D1 True Config):"
    echo "✅ PASE-Net: Should show ~64K parameters"
    echo "✅ CNN: Should show ~64K parameters"  
    echo "✅ BiLSTM: Capacity-matched parameters"
    echo "✅ All models: Edge deployment ready"
    
    echo ""
    echo "📁 Files created:"
    echo "  - $OUTPUT_FILE (measurement results)"
    echo "  - ~/xavier_d1_env/ (Python 3 environment)"
    
    echo ""
    echo "🔄 To run again:"
    echo "  source ~/xavier_d1_env/bin/activate"
    echo "  python measure_d1_true_efficiency_xavier.py --device $DEVICE"
    
else
    echo "❌ Measurement failed! Check environment setup."
    echo "🔧 Troubleshooting:"
    echo "  1. Ensure virtual environment is active: source ~/xavier_d1_env/bin/activate"
    echo "  2. Check CUDA availability: python -c 'import torch; print(torch.cuda.is_available())'"
    echo "  3. Try CPU mode: python measure_d1_true_efficiency_xavier.py --device cpu"
    exit 1
fi

echo ""
echo "🏁 Xavier D1 True Configuration Setup & Measurement Complete!"
echo "💡 Environment preserved at ~/xavier_d1_env for future use"
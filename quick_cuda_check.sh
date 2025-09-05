#!/bin/bash
# Quick CUDA Version Check for Xavier

echo "🔍 Quick Xavier CUDA Check"
echo "========================="

# Basic system info
echo "🖥️  System: $(uname -m) - $(lsb_release -d 2>/dev/null | cut -f2 || echo 'Unknown')"

# JetPack version
if [ -f "/etc/nv_tegra_release" ]; then
    echo "📋 Jetson Release:"
    cat /etc/nv_tegra_release | head -1
    
    if grep -q "R35" /etc/nv_tegra_release; then
        echo "🎯 JetPack: 5.x (CUDA 11.4+)"
    elif grep -q "R32" /etc/nv_tegra_release; then
        echo "🎯 JetPack: 4.x (CUDA 10.2)"
    fi
else
    echo "❌ Not Jetson platform"
fi

# CUDA check
echo ""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ NVCC CUDA: $CUDA_VERSION"
else
    echo "❌ NVCC not found - try: export PATH=/usr/local/cuda/bin:\$PATH"
fi

# GPU check  
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU: $(nvidia-smi -L | head -1)"
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs -I {} echo "💾 Memory: {}MB"
else
    echo "❌ nvidia-smi not available"
fi

# Python PyTorch check
echo ""
if command -v python3 &> /dev/null; then
    python3 -c "
try:
    import torch
    version = torch.__version__
    print(f'✅ PyTorch: {version}')
    
    # Check if it's the expected 1.8 version for JetPack 4.x
    if version.startswith('1.8'):
        print('🎯 Correct PyTorch 1.8 for JetPack 4.x')
    elif version.startswith('1.12'):
        print('🎯 PyTorch 1.12 for JetPack 5.x')  
    else:
        print(f'⚠️  Unexpected PyTorch version: {version}')
    
    print(f'🔥 CUDA: {torch.cuda.is_available()} - {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
    
    if torch.cuda.is_available():
        print(f'🖥️  GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('❌ PyTorch not installed')
except Exception as e:
    print(f'⚠️  PyTorch import error: {e}')
"
else
    echo "❌ Python3 not available"
fi
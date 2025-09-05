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
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'🔥 CUDA: {torch.cuda.is_available()} - {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
except ImportError:
    print('❌ PyTorch not installed')
except:
    print('⚠️  PyTorch import error')
"
else
    echo "❌ Python3 not available"
fi
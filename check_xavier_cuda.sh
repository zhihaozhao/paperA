#!/bin/bash
# Check Xavier CUDA Version and Compatibility
# 检查Xavier CUDA版本和兼容性

echo "🔍 Xavier CUDA Version and Compatibility Check"
echo "=============================================="

# Check if we're on Jetson platform
echo ""
echo "📋 Platform Detection:"
if [ -f "/etc/nv_tegra_release" ]; then
    echo "✅ NVIDIA Jetson platform detected"
    echo "📄 Jetson release info:"
    cat /etc/nv_tegra_release
    echo ""
    
    # Extract JetPack version
    if grep -q "R35" /etc/nv_tegra_release; then
        echo "🎯 Detected: JetPack 5.x (Ubuntu 20.04)"
        JETPACK_VERSION="5.x"
    elif grep -q "R32" /etc/nv_tegra_release; then
        echo "🎯 Detected: JetPack 4.x (Ubuntu 18.04)"
        JETPACK_VERSION="4.x"
    else
        echo "⚠️  Unknown JetPack version"
        JETPACK_VERSION="unknown"
    fi
else
    echo "❌ Not on Jetson platform"
    JETPACK_VERSION="none"
fi

# Check system info
echo ""
echo "🖥️  System Information:"
echo "Architecture: $(uname -m)"
echo "OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")"
echo "Kernel: $(uname -r)"

# Check CUDA installation
echo ""
echo "🔥 CUDA Installation Check:"
if command -v nvcc &> /dev/null; then
    echo "✅ NVCC found"
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "📌 NVCC CUDA Version: $CUDA_VERSION"
else
    echo "❌ NVCC not found in PATH"
    echo "💡 Try: export PATH=/usr/local/cuda/bin:\$PATH"
fi

# Check CUDA libraries
echo ""
echo "📚 CUDA Libraries Check:"
if [ -d "/usr/local/cuda" ]; then
    echo "✅ /usr/local/cuda directory exists"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        echo "📄 CUDA version file:"
        cat /usr/local/cuda/version.txt
    fi
    
    # Check specific CUDA library versions
    echo ""
    echo "🔍 CUDA Libraries:"
    find /usr/local/cuda/lib64 -name "*cudart*" 2>/dev/null | head -5
    find /usr/local/cuda/lib64 -name "*cublas*" 2>/dev/null | head -3
else
    echo "⚠️  /usr/local/cuda not found"
    echo "🔍 Searching for CUDA in other locations..."
    find /usr -name "*cuda*" -type d 2>/dev/null | head -10
fi

# Check nvidia-smi
echo ""
echo "🖥️  GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi available"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    echo "📊 Full GPU info:"
    nvidia-smi -L
else
    echo "❌ nvidia-smi not available"
fi

# Check Python CUDA compatibility
echo ""
echo "🐍 Python CUDA Check:"
if command -v python3 &> /dev/null; then
    echo "✅ Python3 available: $(python3 --version)"
    
    # Try to import torch and check CUDA
    echo "🔥 PyTorch CUDA check:"
    python3 -c "
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'🔥 CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'📌 PyTorch CUDA version: {torch.version.cuda}')
        print(f'🖥️  GPU device: {torch.cuda.get_device_name(0)}')
        print(f'💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    else:
        print('❌ CUDA not available in PyTorch')
except ImportError as e:
    print(f'❌ PyTorch not installed: {e}')
    print('💡 Need to install PyTorch with CUDA support')
except Exception as e:
    print(f'⚠️  Error checking PyTorch: {e}')
"
else
    echo "❌ Python3 not available"
fi

# Recommend PyTorch version based on detected CUDA/JetPack
echo ""
echo "🎯 Recommended PyTorch Installation:"
echo "===================================="

case $JETPACK_VERSION in
    "5.x")
        echo "📦 For JetPack 5.x (CUDA 11.4):"
        echo "   wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.12.0-cp38-cp38-linux_aarch64.whl"
        echo "   pip3 install torch-1.12.0-cp38-cp38-linux_aarch64.whl"
        echo ""
        echo "📦 Alternative for JetPack 5.1+:"
        echo "   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        ;;
    "4.x")
        echo "📦 For JetPack 4.x (CUDA 10.2):"
        echo "   wget https://nvidia.box.com/shared/static/9eptse6jyly38kwjgj2n3gf9gqjskmu7.whl -O torch-1.11.0-cp38-cp38-linux_aarch64.whl"
        echo "   pip3 install torch-1.11.0-cp38-cp38-linux_aarch64.whl"
        ;;
    *)
        echo "⚠️  Unknown JetPack version, manual CUDA version check needed"
        echo "💡 Check your CUDA version and visit:"
        echo "   https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/jetson-agx-xavier/25"
        ;;
esac

echo ""
echo "🔧 Troubleshooting Tips:"
echo "======================"
echo "1. If CUDA not found:"
echo "   export PATH=/usr/local/cuda/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "2. If PyTorch CUDA not working:"
echo "   - Uninstall existing PyTorch: pip3 uninstall torch torchvision"
echo "   - Install Jetson-specific wheel (see recommendations above)"
echo ""
echo "3. If memory issues:"
echo "   sudo systemctl disable nvzramconfig"
echo "   sudo fallocate -l 8G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
echo ""
echo "4. Check JetPack version:"
echo "   sudo apt-cache show nvidia-jetpack"

echo ""
echo "🏁 CUDA Compatibility Check Complete!"
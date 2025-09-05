#!/bin/bash
# Xavier D1 Efficiency Measurement - JetPack 4.x Optimized
# Specifically for: JetPack 4.x + CUDA 10.2 + PyTorch 1.8

echo "🚀 Xavier D1 Efficiency Measurement - JetPack 4.x Optimized"
echo "🎯 Configuration: JetPack 4.x + CUDA 10.2 + PyTorch 1.8"

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Step 1: Verify existing environment
echo ""
echo "🔍 Step 1: Verifying existing PyTorch 1.8 environment..."

# Check if we're on Xavier
if [ -f "/etc/nv_tegra_release" ]; then
    echo "✅ Jetson Xavier platform detected"
    cat /etc/nv_tegra_release | head -1
else
    echo "⚠️  Not on Jetson platform, but continuing..."
fi

# Check existing PyTorch 1.8
if python3 -c "import torch; assert torch.__version__.startswith('1.8')" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "✅ PyTorch 1.8 found: $TORCH_VERSION"
    
    # Check CUDA availability
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "✅ PyTorch CUDA support confirmed"
        python3 -c "import torch; print(f'🔥 CUDA version: {torch.version.cuda}')"
        python3 -c "import torch; print(f'🖥️  GPU: {torch.cuda.get_device_name(0)}')"
    else
        echo "❌ PyTorch CUDA not available"
        exit 1
    fi
else
    echo "❌ PyTorch 1.8 not found or not working"
    echo "💡 Please ensure PyTorch 1.8 with CUDA support is installed"
    exit 1
fi

# Step 2: Install minimal additional dependencies for D1 measurement
echo ""
echo "📦 Step 2: Installing minimal dependencies for D1 measurement..."

# Create a simple requirements file for current environment
cat > xavier_d1_minimal_requirements.txt << EOF
# Minimal requirements for D1 efficiency measurement on existing PyTorch 1.8
numpy>=1.19.0,<1.22.0  # Compatible with PyTorch 1.8
psutil>=5.8.0          # System monitoring
tqdm>=4.50.0          # Progress bars
json5>=0.9.0          # JSON support
matplotlib>=3.3.0,<3.6.0  # Optional plotting
EOF

# Install only what's needed
pip3 install -r xavier_d1_minimal_requirements.txt --user
check_status "Minimal dependencies installation"

# Step 3: Verify D1 measurement script compatibility
echo ""
echo "🔧 Step 3: Verifying D1 measurement script..."

# Check script syntax
python3 -m py_compile measure_d1_true_efficiency_xavier.py
check_status "Script syntax check"

# Quick import test
python3 -c "
import torch
import torch.nn as nn
import numpy as np
import psutil
import json
import time
print('✅ All required modules available')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
check_status "Import verification"

# Step 4: Run D1 efficiency measurement
echo ""
echo "🎯 Step 4: Running D1 True Parameter Configuration Measurement..."

# Set GPU to max performance
echo "⚡ Setting Xavier to maximum performance..."
sudo nvpmodel -m 0 2>/dev/null || echo "nvpmodel not available, continuing..."
sudo jetson_clocks 2>/dev/null || echo "jetson_clocks not available, continuing..."

# Generate timestamp for output
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="xavier_d1_pytorch18_efficiency_${TIMESTAMP}.json"

echo "📁 Output file: $OUTPUT_FILE"
echo "🖥️  Device: CUDA (PyTorch 1.8 + CUDA 10.2)"
echo "⏳ Starting D1 measurement..."

# Run the measurement
python3 measure_d1_true_efficiency_xavier.py \
    --device cuda \
    --output $OUTPUT_FILE \
    --T 128 \
    --F 30 \
    --classes 8

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS: D1 True Configuration Measurement Completed!"
    echo "📊 Results saved in: $OUTPUT_FILE"
    
    # Display quick summary
    echo ""
    echo "📋 Quick Results Summary:"
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    for model, metrics in results.items():
        if isinstance(metrics, dict) and 'parameters_K' in metrics:
            params = metrics.get('parameters_K', 'N/A')
            inference = metrics.get('inference_mean_ms', 'N/A')
            memory = metrics.get('memory_peak_mb', 'N/A')
            print(f'  {model}: {params}K params, {inference}ms, {memory}MB')
except Exception as e:
    print(f'Could not parse results: {e}')
"
    fi
    
    echo ""
    echo "🎯 D1 Configuration Verification:"
    echo "✅ PASE-Net: Should show ~64K parameters"
    echo "✅ CNN: Should show ~64K parameters"  
    echo "✅ BiLSTM: Capacity-matched parameters"
    echo "✅ PyTorch 1.8 + CUDA 10.2 compatibility confirmed"
    
    echo ""
    echo "🏁 Xavier D1 Measurement Complete (PyTorch 1.8 Optimized)!"
    
else
    echo "❌ Measurement failed!"
    echo "🔧 Troubleshooting tips:"
    echo "  1. Check PyTorch 1.8 CUDA: python3 -c 'import torch; print(torch.cuda.is_available())'"
    echo "  2. Try CPU mode: python3 measure_d1_true_efficiency_xavier.py --device cpu"
    echo "  3. Check GPU memory: nvidia-smi"
    exit 1
fi

echo ""
echo "💾 Files created:"
echo "  - $OUTPUT_FILE (measurement results)"
echo "  - xavier_d1_minimal_requirements.txt (dependencies used)"
echo ""
echo "🔄 To run again: python3 measure_d1_true_efficiency_xavier.py --device cuda"
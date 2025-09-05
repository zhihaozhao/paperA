#!/bin/bash
# Xavier D1 Direct Measurement - Skip Dependencies Installation
# For systems where pip installation fails or dependencies are already satisfied

echo "🚀 Xavier D1 Direct Measurement (Skip Dependencies)"
echo "🎯 Running D1 measurement with existing environment"

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Step 1: Environment verification
echo ""
echo "🔍 Step 1: Verifying environment..."

# Check if we're on Xavier
if [ -f "/etc/nv_tegra_release" ]; then
    echo "✅ Jetson Xavier platform detected"
    cat /etc/nv_tegra_release | head -1
fi

# Check PyTorch availability
echo "🔍 Checking PyTorch..."
python3 -c "
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.version.cuda}')
        print(f'🖥️  GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️  CUDA not available, will use CPU')
    
    # Check other required modules
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
    
    import json
    print('✅ JSON: Available')
    
    import time, os
    print('✅ Standard libraries: Available')
    
except ImportError as e:
    print(f'❌ Missing module: {e}')
    print('💡 Please install required dependencies first')
    exit(1)
except Exception as e:
    print(f'⚠️  Environment check warning: {e}')
" || exit 1

# Step 2: Set performance mode
echo ""
echo "⚡ Step 2: Setting Xavier to maximum performance..."
sudo nvpmodel -m 0 2>/dev/null || echo "nvpmodel not available"
sudo jetson_clocks 2>/dev/null || echo "jetson_clocks not available"

# Step 3: Run measurement
echo ""
echo "🎯 Step 3: Running D1 True Parameter Configuration Measurement..."

# Check if measurement script exists
if [ ! -f "measure_d1_true_efficiency_xavier.py" ]; then
    echo "❌ measure_d1_true_efficiency_xavier.py not found"
    echo "💡 Please ensure you're in the correct directory"
    exit 1
fi

# Quick syntax check
python3 -m py_compile measure_d1_true_efficiency_xavier.py
check_status "Script syntax verification"

# Generate output filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="xavier_d1_direct_efficiency_${TIMESTAMP}.json"

echo "📁 Output file: $OUTPUT_FILE"
echo "⏳ Starting D1 measurement..."

# Determine device (prefer CUDA if available)
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "🖥️  Using CUDA device"
else
    echo "🖥️  Using CPU device"
fi

# Run the measurement
python3 measure_d1_true_efficiency_xavier.py \
    --device $DEVICE \
    --output $OUTPUT_FILE \
    --T 128 \
    --F 30 \
    --classes 8

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS: D1 Measurement Completed!"
    echo "📊 Results saved in: $OUTPUT_FILE"
    
    # Try to show results summary
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "📋 Results Preview:"
        python3 -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    
    print(f'Timestamp: {data.get(\"timestamp\", \"N/A\")}')
    print(f'Device: {data.get(\"system_info\", {}).get(\"device\", \"N/A\")}')
    print('')
    
    results = data.get('results', {})
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'error' not in metrics:
            params_k = metrics.get('parameters_K', 'N/A')
            inference_ms = metrics.get('inference_mean_ms', 'N/A')
            memory_mb = metrics.get('memory_peak_mb', 'N/A')
            edge_ready = metrics.get('edge_ready', False)
            print(f'{model_name}:')
            print(f'  Parameters: {params_k}K')
            print(f'  Inference: {inference_ms}ms')
            print(f'  Memory: {memory_mb}MB')
            print(f'  Edge Ready: {\"✅\" if edge_ready else \"❌\"}')
            print('')
        elif isinstance(metrics, dict) and 'error' in metrics:
            print(f'{model_name}: ❌ Error - {metrics[\"error\"]}')
            print('')
    
except Exception as e:
    print(f'Could not parse results: {e}')
    print('Results file exists but may be incomplete.')
"
    fi
    
    echo ""
    echo "🎯 D1 Configuration Summary:"
    echo "✅ PASE-Net: Should show ~64K parameters"
    echo "✅ CNN: Should show ~64K parameters"
    echo "✅ BiLSTM: Capacity-matched parameters"
    echo ""
    echo "🏁 Xavier D1 Direct Measurement Complete!"
    
else
    echo ""
    echo "❌ Measurement failed!"
    echo "🔧 Possible solutions:"
    echo "  1. Check available memory: free -h"
    echo "  2. Try CPU mode: python3 measure_d1_true_efficiency_xavier.py --device cpu"
    echo "  3. Check GPU status: nvidia-smi"
    echo "  4. Verify PyTorch CUDA: python3 -c 'import torch; print(torch.cuda.is_available())'"
    exit 1
fi

echo ""
echo "📁 Output saved as: $OUTPUT_FILE"
echo "🔄 To run again: ./$(basename $0)"
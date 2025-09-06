#!/bin/bash
# Conformer-lite Xavier Measurement Script
# This script runs the Conformer-lite efficiency measurement on NVIDIA AGX Xavier 32G

set -e  # Exit on any error

echo "🚀 Starting Conformer-lite Xavier Efficiency Measurement"
echo "📅 Date: $(date)"
echo "🖥️  Host: $(hostname)"
echo "📂 Working Directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "src/models.py" ]; then
    echo "❌ Error: src/models.py not found. Please run from project root directory."
    exit 1
fi

# Check Python environment
echo "🐍 Python Environment Check:"
python3 --version
which python3

# Check PyTorch and CUDA
echo "🔥 PyTorch and CUDA Check:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Memory Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
fi

# Create results directory if it doesn't exist
mkdir -p results_gpu
echo "📁 Results will be saved to: results_gpu/"

# Run Conformer-lite measurement
echo ""
echo "🔧 Running Conformer-lite Efficiency Measurement..."
echo "⏰ Start time: $(date)"

# Run with CUDA if available, otherwise CPU
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "🎮 Using CUDA device"
    python3 measure_conformer_lite_xavier.py --device cuda --output results_gpu/xavier_conformer_lite_{timestamp}.json
else
    echo "💻 Using CPU device (CUDA not available)"
    python3 measure_conformer_lite_xavier.py --device cpu --output results_gpu/xavier_conformer_lite_{timestamp}.json
fi

# Check if measurement was successful
if [ $? -eq 0 ]; then
    echo "✅ Conformer-lite measurement completed successfully!"
    
    # Find the latest result file
    LATEST_RESULT=$(ls -t results_gpu/xavier_conformer_lite_*.json 2>/dev/null | head -n1)
    if [ -n "$LATEST_RESULT" ]; then
        echo "📊 Latest result file: $LATEST_RESULT"
        
        # Display summary
        echo ""
        echo "📋 Measurement Summary:"
        python3 -c "
import json
import sys
try:
    with open('$LATEST_RESULT', 'r') as f:
        data = json.load(f)
    
    if 'results' in data and 'conformer_lite' in data['results']:
        r = data['results']['conformer_lite']
        if 'error' not in r:
            print(f'  Model: {r[\"model\"]}')
            print(f'  Parameters: {r[\"parameters\"]:,} ({r[\"parameters_K\"]:.1f}K)')
            print(f'  Inference Time: {r[\"inference_mean_ms\"]:.2f}±{r[\"inference_std_ms\"]:.2f}ms')
            print(f'  Memory Usage: {r[\"memory_peak_mb\"]:.1f}MB')
            print(f'  Edge Ready: {\"✓\" if r[\"edge_ready\"] else \"✗\"}')
            
            # Paper 1 comparison
            expected_params = 1498672
            expected_gpu_ms = 5.16
            params_diff = r['parameters'] - expected_params
            gpu_diff = r['inference_mean_ms'] - expected_gpu_ms
            
            print(f'')
            print(f'  Paper 1 Table 1 Comparison:')
            print(f'    Parameters: {r[\"parameters\"]:,} vs {expected_params:,} (diff: {params_diff:,})')
            print(f'    GPU Latency: {r[\"inference_mean_ms\"]:.2f}ms vs {expected_gpu_ms:.2f}ms (diff: {gpu_diff:.2f}ms)')
            
            params_match = abs(params_diff) < 1000
            gpu_match = abs(gpu_diff) < 0.5
            status = '✅ Match' if (params_match and gpu_match) else '⚠️ Different'
            print(f'    Status: {status}')
        else:
            print(f'  Error: {r[\"error\"]}')
    else:
        print('  No results found in file')
except Exception as e:
    print(f'  Error reading results: {e}')
"
    else
        echo "⚠️  No result files found in results_gpu/"
    fi
else
    echo "❌ Conformer-lite measurement failed!"
    exit 1
fi

echo ""
echo "🏁 Conformer-lite Xavier Measurement Complete!"
echo "⏰ End time: $(date)"

# Optional: Run all models measurement as well
echo ""
read -p "🤔 Would you like to run the full D1 models measurement as well? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔧 Running All D1 Models Measurement..."
    python3 measure_all_models_xavier.py --device cuda --output results_gpu/xavier_d1_all_models_{timestamp}.json
    
    if [ $? -eq 0 ]; then
        echo "✅ All models measurement completed successfully!"
    else
        echo "❌ All models measurement failed!"
    fi
fi

echo "🎯 All measurements complete!"
#!/bin/bash
# Run D1 True Parameter Configuration Xavier Efficiency Measurement
# This script measures models with actual D1 experiment parameters:
# PASE-Net & CNN: ~64K parameters, BiLSTM: capacity-matched

echo "🚀 D1 True Parameter Configuration Xavier Efficiency Measurement"
echo "🎯 Target: PASE-Net & CNN ~64K parameters (D1 experiment actual config)"

# Check platform
if [[ $(uname -m) == "aarch64" ]]; then
    echo "✅ Detected ARM64 architecture (Xavier/Jetson platform)"
    DEVICE="cuda"
else
    echo "⚠️  Not on ARM64, running on current platform"
    DEVICE="cpu"
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "🖥️  No NVIDIA GPU detected, using CPU"
    DEVICE="cpu"
fi

# Set output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="xavier_d1_true_efficiency_${TIMESTAMP}.json"

echo "📁 Output: $OUTPUT_FILE"
echo "⏳ Starting measurement..."

# Run with D1 true configuration (T=128, F=30, classes=8)
python measure_d1_true_efficiency_xavier.py \
    --device $DEVICE \
    --output $OUTPUT_FILE \
    --T 128 \
    --F 30 \
    --classes 8

if [ $? -eq 0 ]; then
    echo "✅ Measurement completed!"
    echo "📊 Results saved in: $OUTPUT_FILE"
    
    # Quick summary if jq available
    if command -v jq &> /dev/null; then
        echo ""
        echo "📋 Quick Summary:"
        jq -r '.results | to_entries[] | "\(.key): \(.value.parameters_K // "ERROR")K params, \(.value.inference_mean_ms // "N/A")ms"' $OUTPUT_FILE
    fi
    
    echo ""
    echo "🎯 Key Results (D1 True Config):"
    echo "- PASE-Net & CNN: Should be ~64K parameters"
    echo "- BiLSTM: Capacity-matched with other models"
    echo "- All models: Edge deployment ready"
    
else
    echo "❌ Measurement failed!"
    exit 1
fi

echo "🏁 D1 True Configuration Measurement Complete!"
#!/usr/bin/env bash
set -e
set -euo pipefail

# D3 LORO (Leave-One-Room-Out) Cross-Domain Experiments
# Evaluating model robustness across different environments/rooms

# Environment setup
export LC_ALL=C
export LANG=C

ROOT="$(pwd)"
OUT_DIR="${ROOT}/results/d3/loro"
BENCHMARK_PATH="${ROOT}/benchmarks/WiFi-CSI-Sensing-Benchmark-main"

mkdir -p "${OUT_DIR}"

PYTHON=${PYTHON:-python}

# D3 Configuration - based on D2 validated models
models=("enhanced" "cnn" "bilstm" "conformer_lite")
seeds=(0 1 2 3 4)  # Consistent with D2

epochs=${EPOCHS:-100}
extra_args=${EXTRA_ARGS:-}

echo "[INFO] Starting D3 LORO Cross-Domain Experiments..."
echo "[INFO] Benchmark Path: ${BENCHMARK_PATH}"
echo "[INFO] Models: ${models[*]}"
echo "[INFO] Seeds: ${seeds[*]}"
echo "[INFO] Output Directory: ${OUT_DIR}"

# Check benchmark dataset availability
if [ ! -d "${BENCHMARK_PATH}" ]; then
    echo "[ERROR] Benchmark dataset not found at ${BENCHMARK_PATH}"
    echo "[ERROR] Please ensure WiFi-CSI-Sensing-Benchmark dataset is available"
    exit 1
fi

# Run LORO experiments
total_runs=$((${#models[@]} * ${#seeds[@]}))
current_run=0

for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        current_run=$((current_run + 1))
        echo ""
        echo "[RUN ${current_run}/${total_runs}] LORO: model=${model}, seed=${seed}"
        
        ${PYTHON} -m src.train_cross_domain \
            --model "${model}" \
            --protocol "loro" \
            --benchmark_path "${BENCHMARK_PATH}" \
            --seed "${seed}" \
            --epochs "${epochs}" \
            --output_dir "${OUT_DIR}" \
            ${extra_args}
        
        echo "[OK] Completed LORO for ${model} seed ${seed}"
    done
done

echo ""
echo "[INFO] D3 LORO experiments completed!"
echo "[INFO] Results saved to: ${OUT_DIR}"
echo "[INFO] Next steps:"
echo "  1. Run validation: python scripts/validate_d3_acceptance.py --protocol loro"
echo "  2. Generate summary: python scripts/export_d3_summary.py --protocol loro"
echo "  3. Compare with LOSO: python scripts/compare_d3_protocols.py"

# Optional: Generate quick summary
if command -v python &> /dev/null; then
    echo ""
    echo "[INFO] Generating quick summary..."
    ${PYTHON} -c "
import json, glob
from pathlib import Path

loro_files = glob.glob('${OUT_DIR}/*.json')
print(f'Generated {len(loro_files)} LORO result files')

if loro_files:
    sample_file = loro_files[0]
    with open(sample_file) as f:
        data = json.load(f)
    
    n_folds = len(data.get('fold_results', []))
    model = data.get('model', 'unknown')
    
    print(f'Sample: {model} with {n_folds} LORO folds')
    
    if 'aggregate_stats' in data:
        stats = data['aggregate_stats']
        if 'macro_f1' in stats:
            print(f'  Macro F1: {stats[\"macro_f1\"][\"mean\"]:.3f}±{stats[\"macro_f1\"][\"std\"]:.3f}')
        if 'falling_f1' in stats:
            print(f'  Falling F1: {stats[\"falling_f1\"][\"mean\"]:.3f}±{stats[\"falling_f1\"][\"std\"]:.3f}')
"
fi
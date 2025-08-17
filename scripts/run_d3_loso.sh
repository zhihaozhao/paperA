#!/usr/bin/env bash
set -e
set -euo pipefail

# D3 LOSO (Leave-One-Subject-Out) Cross-Domain Experiments
# Based on successful D2 validation, extending to real WiFi CSI benchmark data

# Environment setup
export LC_ALL=C
export LANG=C

ROOT="$(pwd)"
OUT_DIR="${ROOT}/results/d3/loso"
BENCHMARK_PATH="${ROOT}/benchmarks/WiFi-CSI-Sensing-Benchmark-main"

mkdir -p "${OUT_DIR}"

PYTHON=${PYTHON:-python}

# D3 Configuration - based on D2 validated models
models=("enhanced" "cnn" "bilstm" "conformer_lite")
seeds=(0 1 2 3 4)  # Consistent with D2

epochs=${EPOCHS:-100}
extra_args=${EXTRA_ARGS:-}

echo "[INFO] Starting D3 LOSO Cross-Domain Experiments..."
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

# Run LOSO experiments
total_runs=$((${#models[@]} * ${#seeds[@]}))
current_run=0

for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        current_run=$((current_run + 1))
        echo ""
        echo "[RUN ${current_run}/${total_runs}] LOSO: model=${model}, seed=${seed}"
        
        ${PYTHON} -m src.train_cross_domain \
            --model "${model}" \
            --protocol "loso" \
            --benchmark_path "${BENCHMARK_PATH}" \
            --seed "${seed}" \
            --epochs "${epochs}" \
            --output_dir "${OUT_DIR}" \
            ${extra_args}
        
        echo "[OK] Completed LOSO for ${model} seed ${seed}"
    done
done

echo ""
echo "[INFO] D3 LOSO experiments completed!"
echo "[INFO] Results saved to: ${OUT_DIR}"
echo "[INFO] Next steps:"
echo "  1. Run validation: python scripts/validate_d3_acceptance.py --protocol loso"
echo "  2. Generate summary: python scripts/export_d3_summary.py --protocol loso"
echo "  3. Run LORO experiments: bash scripts/run_d3_loro.sh"
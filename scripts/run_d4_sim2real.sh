#!/usr/bin/env bash
set -e
set -euo pipefail

# D4 Sim2Real Label Efficiency Experiments
# Evaluating synthetic-to-real transfer with minimal labeling requirements

# Environment setup
export LC_ALL=C
export LANG=C

ROOT="$(pwd)"
OUT_DIR="${ROOT}/results/d4/sim2real"
BENCHMARK_PATH="${ROOT}/benchmarks/WiFi-CSI-Sensing-Benchmark-main"
D2_MODELS_DIR="${ROOT}/checkpoints/d2"  # Pre-trained models from D2

mkdir -p "${OUT_DIR}"

PYTHON=${PYTHON:-python}

# D4 Configuration
models=("enhanced" "cnn" "bilstm" "conformer_lite")
seeds=(0 1 2 3 4)
label_ratios=(0.01 0.05 0.10 0.15 0.20 0.50 1.00)  # 1% to 100% real data
transfer_methods=("zero_shot" "linear_probe" "fine_tune" "temp_scale")

# Training parameters for adaptation methods
linear_probe_epochs=${LINEAR_EPOCHS:-50}
fine_tune_epochs=${FINETUNE_EPOCHS:-30}
extra_args=${EXTRA_ARGS:-}

echo "[INFO] Starting D4 Sim2Real Label Efficiency Experiments..."
echo "[INFO] Benchmark Path: ${BENCHMARK_PATH}"
echo "[INFO] D2 Models Path: ${D2_MODELS_DIR}"
echo "[INFO] Models: ${models[*]}"
echo "[INFO] Seeds: ${seeds[*]}"
echo "[INFO] Label Ratios: ${label_ratios[*]}"
echo "[INFO] Transfer Methods: ${transfer_methods[*]}"
echo "[INFO] Output Directory: ${OUT_DIR}"

# Check prerequisites
if [ ! -d "${BENCHMARK_PATH}" ]; then
    echo "[ERROR] Benchmark dataset not found at ${BENCHMARK_PATH}"
    echo "[ERROR] Please ensure WiFi-CSI-Sensing-Benchmark dataset is available"
    exit 1
fi

if [ ! -d "${D2_MODELS_DIR}" ]; then
    echo "[WARNING] D2 models directory not found at ${D2_MODELS_DIR}"
    echo "[WARNING] Will train from scratch if pre-trained models unavailable"
fi

# Calculate total experiments
total_runs=$((${#models[@]} * ${#seeds[@]} * ${#label_ratios[@]} * ${#transfer_methods[@]}))
current_run=0

echo "[INFO] Total experiments to run: ${total_runs}"
echo ""

# Main experimental loop
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        for ratio in "${label_ratios[@]}"; do
            for method in "${transfer_methods[@]}"; do
                current_run=$((current_run + 1))
                
                echo "[RUN ${current_run}/${total_runs}] Sim2Real: ${model}, ratio=${ratio}, method=${method}, seed=${seed}"
                
                # Construct output filename
                out_file="${OUT_DIR}/sim2real_${model}_${method}_ratio${ratio}_seed${seed}.json"
                
                # Skip if already exists (for resumability)
                if [ -f "${out_file}" ]; then
                    echo "[SKIP] Output file already exists: ${out_file}"
                    continue
                fi
                
                # Set method-specific parameters
                method_args=""
                case "${method}" in
                    "linear_probe")
                        method_args="--adaptation_epochs ${linear_probe_epochs} --freeze_backbone"
                        ;;
                    "fine_tune")
                        method_args="--adaptation_epochs ${fine_tune_epochs} --fine_tune_lr 1e-4"
                        ;;
                    "temp_scale"|"zero_shot")
                        method_args=""
                        ;;
                esac
                
                # Run experiment
                ${PYTHON} -m src.run_sim2real_single \
                    --model "${model}" \
                    --transfer_method "${method}" \
                    --benchmark_path "${BENCHMARK_PATH}" \
                    --d2_models_dir "${D2_MODELS_DIR}" \
                    --label_ratio "${ratio}" \
                    --seed "${seed}" \
                    --output_file "${out_file}" \
                    ${method_args} \
                    ${extra_args}
                
                echo "[OK] Completed: ${out_file}"
            done
        done
    done
done

echo ""
echo "[INFO] D4 Sim2Real experiments completed!"
echo "[INFO] Results saved to: ${OUT_DIR}"
echo "[INFO] Next steps:"
echo "  1. Run validation: python scripts/validate_d4_acceptance.py"
echo "  2. Generate label efficiency plots: python scripts/plot_sim2real_curves.py"
echo "  3. Export summary: python scripts/export_d4_summary.py"

# Generate comprehensive summary
if command -v python &> /dev/null; then
    echo ""
    echo "[INFO] Generating D4 experiment summary..."
    ${PYTHON} -c "
import json, glob
import numpy as np
from collections import defaultdict

sim2real_files = glob.glob('${OUT_DIR}/*.json')
print(f'Generated {len(sim2real_files)} Sim2Real result files')

# Group results by model and method
results_by_method = defaultdict(list)
for file_path in sim2real_files:
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        model = data.get('model', 'unknown')
        method = data.get('method', 'unknown')
        ratio = data.get('label_ratio', 0.0)
        metrics = data.get('metrics', {})
        
        key = f'{model}_{method}'
        results_by_method[key].append({
            'ratio': ratio,
            'falling_f1': metrics.get('falling_f1', 0.0),
            'macro_f1': metrics.get('macro_f1', 0.0)
        })
    except Exception as e:
        print(f'Error reading {file_path}: {e}')

# Summary statistics
print('\nD4 Sim2Real Summary:')
for method_key, results in results_by_method.items():
    if results:
        falling_f1s = [r['falling_f1'] for r in results if r['falling_f1'] > 0]
        if falling_f1s:
            print(f'  {method_key}: Falling F1 = {np.mean(falling_f1s):.3f}±{np.std(falling_f1s):.3f} (n={len(falling_f1s)})')

print(f'\nTotal configurations: {len(sim2real_files)} / {total_runs}')
completion_rate = len(sim2real_files) / ${total_runs} * 100
print(f'Completion rate: {completion_rate:.1f}%')
"
fi
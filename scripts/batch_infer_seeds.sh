#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/batch_infer_seeds.sh 0 1 2
# If no args provided, defaults to "0 1 2"

SEEDS=${@:-"0 1 2"}
OUTJSON=results/synth/out_mid.json

mkdir -p results/preds results/plots results/ckpt results/synth

for S in $SEEDS; do
  CKPT="results/ckpt/best_enhanced_mid_s${S}.pt"
  NPZ="results/preds/enhanced_mid_s${S}.npz"
  PLOT="results/plots/reliability_enhanced_mid_s${S}.png"

  echo "=== Seed ${S} ==="
  bash scripts/run_infer.sh "$CKPT" "$OUTJSON" "$NPZ" te
  bash scripts/plot_reliability.sh "$NPZ" 15 "$PLOT"
done

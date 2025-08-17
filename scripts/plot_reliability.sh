#!/usr/bin/env bash
set -euo pipefail

# Args:
#   NPZ [N_BINS] [PLOT] [CSV] [CSV_PER_CLASS]
# Defaults:
#   NPZ=results/preds/enhanced_mid_s0.npz
#   N_BINS=15
#   PLOT=results/plots/reliability_enhanced_mid.png
#   CSV="" (no export)
#   CSV_PER_CLASS="" (no per-class export)

NPZ=${1:-results/preds/enhanced_mid_s0.npz}
N_BINS=${2:-15}
PLOT=${3:-results/plots/reliability_enhanced_mid.png}
CSV_PATH=${4:-}
CSV_PER_CLASS=${5:-}

mkdir -p "$(dirname "$PLOT")"
if [[ -n "${CSV_PATH:-}" ]]; then
  mkdir -p "$(dirname "$CSV_PATH")"
fi
if [[ -n "${CSV_PER_CLASS:-}" ]]; then
  mkdir -p "$(dirname "$CSV_PER_CLASS")"
fi
export MPLBACKEND=Agg

echo "[reliability] npz=$NPZ n_bins=$N_BINS out=$PLOT csv=${CSV_PATH:-<none>} csv_per_class=${CSV_PER_CLASS:-<none>}"
if [[ -n "${CSV_PATH:-}" || -n "${CSV_PER_CLASS:-}" ]]; then
  python -m src.reliability \
    --from_npz "$NPZ" \
    --n_bins "$N_BINS" \
    --save "$PLOT" \
    ${CSV_PATH:+--csv "$CSV_PATH"} \
    ${CSV_PER_CLASS:+--csv_per_class "$CSV_PER_CLASS"}
else
  python -m src.reliability \
    --from_npz "$NPZ" \
    --n_bins "$N_BINS" \
    --save "$PLOT"
fi
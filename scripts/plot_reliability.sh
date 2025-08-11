#!/usr/bin/env bash
set -euo pipefail

# Args:
#   NPZ [N_BINS] [PLOT]
# Defaults:
#   NPZ=results/preds/enhanced_mid_s0.npz
#   N_BINS=15
#   PLOT=results/plots/reliability_enhanced_mid.png

NPZ=${1:-results/preds/enhanced_mid_s0.npz}
N_BINS=${2:-15}
PLOT=${3:-results/plots/reliability_enhanced_mid.png}

mkdir -p "$(dirname "$PLOT")"
export MPLBACKEND=Agg

echo "[reliability] npz=$NPZ n_bins=$N_BINS out=$PLOT"
python -m src.reliability \
  --from_npz "$NPZ" \
  --n_bins "$N_BINS" \
  --save "$PLOT"

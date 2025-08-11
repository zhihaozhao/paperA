#!/usr/bin/env bash
set -euo pipefail

# Args:
#   CKPT [OUTJSON] [NPZ] [SPLIT]
# Defaults:
#   CKPT=results/ckpt/best_enhanced_mid_s0.pt
#   OUTJSON=results/synth/out_mid.json
#   NPZ=results/preds/enhanced_mid_s0.npz
#   SPLIT=te

CKPT=${1:-results/ckpt/best_enhanced_mid_s0.pt}
OUTJSON=${2:-results/synth/out_mid.json}
NPZ=${3:-results/preds/enhanced_mid_s0.npz}
SPLIT=${4:-te}

if [[ ! -f "$CKPT" ]]; then
  echo "[infer] CKPT not found: $CKPT"
  exit 1
fi

mkdir -p "$(dirname "$NPZ")"
mkdir -p "$(dirname "$OUTJSON")"

echo "[infer] ckpt=$CKPT out_json=$OUTJSON split=$SPLIT npz=$NPZ"
python -m src.infer \
  --ckpt "$CKPT" \
  --out_json "$OUTJSON" \
  --split "$SPLIT" \
  --save_npz "$NPZ"

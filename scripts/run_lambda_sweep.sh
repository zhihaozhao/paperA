#!/usr/bin/env bash
set -e
source "$(dirname "$0")/env.sh"
mkdir -p results/synth_lambda
LAMBDAS=(0 0.02 0.05 0.08 0.12 0.18)
for L in "${LAMBDAS[@]}"; do
  "$PY" -m src.train_eval --model enhanced --logit_l2 $L --seed 0 --difficulty mid --out "results/synth_lambda/enhanced_l${L}.json"
done
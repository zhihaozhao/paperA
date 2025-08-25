#!/usr/bin/env bash
set -euo pipefail

MODEL="pinn_lstm_ms"
SEED=${SEED:-0}
OUT=${OUT:-results/pinn_lstm_ms_out.json}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-256}
F=${F:-52}
T=${T:-128}
N=${N:-20000}
LAMBDA_SMOOTH=${LAMBDA_SMOOTH:-0.01}
LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.0}
WIN_SIZES=${WIN_SIZES:-"32,64,128"}

python -u src/train_eval.py \
  --model ${MODEL} \
  --seed ${SEED} \
  --F ${F} --T ${T} --n_samples ${N} \
  --epochs ${EPOCHS} --batch ${BATCH} \
  --out_json ${OUT} \
  --logit_l2 0.0 \
  --class_overlap 0.6 --env_burst_rate 0.1 --gain_drift_std 0.1 --sc_corr_rho 0.5 \
  --temp_mode logspace --temp_min 1.0 --temp_max 5.0 --temp_steps 50 \
  --pinn_lambda_smooth ${LAMBDA_SMOOTH} \
  --pinn_lambda_energy ${LAMBDA_ENERGY} \
  --pinn_ms_windows ${WIN_SIZES}
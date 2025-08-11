#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.

PY=${PYTHON_BIN:-python}
ENTRY=${ENTRY:--m src.train_eval}

N=${N:-2000}
TT=${T:-128}
F=${F:-30}
DIFF=${DIFF:-mid}
SEED=${SEED:-0}
BATCH=${BATCH:-64}
CLASS_OVERLAP=${CLASS_OVERLAP:-0.0}
SC_CORR=${SC_CORR:-None}
BURST=${BURST:-0.0}
DRIFT=${DRIFT:-0.0}

OUTDIR=${OUTDIR:-results/synth}
mkdir -p "${OUTDIR}" results/ckpt results/preds results/plots

echo "[1/4] Default (no perturbations)"
${PY} ${ENTRY} --model enhanced --difficulty ${DIFF} --seed ${SEED} \
  --epochs 60 --patience 10 --early_metric macro_f1 --ckpt_dir results/ckpt \
  --batch ${BATCH} --n_samples ${N} --T ${TT} --F ${F} \
  --class_overlap ${CLASS_OVERLAP} \
  --sc_corr_rho None --env_burst_rate 0.0 --gain_drift_std 0.0 \
  --out ${OUTDIR}/out_default.json

echo "[2/4] Subcarrier correlation (rho=0.7)"
${PY} ${ENTRY} --model enhanced --difficulty ${DIFF} --seed ${SEED} \
  --epochs 60 --patience 10 --early_metric macro_f1 --ckpt_dir results/ckpt \
  --batch ${BATCH} --n_samples ${N} --T ${TT} --F ${F} \
  --class_overlap ${CLASS_OVERLAP} \
  --sc_corr_rho 0.7 --env_burst_rate 0.0 --gain_drift_std 0.0 \
  --out ${OUTDIR}/out_sc_corr.json

echo "[3/4] Environmental bursts (rate=0.05)"
${PY} ${ENTRY} --model enhanced --difficulty ${DIFF} --seed ${SEED} \
  --epochs 60 --patience 10 --early_metric macro_f1 --ckpt_dir results/ckpt \
  --batch ${BATCH} --n_samples ${N} --T ${TT} --F ${F} \
  --class_overlap ${CLASS_OVERLAP} \
  --sc_corr_rho None --env_burst_rate 0.05 --gain_drift_std 0.0 \
  --out ${OUTDIR}/out_burst.json

echo "[4/4] Gain drift (std=0.003)"
${PY} ${ENTRY} --model enhanced --difficulty ${DIFF} --seed ${SEED} \
  --epochs 60 --patience 10 --early_metric macro_f1 --ckpt_dir results/ckpt \
  --batch ${BATCH} --n_samples ${N} --T ${TT} --F ${F} \
  --class_overlap ${CLASS_OVERLAP} \
  --sc_corr_rho None --env_burst_rate 0.0 --gain_drift_std 0.003 \
  --out ${OUTDIR}/out_drift.json
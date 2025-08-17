#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
# No YAML config required. Override via env vars if needed:
#   ENTRY="-m src.train_eval" or default to "src/train_eval.py"
#   N/T/F/DIFF/SEED/POS_CLS environment variables

PYTHON_BIN=${PYTHON_BIN:-python3}
ENTRY=${ENTRY:-src/train_eval.py}
N=${N:-2000}
T=${T:-128}
F=${F:-30}
DIFF=${DIFF:-mid}
SEED=${SEED:-0}
POS_CLS=${POS_CLS:-1}

echo "[1/4] Default (no perturbations)"
${PYTHON_BIN} ${ENTRY} \
  --n_samples ${N} --T ${T} --F ${F} --difficulty ${DIFF} --seed ${SEED} \
  --positive_class ${POS_CLS} \
  --sc_corr_rho None --env_burst_rate 0.0 --gain_drift_std 0.0

echo "[2/4] Subcarrier correlation (rho=0.7)"
${PYTHON_BIN} ${ENTRY} \
  --n_samples ${N} --T ${T} --F ${F} --difficulty ${DIFF} --seed ${SEED} \
  --positive_class ${POS_CLS} \
  --sc_corr_rho 0.7 --env_burst_rate 0.0 --gain_drift_std 0.0

echo "[3/4] Environmental bursts (rate=0.05)"
${PYTHON_BIN} ${ENTRY} \
  --n_samples ${N} --T ${T} --F ${F} --difficulty ${DIFF} --seed ${SEED} \
  --positive_class ${POS_CLS} \
  --sc_corr_rho None --env_burst_rate 0.05 --gain_drift_std 0.0

echo "[4/4] Gain drift (std=0.003)"
${PYTHON_BIN} ${ENTRY} \
  --n_samples ${N} --T ${T} --F ${F} --difficulty ${DIFF} --seed ${SEED} \
  --positive_class ${POS_CLS} \
  --sc_corr_rho None --env_burst_rate 0.0 --gain_drift_std 0.003

echo "Done. If Falling index != 1, run with: POS_CLS=3 bash scripts/run_main.sh"
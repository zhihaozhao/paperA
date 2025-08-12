
#!/usr/bin/env bash
set -euo pipefail

DIFF=${1:-mid}
SEED=${2:-0}
LAMBDAS=${3:-"0,0.02,0.05,0.08,0.12,0.18"}
TEMP_MODE=${4:-"logspace"}
TEMP_MIN=${5:-0.3}
TEMP_MAX=${6:-8.0}
TEMP_STEPS=${7:-80}
USE_ECE_CAL=${8:-1}

python scripts/sweep_lambda.py \
  --only ${DIFF}:${SEED} \
  --lambdas ${LAMBDAS} \
  --force \
  --temp_mode ${TEMP_MODE} \
  --temp_min ${TEMP_MIN} \
  --temp_max ${TEMP_MAX} \
  --temp_steps ${TEMP_STEPS}

CSV="results/synth_lambda/metrics_lambda_full.csv"
OUTPDF="plots/fig_lambda_curves.pdf"

if [[ "${USE_ECE_CAL}" == "1" ]]; then
  python scripts/plot_lambda_curves.py --csv "${CSV}" --out_pdf "${OUTPDF}" --model enhanced --difficulty "${DIFF}" --seed ${SEED} --use_ece_cal
else
  python scripts/plot_lambda_curves.py --csv "${CSV}" --out_pdf "${OUTPDF}" --model enhanced --difficulty "${DIFF}" --seed ${SEED}
fi

echo "Done. CSV=${CSV}, PDF=${OUTPDF}"

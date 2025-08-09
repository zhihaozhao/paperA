#!/usr/bin/env bash
set -euo pipefail

# 可选：激活环境
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate py310

# 减少 locale 噪音（Windows Git Bash 下）
export LC_ALL=C
export LANG=C

ROOT="$(pwd)"
OUT_DIR="${ROOT}/results/synth"
mkdir -p "${OUT_DIR}"

PYTHON=${PYTHON:-python}

models=("enhanced" "lstm" "tcn" "txf")
difficulties=("mid")   # 初始只跑 mid；验证通过后可改为 ("low" "mid" "high")
seeds=(0)              # 初始只跑 seed=0；通过后可改为 (0 1 2 3 4 5 6 7)

epochs=${EPOCHS:-20}
extra_args=${EXTRA_ARGS:-}   # 例如传 --logit_l2 0.05

echo "[INFO] Running synth eval..."
for m in "${models[@]}"; do
  for d in "${difficulties[@]}"; do
    for s in "${seeds[@]}"; do
      out="${OUT_DIR}/${m}_${d}_s${s}.json"
      echo "[RUN] model=${m} diff=${d} seed=${s} -> ${out}"
      ${PYTHON} -m src.train_eval \
        --model "${m}" \
        --difficulty "${d}" \
        --seed "${s}" \
        --epochs "${epochs}" \
        --out "${out}" \
        ${extra_args}
    done
  done
done

echo "[OK] Done. Outputs in ${OUT_DIR}"

# 扩展到全量（验证后解开）
# difficulties=("low" "mid" "high")
# seeds=(0 1 2 3 4 5 6 7)
# 然后重新运行本脚本即可

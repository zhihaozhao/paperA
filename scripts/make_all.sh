#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export LANG=C
# 0) 统一 UTF-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

# 1) 激活 conda 环境
# 说明：在 Git Bash/WSL 下，conda init bash 后要 source ~/.bashrc
# 如果你已经执行过 `conda init bash`，保留下面这一行即可：
source ~/.bashrc 2>/dev/null || true
if [ -f "/d/workspace_AI/Anaconda3/etc/profile.d/conda.sh" ]; then
source "/d/workspace_AI/Anaconda3/etc/profile.d/conda.sh"
else
echo "[ERROR] conda.sh not found. Fix the path to your Anaconda3." >&2
exit 1
fi
# 默认环境名
ENV_NAME="${ENV_NAME:-py310}"

# 如果你不想依赖 bashrc，可改为直接 source conda.sh：
# CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
# [ -f "$CONDA_SH" ] && source "$CONDA_SH"
# conda activate "$ENV_NAME"

conda activate "$ENV_NAME"

# 2) 切换到项目根
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "[INFO] ROOT = $ROOT_DIR"
python - <<'PY'
import sys, os, platform
print("python =", sys.executable)
print("cwd    =", os.getcwd())
print("pyver  =", platform.python_version())
PY

# 3) 运行任务
python -m src.train_eval --model enhanced --logit_l2 0.05 --seed 0 --difficulty mid --out results/synth/enhanced_s0.json
python -m src.train_eval --model lstm     --seed 0 --difficulty mid --out results/synth/lstm_s0.json

# 4) 生成绘图与表格
# 先确保缺失的必需图会有占位，避免 LaTeX 报错
python - <<'PY'
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

def ensure_pdf(path, title="placeholder"):
    if os.path.exists(path):
        return
    plt.figure(figsize=(4.5,3.2))
    plt.text(0.5, 0.5, title, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

needed = [
    "plots/fig_overlap_scatter.pdf",
    "plots/fig_synth_bars.pdf",
    "plots/fig_reliability_enhanced_vs_baselines.pdf",
    "plots/fig_bucket_perf_curves.pdf",
    "plots/fig_cost_sensitive.pdf",
    "plots/fig_sim2real_curve.pdf",
]
for p in needed:
    ensure_pdf(p, title=os.path.basename(p))

PY

python -m src.plotting
python -m src.tables

# 5) 编译论文
# 优先用 latexmk，否则退回 pdflatex
if command -v latexmk >/dev/null 2>&1; then
  (cd paper && latexmk -pdf -silent main.tex || pdflatex main.tex)
else
  (cd paper && pdflatex main.tex || true)
fi

echo "Done. See plots/, tables/, paper/main.pdf"
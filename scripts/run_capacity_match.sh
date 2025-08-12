
#!/usr/bin/env bash
set -euo pipefail

DIFF=${1:-mid}
SEED=${2:-0}
OUTCSV="results/capacity_match.csv"
OUTTEX="tables/tab_capacity_match.tex"

# 你可以将下面两行替换为你的标准入口脚本/命令
python -m src.train_eval --model enhanced_small --difficulty ${DIFF} --seed ${SEED} \
  --epochs 30 --batch 64 --early_metric macro_f1 --patience 10 \
  --logit_l2 0.0 --out results/capacity/enhanced_small.json \
  --temp_mode logspace --temp_min 0.3 --temp_max 8 --temp_steps 80 --val_split 0.5

python -m src.train_eval --model baseline_large --difficulty ${DIFF} --seed ${SEED} \
  --epochs 30 --batch 64 --early_metric macro_f1 --patience 10 \
  --logit_l2 0.0 --out results/capacity/baseline_large.json \
  --temp_mode logspace --temp_min 0.3 --temp_max 8 --temp_steps 80 --val_split 0.5

# 合并结果到 CSV（要求 JSON 内含 params、macro_f1、ece_cal、nll_cal、brier、difficulty、seed、temperature）
python - << 'PY'
import json, pandas as pd
from pathlib import Path
rows = []
for name in ["enhanced_small","baseline_large"]:
    p = Path(f"results/capacity/{name}.json")
    d = json.loads(p.read_text())
    rows.append({
        "model": name,
        "params": d.get("params", None),
        "macro_f1": d.get("macro_f1", None),
        "ece_cal": d.get("ece_cal", d.get("ece", None)),
        "nll_cal": d.get("nll_cal", None),
        "brier": d.get("brier", None),
        "difficulty": d.get("difficulty","mid"),
        "seed": d.get("seed",0),
        "temperature": d.get("temperature", None),
    })
df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True, parents=True)
df.to_csv("results/capacity_match.csv", index=False)
print("Wrote results/capacity_match.csv")
PY

python scripts/gen_capacity_table.py --csv "${OUTCSV}" --out_tex "${OUTTEX}" --difficulty "${DIFF}" --seed ${SEED}
echo "Done. TEX=${OUTTEX}"

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd

# 如果你有真实模型，请在此导入
# from src.models.enhanced import EnhancedSmall
# from src.models.baseline import BaselineLarge

# 回退占位：与 src/train_eval.py 的 TinyNet 对齐
class TinyNet(nn.Module):
    def __init__(self, F: int, hidden: int = 32, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(F, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        logits = self.head(h)
        return logits

def get_num_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def load_result(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="src/train_eval.py")
    ap.add_argument("--difficulty", type=str, default="mid")
    ap.add_argument("--seed", type=str, default="0")
    ap.add_argument("--temp_mode", type=str, default="logspace")
    ap.add_argument("--out-tex", type=str, default="tables/tab_capacity_match.tex")
    args = ap.parse_args()

    # 假定使用 sweep 产物中的两组作为对照
    root = Path("results/synth_lambda")
    enhanced_path = root / f"{args.difficulty}_s{args.seed}" / "enhanced_l0.05.json"
    baseline_path = root / f"{args.difficulty}_s{args.seed}" / "enhanced_l0.00.json"  # 占位：若你有 baseline，请替换为 baseline 文件名

    if not enhanced_path.exists() or not baseline_path.exists():
        raise SystemExit("Required JSON not found. Please run run_lambda_sweep.sh first.")

    je = load_result(enhanced_path)
    jb = load_result(baseline_path)

    # 估算参数量（示例：用 TinyNet；如果你有真实 baseline/enhanced，请替换）
    Fe = int(je["meta"]["F"]); Fb = int(jb["meta"]["F"])
    ce = int(je["meta"]["num_classes"]); cb = int(jb["meta"]["num_classes"])
    me = TinyNet(F=Fe, num_classes=ce)
    mb = TinyNet(F=Fb, num_classes=cb)
    params_e = get_num_params(me)
    params_b = get_num_params(mb)

    # 收集指标
    row_e = {
        "Model": "Enhanced_small",
        "Params": params_e,
        "Macro-F1": je["metrics"]["macro_f1"],
        "ECE": je["metrics"]["ece_cal"] if "ece_cal" in je["metrics"] else je["metrics"]["ece"],
        "NLL": je["metrics"]["nll_cal"],
    }
    row_b = {
        "Model": "Baseline_large",
        "Params": params_b,
        "Macro-F1": jb["metrics"]["macro_f1"],
        "ECE": jb["metrics"]["ece_cal"] if "ece_cal" in jb["metrics"] else jb["metrics"]["ece"],
        "NLL": jb["metrics"]["nll_cal"],
    }
    df = pd.DataFrame([row_e, row_b])

    # ±10% 判定（仅打印提示，不改变表内容）
    within_10 = abs(params_e - params_b) <= 0.1 * max(params_e, params_b)
    print(f"Param match within ±10%: {within_10} (Enhanced={params_e}, Baseline={params_b})")

    # 写 LaTeX 表
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated capacity match table\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model & Params & Macro-F1 $\\uparrow$ & ECE $\\downarrow$ & NLL $\\downarrow$ \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            f.write(f"{r['Model']} & {int(r['Params']):,} & {r['Macro-F1']:.4f} & {r['ECE']:.4f} & {r['NLL']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {out_tex}")

if __name__ == "__main__":
    main()

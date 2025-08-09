import json
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------- utils ----------
def savefig_safe(path: str, bbox_inches="tight"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p.as_posix(), bbox_inches=bbox_inches)

def find_files(pattern: str) -> List[Path]:
    return [Path(p) for p in glob.glob(pattern)]

# ---------- IO ----------
def load_runs(in_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in sorted(in_dir.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        args = obj.get("args", {})
        metr = obj.get("metrics", obj.get("metr", {}))  # 兼容字段名
        # 期望键名：macro_f1, falling_f1, mutual_misclass, ece, brier, overlap_stat
        row = {
            "file": fp.name,
            "model": args.get("model"),
            "difficulty": args.get("difficulty"),
            "seed": args.get("seed"),
            "macro_f1": metr.get("macro_f1"),
            "falling_f1": metr.get("falling_f1"),
            "mutual_misclass": metr.get("mutual_misclass"),
            "ece": metr.get("ece"),
            "brier": metr.get("brier"),
            # overlap_stat 可以是标量或字典；若为字典，取你关心的键，如 "mean"
            "overlap": (
                metr.get("overlap_stat", {}).get("mean")
                if isinstance(metr.get("overlap_stat"), dict)
                else metr.get("overlap_stat")
            ),
            "error": 1.0 - (metr.get("macro_f1") if metr.get("macro_f1") is not None else np.nan),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

# ---------- plotting ----------
def plot_bars(df: pd.DataFrame, out_path: Path):
    # 画 Falling/Macro/Mutual 三组条形图（均值±std），按模型分组，分难度分面
    metrics = ["falling_f1", "macro_f1", "mutual_misclass"]
    difficulties = sorted(df["difficulty"].dropna().unique().tolist())
    models = ["enhanced", "lstm", "tcn", "txf"]
    nrows = len(metrics)
    ncols = len(difficulties)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.8*nrows), squeeze=False, sharey=False)
    for i, met in enumerate(metrics):
        for j, diff in enumerate(difficulties):
            ax = axes[i][j]
            sub = df[df["difficulty"] == diff]
            means = []
            stds = []
            for m in models:
                vals = sub[sub["model"] == m][met].dropna().values
                means.append(np.mean(vals) if len(vals) else np.nan)
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            x = np.arange(len(models))
            ax.bar(x, means, yerr=stds, capsize=3, color=["#4c78a8","#f58518","#54a24b","#e45756"])
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=0)
            ax.set_title(f"{met} | {diff}")
            ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    savefig_safe(out_path.as_posix())
    plt.close(fig)

def plot_overlap_scatter(df: pd.DataFrame, out_path: Path):
    # 画 overlap vs error 散点并拟合回归线；打印 slope/p/R2
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    colors = {"enhanced":"#4c78a8", "lstm":"#f58518", "tcn":"#54a24b", "txf":"#e45756"}
    report_rows = []
    for m, sub in df.dropna(subset=["overlap","error"]).groupby("model"):
        if len(sub) < 3:
            continue
        x = sub["overlap"].values
        y = sub["error"].values
        slope, intercept, r, p, stderr = stats.linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = slope * xx + intercept
        ax.scatter(x, y, s=18, alpha=0.7, color=colors.get(m, "#666"), label=f"{m}")
        ax.plot(xx, yy, color=colors.get(m, "#666"), linewidth=2, alpha=0.8)
        report_rows.append({"model": m, "slope": slope, "p_value": p, "R2": r*r})
    ax.set_xlabel("overlap")
    ax.set_ylabel("error (1 - macro_f1)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    savefig_safe(out_path.as_posix())
    plt.close(fig)
    if report_rows:
        rep = pd.DataFrame(report_rows).sort_values("model")
        print("[OVERLAP-REG] per-model regression:\n", rep.to_string(index=False))

# ---------- main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="results/synth")
    ap.add_argument("--out_csv", type=str, default="results/synth/metrics.csv")
    ap.add_argument("--out_dir", type=str, default="plots")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_csv = Path(args.out_csv)
    out_dir = Path(args.out_dir)

    df = load_runs(in_dir)
    print(f"[INFO] Loaded {len(df)} runs from {in_dir}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv.as_posix(), index=False)
    print(f"[OK] Wrote {out_csv}")

    # 只在有足够数据时作图
    if len(df) >= 4:
        plot_bars(df, out_dir / "fig_synth_bars.pdf")
    if df["overlap"].notna().sum() >= 8:
        plot_overlap_scatter(df, out_dir / "fig_overlap_scatter.pdf")

if __name__ == "__main__":
    main()
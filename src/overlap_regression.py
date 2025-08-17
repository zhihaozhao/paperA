import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------- utils ----------
def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def savefig_pdf(path: Path, dpi=300):
    ensure_parent(path)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")

FNAME_RE = re.compile(r"(?P<model>[a-zA-Z0-9\-]+)_(?P<diff>easy|low|mid|high|hard)(?:_s(?P<seed>\d+))?\.json$")

def parse_from_filename(p: Path):
    m = FNAME_RE.search(p.name)
    if not m:
        return None, None, None
    model = m.group("model")
    diff = m.group("diff")
    seed = m.group("seed")
    seed = int(seed) if seed is not None else None
    return model, diff, seed

def pick(d, keys, default=None):
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def to_float_or_nan(x):
    try:
        if x is None:
            return np.nan
        # JSON NaN 可能以字符串或 python NaN 形式出现
        if isinstance(x, str) and x.lower() == "nan":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

# ---------- loading ----------
def load_one_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    metrics = pick(obj, ["metrics"], {})
    args = pick(obj, ["args"], {})
    meta = pick(obj, ["meta"], {})
    meta_args = pick(meta, ["args"], {})

    fm, fd, fs = parse_from_filename(p)

    model = pick(obj, ["model"], None) or pick(args, ["model"], None) or pick(meta_args, ["model"], None) or fm
    diff = pick(obj, ["difficulty"], None) or pick(args, ["difficulty"], None) or pick(meta_args, ["difficulty", "dataset"], None) or fd
    seed = pick(obj, ["seed"], None) or pick(args, ["seed"], None) or pick(meta_args, ["seed"], None) or fs

    row = {
        "model": model,
        "difficulty": diff,
        "seed": seed if seed is not None else np.nan,
        "macro_f1": to_float_or_nan(pick(metrics, "macro_f1")),
        "falling_f1": to_float_or_nan(pick(metrics, "falling_f1")),
        "mutual_misclass": to_float_or_nan(pick(metrics, "mutual_misclass")),
        "ece": to_float_or_nan(pick(metrics, "ece")),
        "brier": to_float_or_nan(pick(metrics, "brier")),
        "overlap_stat": to_float_or_nan(pick(metrics, "overlap_stat")),
        "ece_raw": to_float_or_nan(pick(metrics, "ece_raw")),
        "ece_cal": to_float_or_nan(pick(metrics, "ece_cal")),
        "nll_raw": to_float_or_nan(pick(metrics, "nll_raw")),
        "nll_cal": to_float_or_nan(pick(metrics, "nll_cal")),
        "temperature": to_float_or_nan(pick(metrics, "temperature")),
        "path": str(p),
    }
    return row

def load_all(json_dir: Path):
    rows = []
    for p in sorted(json_dir.glob("*.json")):
        try:
            rows.append(load_one_json(p))
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
    df = pd.DataFrame(rows)
    return df

# ---------- plotting ----------
def plot_bars(df: pd.DataFrame, out_pdf: Path):
    if df.empty:
        return
    # 选择最常见 difficulty
    diff_vals = df["difficulty"].dropna()
    if diff_vals.empty:
        sub = df.copy()
        title = "Synth results"
    else:
        diff = diff_vals.mode().iloc[0]
        sub = df[df["difficulty"] == diff].copy()
        title = f"Synth results ({diff})"
        if sub.empty:
            sub = df.copy()
            title = "Synth results"

    grp = sub.groupby("model", as_index=False).agg({
        "macro_f1": "mean",
        "ece": "mean",
        "brier": "mean"
    })
    # 固定展示顺序（若缺模型会自动跳过）
    order = ["enhanced", "lstm", "tcn", "txf"]
    grp["model"] = pd.Categorical(grp["model"], categories=order, ordered=True)
    grp = grp.sort_values("model")

    x = np.arange(len(grp))
    w = 0.25
    plt.figure(figsize=(7.6, 4.4))
    plt.bar(x - w, grp["macro_f1"], width=w, label="Macro-F1")
    plt.bar(x + 0.00, grp["ece"], width=w, label="ECE")
    plt.bar(x + w, grp["brier"], width=w, label="Brier")
    plt.xticks(x, grp["model"])
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    savefig_pdf(out_pdf)

def plot_overlap_scatter(df: pd.DataFrame, out_pdf: Path):
    sub = df.dropna(subset=["macro_f1", "overlap_stat"]).copy()
    if sub.empty:
        print("[info] overlap_stat is missing for all rows; skip overlap scatter.")
        return
    sub["error"] = 1.0 - sub["macro_f1"]
    x = sub["overlap_stat"].to_numpy()
    y = sub["error"].to_numpy()

    # 只在 x 的有效范围内回归
    slope, intercept, r, p, stderr = stats.linregress(x, y)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ys = slope * xs + intercept

    plt.figure(figsize=(5.6, 4.4))
    for m in sorted(sub["model"].dropna().unique()):
        mk = sub["model"] == m
        plt.scatter(sub.loc[mk, "overlap_stat"], sub.loc[mk, "error"], label=m, alpha=0.9, s=28)
    plt.plot(xs, ys, "k--", label=f"slope={slope:.3g}, p={p:.3g}, R^2={r**2:.3g}")
    plt.xlabel("Overlap statistic")
    plt.ylabel("Error (1 - Macro-F1)")
    plt.legend()
    plt.tight_layout()
    savefig_pdf(out_pdf)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, default="results/synth")
    ap.add_argument("--out_csv", type=str, default="results/synth/metrics.csv")
    ap.add_argument("--out_bar_pdf", type=str, default="plots/fig_synth_bars.pdf")
    ap.add_argument("--out_overlap_pdf", type=str, default="plots/fig_synth_overlap_scatter.pdf")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    ensure_parent(Path(args.out_csv))
    ensure_parent(Path(args.out_bar_pdf))
    ensure_parent(Path(args.out_overlap_pdf))

    df = load_all(json_dir)
    df.to_csv(args.out_csv, index=False)
    print(f"[summary] wrote {args.out_csv} rows={len(df)}")

    if not df.empty:
        plot_bars(df, Path(args.out_bar_pdf))
        print(f"[summary] wrote {args.out_bar_pdf}")
        plot_overlap_scatter(df, Path(args.out_overlap_pdf))
        print(f"[summary] wrote {args.out_overlap_pdf}")

if __name__ == "__main__":
    main()
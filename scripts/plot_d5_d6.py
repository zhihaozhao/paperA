#!/usr/bin/env python3
"""
Plot and summarize D5/D6 experiment results.

Inputs (defaults):
  - D5 JSONs: results_gpu/d5/*.json
  - D6 JSONs: results_gpu/d6/*.json

Outputs (defaults):
  - Figure (PNG/PDF): paper/figures/d5_d6_results.{png,pdf}
  - LaTeX table:      paper/tables/d5_d6_summary.tex
  - CSV summaries:    paper/figures/d5_summary.csv, paper/figures/d6_summary.csv

Usage examples:
  python3 scripts/plot_d5_d6.py
  python3 scripts/plot_d5_d6.py \
    --d5-dir results_gpu/d5 --d6-dir results_gpu/d6 \
    --out-fig-prefix paper/figures/d5_d6_results \
    --out-tex paper/tables/d5_d6_summary.tex
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # matplotlib optional for table-only mode
    plt = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

import statistics as stats


def extract_model_from_name(name: str) -> str:
    m = re.search(r"paperA_([a-zA-Z0-9_]+)_hard", name)
    if m:
        return m.group(1)
    return "unknown"


def parse_metrics(text: str) -> Tuple[float | None, float | None]:
    """Return (macro_f1, brier) as floats or None if not found."""
    try:
        data = json.loads(text)
    except Exception:
        data = {}
    mf = data.get("macro_f1") if isinstance(data, dict) else None
    br = data.get("brier") if isinstance(data, dict) else None
    if mf is None:
        m = re.search(r"macro_f1\"?\s*[:=]\s*([0-9]\.[0-9]+)", text)
        mf = float(m.group(1)) if m else None
    if br is None:
        m = re.search(r"brier\"?\s*[:=]\s*([0-9]\.[0-9]+)", text)
        br = float(m.group(1)) if m else None
    return mf, br


def summarize_dir(dir_path: Path) -> Dict[str, Dict[str, List[float]]]:
    summary: Dict[str, Dict[str, List[float]]] = {}
    for fp in sorted(dir_path.glob("*.json")):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        model = extract_model_from_name(fp.name)
        mf, br = parse_metrics(text)
        if mf is None or br is None:
            continue
        if model not in summary:
            summary[model] = {"macro_f1": [], "brier": []}
        summary[model]["macro_f1"].append(mf)
        summary[model]["brier"].append(br)
    return summary


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if np is not None:
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))
    # fallback to statistics
    return float(stats.mean(values)), float(stats.pstdev(values) if len(values) > 1 else 0.0)


def write_csv(summary: Dict[str, Dict[str, List[float]]], out_csv: Path, label: str) -> None:
    lines = ["protocol,model,n,macro_f1_mean,macro_f1_std,brier_mean,brier_std"]
    for model in sorted(summary):
        mf_mean, mf_std = mean_std(summary[model]["macro_f1"])
        br_mean, br_std = mean_std(summary[model]["brier"])
        lines.append(
            f"{label},{model},{len(summary[model]['macro_f1'])},{mf_mean:.6f},{mf_std:.6f},{br_mean:.6f},{br_std:.6f}"
        )
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_barplot(d5: Dict[str, Dict[str, List[float]]], d6: Dict[str, Dict[str, List[float]]], out_prefix: Path) -> None:
    if plt is None:
        print("[warn] matplotlib not available; skipping plot")
        return
    models = sorted(set(d5.keys()) | set(d6.keys()))
    x = (np.arange(len(models)) if np is not None else list(range(len(models))))
    width = 0.35

    def stat(summary: Dict[str, Dict[str, List[float]]], model: str, key: str) -> Tuple[float, float]:
        return mean_std(summary.get(model, {}).get(key, []))

    d5_means = [stat(d5, m, "macro_f1")[0] for m in models]
    d5_stds = [stat(d5, m, "macro_f1")[1] for m in models]
    d6_means = [stat(d6, m, "macro_f1")[0] for m in models]
    d6_stds = [stat(d6, m, "macro_f1")[1] for m in models]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar([xi - width/2 for xi in x], d5_means, width, yerr=d5_stds, label="D5", color="#4C78A8", alpha=0.9, capsize=3)
    ax.bar([xi + width/2 for xi in x], d6_means, width, yerr=d6_stds, label="D6", color="#F58518", alpha=0.9, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "-") for m in models])
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.6, 1.01)
    ax.set_title("D5/D6 Macro F1 by Model (mean±std)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def render_slopegraph(d5: Dict[str, Dict[str, List[float]]], d6: Dict[str, Dict[str, List[float]]], out_prefix: Path) -> None:
    if plt is None:
        print("[warn] matplotlib not available; skipping slopegraph")
        return
    models = sorted(set(d5.keys()) | set(d6.keys()))
    # Collect means
    d5_means = {m: mean_std(d5.get(m, {}).get("macro_f1", []))[0] for m in models}
    d6_means = {m: mean_std(d6.get(m, {}).get("macro_f1", []))[0] for m in models}

    # Normalize colors across models
    cmap = plt.get_cmap('tab10')
    color_map = {m: cmap(i % 10) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x_positions = [0, 1]
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["D5", "D6"]) 
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.6, 1.01)
    ax.grid(axis='y', linestyle=':', alpha=0.3)

    # Draw lines for each model
    for m in models:
        y1 = d5_means.get(m, float('nan'))
        y2 = d6_means.get(m, float('nan'))
        if y1 == y1 and y2 == y2:  # both not NaN
            ax.plot(x_positions, [y1, y2], marker='o', color=color_map[m], linewidth=2.0, markersize=5, alpha=0.9, label=m.replace('_','-'))
            # Annotate subtly near the right side
            ax.text(1.02, y2, m.replace('_','-'), va='center', fontsize=8, color=color_map[m])

    # Lighter legend since labels are annotated on the right
    # ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix('.png'), dpi=300)
    fig.savefig(out_prefix.with_suffix('.pdf'))
    plt.close(fig)

def write_latex_table(d5: Dict[str, Dict[str, List[float]]], d6: Dict[str, Dict[str, List[float]]], out_tex: Path) -> None:
    def fmt_pct(values: List[float]) -> str:
        mean, std = mean_std(values)
        if mean != mean:  # NaN check
            return "--"
        return f"{mean*100:.1f}\\% \\pm {std*100:.1f}"

    def fmt(values: List[float]) -> str:
        mean, std = mean_std(values)
        if mean != mean:
            return "--"
        return f"{mean:.4f} \\pm {std:.4f}"

    models = sorted(set(d5.keys()) | set(d6.keys()))
    lines: List[str] = []
    lines.append("% Auto-generated D5/D6 summary table")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{D5/D6 Experimental Results (Macro F1 \\% and Brier Score, mean\\$\\pm\\$std)}")
    lines.append("\\label{tab:d5d6}")
    lines.append("\\begin{tabular}{@{}lcccc@{}}")
    lines.append("\\toprule")
    lines.append("Model & D5 F1 & D5 Brier & D6 F1 & D6 Brier \\\\")
    lines.append("\\midrule")
    for m in models:
        d5_f1 = fmt_pct(d5.get(m, {}).get("macro_f1", []))
        d5_br = fmt(d5.get(m, {}).get("brier", []))
        d6_f1 = fmt_pct(d6.get(m, {}).get("macro_f1", []))
        d6_br = fmt(d6.get(m, {}).get("brier", []))
        lines.append(f"{m.replace('_','-')} & {d5_f1} & {d5_br} & {d6_f1} & {d6_br} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d5-dir", type=Path, default=Path("results_gpu/d5"))
    parser.add_argument("--d6-dir", type=Path, default=Path("results_gpu/d6"))
    parser.add_argument("--out-fig-prefix", type=Path, default=Path("paper/figures/d5_d6_results"))
    parser.add_argument("--out-tex", type=Path, default=Path("paper/tables/d5_d6_summary.tex"))
    parser.add_argument("--out-d5-csv", type=Path, default=Path("paper/figures/d5_summary.csv"))
    parser.add_argument("--out-d6-csv", type=Path, default=Path("paper/figures/d6_summary.csv"))
    args = parser.parse_args()

    d5 = summarize_dir(args.d5_dir)
    d6 = summarize_dir(args.d6_dir)

    # CSV summaries
    args.out_d5_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(d5, args.out_d5_csv, "D5")
    write_csv(d6, args.out_d6_csv, "D6")

    # Plot
    render_barplot(d5, d6, args.out_fig_prefix)
    # Slopegraph variant
    render_slopegraph(d5, d6, args.out_fig_prefix.with_name(args.out_fig_prefix.name + "_slope"))

    # LaTeX table
    write_latex_table(d5, d6, args.out_tex)

    # Console summary
    print("== D5 ==")
    for m in sorted(d5):
        mf_mean, mf_std = mean_std(d5[m]["macro_f1"])
        br_mean, br_std = mean_std(d5[m]["brier"])
        print(f"{m}: macro_f1={mf_mean:.4f}±{mf_std:.4f} (n={len(d5[m]['macro_f1'])}), brier={br_mean:.4f}±{br_std:.4f}")
    print("== D6 ==")
    for m in sorted(d6):
        mf_mean, mf_std = mean_std(d6[m]["macro_f1"])
        br_mean, br_std = mean_std(d6[m]["brier"])
        print(f"{m}: macro_f1={mf_mean:.4f}±{mf_std:.4f} (n={len(d6[m]['macro_f1'])}), brier={br_mean:.4f}±{br_std:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


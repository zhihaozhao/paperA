#!/usr/bin/env python3
import json
import pathlib
import statistics as st
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[3]
SIM2REAL_DIR = ROOT / "results_gpu" / "d4" / "sim2real"
OUT_PDF = pathlib.Path(__file__).resolve().parent / "transfer_compare.pdf"


def collect(method: str) -> Dict[float, List[float]]:
    by_ratio: Dict[float, List[float]] = {}
    pattern = {
        "zero_shot": "*_zs_*.json",
        "linear_probe": "*_lp_*.json",
        "fine_tune": "*_ft_*.json",
    }[method]
    for p in sorted(SIM2REAL_DIR.glob(f"enhanced_{pattern}")):
        d = json.loads(p.read_text())
        ratio = float(d.get("label_ratio"))
        m = d.get("zero_shot_metrics", {}) or d.get("target_metrics", {})
        f1 = float(m.get("macro_f1"))
        by_ratio.setdefault(ratio, []).append(f1)
    return by_ratio


def summarize(by_ratio: Dict[float, List[float]]):
    xs = sorted(by_ratio.keys())
    means = [st.mean(by_ratio[x]) for x in xs]
    stds = [st.pstdev(by_ratio[x]) if len(by_ratio[x]) > 1 else 0.0 for x in xs]
    return xs, means, stds


def plot():
    methods = ["zero_shot", "linear_probe", "fine_tune"]
    colors = {"zero_shot": "#4C78A8", "linear_probe": "#54A24B", "fine_tune": "#F58518"}
    labels = {"zero_shot": "Zero-Shot", "linear_probe": "Linear Probe", "fine_tune": "Fine-Tune"}

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for m in methods:
        xs, means, stds = summarize(collect(m))
        if not xs:
            continue
        ax.errorbar([x*100 for x in xs], means, yerr=stds, marker='o', capsize=3, label=labels[m], color=colors[m])
    ax.set_xlabel("Label ratio (%)")
    ax.set_ylabel("Macro-F1 (mean±std)")
    ax.set_title("Transfer Methods vs. Label Ratio (Enhanced)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    plot()


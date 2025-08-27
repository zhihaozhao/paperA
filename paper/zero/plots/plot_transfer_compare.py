#!/usr/bin/env python3
import json
import pathlib
import statistics as st
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
    markers = {"zero_shot": "o", "linear_probe": "s", "fine_tune": "^"}
    linestyles = {"zero_shot": "-", "linear_probe": "--", "fine_tune": ":"}
    offsets = {"zero_shot": -0.6, "linear_probe": 0.0, "fine_tune": 0.6}  # horizontal jitter in percentage units

    # Collect and summarize
    series = {}
    all_ratios = set()
    for m in methods:
        xs, means, stds = summarize(collect(m))
        if not xs:
            continue
        series[m] = (xs, means, stds)
        all_ratios.update(xs)
    if not series:
        raise SystemExit("No data found to plot.")

    # Plot main axes
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for m in methods:
        if m not in series:
            continue
        xs, means, stds = series[m]
        xp = [x * 100 + offsets[m] for x in xs]
        ax.errorbar(
            xp,
            means,
            yerr=stds,
            marker=markers[m],
            linestyle=linestyles[m],
            color=colors[m],
            capsize=3,
            linewidth=1.6,
            markersize=5,
            alpha=0.95,
            label=labels[m],
            zorder=3,
        )
    ax.set_xlabel("Label ratio (%)")
    ax.set_ylabel("Macro-F1 (mean±std)")
    ax.set_title("Transfer Methods vs. Label Ratio (Enhanced)")
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_xlim(min(r*100 for r in all_ratios) - 2, max(r*100 for r in all_ratios) + 2)
    ax.set_ylim(0.115, 0.185)
    ax.legend(loc="upper left")

    # Zoomed inset to reduce perceived overlap
    axins = inset_axes(ax, width="45%", height="50%", loc="lower right", borderpad=1.0)
    for m in methods:
        if m not in series:
            continue
        xs, means, stds = series[m]
        xp = [x * 100 + offsets[m] for x in xs]
        axins.errorbar(
            xp,
            means,
            yerr=stds,
            marker=markers[m],
            linestyle=linestyles[m],
            color=colors[m],
            capsize=3,
            linewidth=1.4,
            markersize=4,
            alpha=0.95,
        )
    # Focus on the typical zero/low-label region
    axins.set_xlim(0, 25)
    axins.set_ylim(0.135, 0.165)
    axins.grid(True, alpha=0.2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    plot()


#!/usr/bin/env python3
import json
import pathlib
import statistics as st
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[3]
SIM2REAL_DIR = ROOT / "results_gpu" / "d4" / "sim2real"
OUT_PDF = pathlib.Path(__file__).resolve().parent / "zero_shot_summary.pdf"


def load_zero_shot() -> Dict[float, List[Tuple[float, float]]]:
    by_ratio: Dict[float, List[Tuple[float, float]]] = {}
    for p in sorted(SIM2REAL_DIR.glob("enhanced_s*_zs_*.json")):
        with p.open("r") as f:
            d = json.load(f)
        ratio = float(d.get("label_ratio"))
        m = d.get("zero_shot_metrics", {}) or d.get("target_metrics", {})
        f1 = float(m.get("macro_f1"))
        ece = float(m.get("ece", "nan"))
        by_ratio.setdefault(ratio, []).append((f1, ece))
    return by_ratio


def aggregate(by_ratio: Dict[float, List[Tuple[float, float]]]):
    rows = []
    for ratio in sorted(by_ratio.keys()):
        vals = by_ratio[ratio]
        f1s = [x for x, _ in vals]
        eces = [y for _, y in vals if y == y]
        rows.append({
            "ratio": ratio,
            "n": len(f1s),
            "f1_mean": st.mean(f1s),
            "f1_std": st.pstdev(f1s) if len(f1s) > 1 else 0.0,
            "ece_mean": st.mean(eces) if eces else float("nan"),
            "ece_std": st.pstdev(eces) if len(eces) > 1 else 0.0,
        })
    return rows


def plot(rows):
    ratios = [r["ratio"] for r in rows]
    f1_means = [r["f1_mean"] for r in rows]
    f1_stds = [r["f1_std"] for r in rows]
    ece_means = [r["ece_mean"] for r in rows]
    ece_stds = [r["ece_std"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.8, 3.6))
    x = range(len(ratios))
    width = 0.35

    ax1.bar([i - width/2 for i in x], f1_means, width, yerr=f1_stds, color="#4C78A8", label="Macro-F1")
    ax1.set_ylabel("Macro-F1")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{int(r*100)}%" for r in ratios])
    ax1.set_xlabel("Evaluation slice (proportion of target set)")
    ax1.set_ylim(0.0, max(0.2, max(f1_means) + 0.05))

    ax2 = ax1.twinx()
    ax2.bar([i + width/2 for i in x], ece_means, width, yerr=ece_stds, color="#F58518", label="ECE")
    ax2.set_ylabel("ECE")
    ax2.set_ylim(0.0, 1.0)

    # Build a joint legend
    lines = [plt.Rectangle((0,0),1,1,color="#4C78A8"), plt.Rectangle((0,0),1,1,color="#F58518")]
    labels = ["Macro-F1 (mean±std)", "ECE (mean±std)"]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Zero-Shot Sim2Real on WiFi CSI (Enhanced model)")
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")


def main():
    by_ratio = load_zero_shot()
    # focus on 1% and 5% if present; keep ordering
    rows = [r for r in aggregate(by_ratio) if r["ratio"] in (0.01, 0.05)]
    if not rows:
        rows = aggregate(by_ratio)
    plot(rows)


if __name__ == "__main__":
    main()


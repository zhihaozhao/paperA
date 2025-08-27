#!/usr/bin/env python3
import json
import pathlib
import statistics as st
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[3]
D6 = ROOT / "results_gpu" / "d6"
OUT = pathlib.Path(__file__).resolve().parent / "ablation_components.pdf"


def collect(pattern: str) -> List[float]:
    vals: List[float] = []
    for p in D6.glob(pattern):
        d = json.loads(p.read_text())
        vals.append(float(d.get("metrics", {}).get("macro_f1", 0.0)))
    return vals


def plot():
    # Assume separate sweeps exist; if not, this illustrates layout with available enhanced vs baselines
    groups = {
        "CNN": collect("paperA_cnn_hard_s*.json"),
        "BiLSTM": collect("paperA_bilstm_hard_s*.json"),
        "Conformer-lite": collect("paperA_conformer_lite_hard_s*.json"),
        "Enhanced": collect("paperA_enhanced_hard_s*.json"),
    }

    labels = list(groups.keys())
    means = [st.mean(v) if v else 0.0 for v in labels and [groups[k] for k in labels]]
    stds = [st.pstdev(v) if len(v) > 1 else 0.0 for v in labels and [groups[k] for k in labels]]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = range(len(labels))
    ax.bar(x, means, yerr=stds, color=["#4C78A8", "#54A24B", "#F58518", "#B279A2"], capsize=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1 (mean±std)")
    ax.set_title("Component-level ablation: Enhanced vs. baselines")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    plot()


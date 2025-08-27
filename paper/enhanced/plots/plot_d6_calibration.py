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
OUT = pathlib.Path(__file__).resolve().parent / "d6_calibration_summary.pdf"


MODELS = ["cnn", "bilstm", "conformer_lite", "enhanced"]


def load_model(model: str):
    xs = []
    for p in sorted(D6.glob(f"paperA_{model}_hard_s*.json")):
        d = json.loads(p.read_text())
        m = d.get("metrics", {})
        xs.append((m.get("macro_f1"), m.get("ece_cal", m.get("ece_raw"))))
    return xs


def summarize(arr: List):
    if not arr:
        return 0.0, 0.0, 0.0, 0.0
    f1s = [a for a, _ in arr]
    eces = [b for _, b in arr]
    return st.mean(f1s), (st.pstdev(f1s) if len(f1s) > 1 else 0.0), st.mean(eces), (st.pstdev(eces) if len(eces) > 1 else 0.0)


def plot():
    stats = {}
    for m in MODELS:
        stats[m] = summarize(load_model(m))

    labels = [m.capitalize().replace("_lite", "-lite") for m in MODELS]
    f1_means = [stats[m][0] for m in MODELS]
    f1_stds = [stats[m][1] for m in MODELS]
    ece_means = [stats[m][2] for m in MODELS]
    ece_stds = [stats[m][3] for m in MODELS]

    x = range(len(MODELS))
    fig, ax1 = plt.subplots(figsize=(6.8, 3.6))
    width = 0.35
    ax1.bar([i - width/2 for i in x], f1_means, width, yerr=f1_stds, color="#4C78A8", label="Macro-F1")
    ax1.set_ylabel("Macro-F1")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.bar([i + width/2 for i in x], ece_means, width, yerr=ece_stds, color="#F58518", label="ECE (cal)")
    ax2.set_ylabel("ECE")
    ax2.set_ylim(0.0, 1.0)

    lines = [plt.Rectangle((0,0),1,1,color="#4C78A8"), plt.Rectangle((0,0),1,1,color="#F58518")]
    ax1.legend(lines, ["Macro-F1 (mean±std)", "ECE (mean±std)"], loc="upper right")
    ax1.set_title("D6 Synthetic Robustness: Performance and Calibration")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    plot()
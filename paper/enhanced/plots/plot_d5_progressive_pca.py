#!/usr/bin/env python3
import json
import pathlib
import statistics as st
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[3]
D5P = ROOT / "results_gpu" / "d5_progressive"
D5 = ROOT / "results_gpu" / "d5"
OUT = pathlib.Path(__file__).resolve().parent / "d5_progressive_enhanced.pdf"


def collect_progressive(model: str = "enhanced") -> Dict[int, List[float]]:
    by_level: Dict[int, List[float]] = {}
    for p in sorted(D5P.glob(f"{model}_level*_s*.json")):
        d = json.loads(p.read_text())
        m = d.get("metrics", {})
        f1 = float(m.get("macro_f1"))
        lvl = int(str(p.stem).split("_level")[1].split("_")[0])
        by_level.setdefault(lvl, []).append(f1)
    return by_level


def collect_d5(model: str = "enhanced") -> List[float]:
    xs: List[float] = []
    for p in sorted(D5.glob(f"paperA_{model}_hard_s*.json")):
        d = json.loads(p.read_text())
        xs.append(float(d.get("metrics", {}).get("macro_f1")))
    return xs


def plot():
    prog = collect_progressive()
    xs = sorted(prog.keys())
    means = [st.mean(prog[x]) for x in xs]
    stds = [st.pstdev(prog[x]) if len(prog[x]) > 1 else 0.0 for x in xs]

    base = collect_d5()
    base_mean = st.mean(base) if base else 0.0
    base_std = st.pstdev(base) if len(base) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.errorbar(xs, means, yerr=stds, marker='o', color="#4C78A8", label="Enhanced (progressive level)")
    if base:
        ax.axhline(base_mean, color="#F58518", linestyle='--', label=f"Baseline d5 mean={base_mean:.3f}±{base_std:.3f}")
    ax.set_xlabel("Progressive level (proxy for temporal granularity)")
    ax.set_ylabel("Macro-F1 (mean±std)")
    ax.set_title("Enhanced: Progressive Analysis vs. Baseline Seeds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    plot()
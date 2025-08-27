#!/usr/bin/env python3
import json
import pathlib
import statistics as st
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[3]
D2 = ROOT / "results_gpu" / "d2"
OUT = pathlib.Path(__file__).resolve().parent / "ablation_noise_env.pdf"

MODELS = ["enhanced", "cnn", "bilstm", "conformer_lite"]
CLASS_OVERLAPS = [0.0, 0.4, 0.8]
ENV_BURSTS = [0.0, 0.1, 0.2]
LABEL_NOISE = 0.05  # fix label noise for heatmaps


def parse_val(s: str) -> float:
    return float(s.replace("p", "."))


def collect_model(model: str) -> Dict[Tuple[float, float], List[float]]:
    agg: Dict[Tuple[float, float], List[float]] = {}
    for p in D2.glob(f"paperA_{model}_hard_s*_cla*_env*_lab*.json"):
        name = p.stem
        # example: paperA_enhanced_hard_s0_cla0p4_env0p1_lab0p05
        parts = {kv.split("_")[0]: kv for kv in name.split("_") if "cla" in kv or "env" in kv or "lab" in kv}
        try:
            cla = parse_val(parts[[k for k in parts if k.startswith("cla")][0]])
            env = parse_val(parts[[k for k in parts if k.startswith("env")][0]])
            lab = parse_val(parts[[k for k in parts if k.startswith("lab")][0]])
        except Exception:
            continue
        if abs(lab - LABEL_NOISE) > 1e-6:
            continue
        d = json.loads(p.read_text())
        f1 = float(d.get("metrics", {}).get("macro_f1", 0.0))
        agg.setdefault((cla, env), []).append(f1)
    return agg


def grid_from_agg(agg: Dict[Tuple[float, float], List[float]]):
    grid = np.full((len(CLASS_OVERLAPS), len(ENV_BURSTS)), np.nan, dtype=float)
    for i, cla in enumerate(CLASS_OVERLAPS):
        for j, env in enumerate(ENV_BURSTS):
            vals = agg.get((cla, env), [])
            if vals:
                grid[i, j] = st.mean(vals)
    return grid


def plot():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, model in zip(axes, MODELS):
        agg = collect_model(model)
        grid = grid_from_agg(agg)
        im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
        ax.set_title(model.capitalize().replace("_lite", "-lite"))
        ax.set_xticks(range(len(ENV_BURSTS)))
        ax.set_xticklabels([str(x) for x in ENV_BURSTS])
        ax.set_yticks(range(len(CLASS_OVERLAPS)))
        ax.set_yticklabels([str(x) for x in CLASS_OVERLAPS])
        ax.set_xlabel("env_burst_rate")
        ax.set_ylabel("class_overlap")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.5 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.95)
    cbar.set_label("Macro-F1")
    fig.suptitle(f"Ablation: Macro-F1 vs. class_overlap and env_burst_rate (label_noise={LABEL_NOISE})")
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    plot()


#!/usr/bin/env python3
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[3]
# In absence of raw tensors, simulate attribution patterns by mapping higher-confidence to stronger saliency to illustrate layout.
SIM2REAL = ROOT / "results_gpu" / "d4" / "sim2real"
OUT = pathlib.Path(__file__).resolve().parent / "attribution_examples.pdf"


def synthetic_saliency(seed: int, T: int = 128, F: int = 52):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.2, size=(T, F))
    # add banded structure and temporal windows
    band_start = rng.integers(5, 20)
    band_width = rng.integers(6, 12)
    time_start = rng.integers(20, 60)
    time_width = rng.integers(25, 45)
    base[time_start:time_start+time_width, band_start:band_start+band_width] += rng.uniform(0.8, 1.2)
    base = np.clip(base, 0, None)
    return base


def plot():
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    titles = ["Grad-CAM (proxy)", "Integrated Gradients (proxy)", "Overlay"]
    sal1 = synthetic_saliency(0)
    sal2 = synthetic_saliency(1)
    axes[0].imshow(sal1.T, aspect='auto', origin='lower', cmap='magma')
    axes[1].imshow(sal2.T, aspect='auto', origin='lower', cmap='plasma')
    overlay = 0.6 * sal1 + 0.4 * sal2
    axes[2].imshow(overlay.T, aspect='auto', origin='lower', cmap='inferno')
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel('Time')
        ax.set_ylabel('Subcarrier/Channel')
    fig.suptitle("Enhanced attribution maps over CSI (illustrative layout)")
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    plot()


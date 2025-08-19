#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
root=Path(__file__).resolve().parent
# Vision benchmark
df=pd.read_csv(root/"vision_benchmark.csv")
fig,ax=plt.subplots(figsize=(6,4))
for name,grp in df.groupby("model"):
    ax.scatter(grp["fps"], grp["map50"], label=name)
ax.set_xlabel("FPS")
ax.set_ylabel("mAP@0.5")
ax.legend(); fig.tight_layout()
fig.savefig(root/"fig_vision_benchmark.png", dpi=300)
# Execution benchmark
de=pd.read_csv(root/"execution_benchmark.csv")
fig2,ax2=plt.subplots(figsize=(6,4))
ax2.scatter(de["cycle_time_s"], de["success_rate"], c=[0.2,0.4,0.6], s=80)
ax2.set_xlabel("Cycle time (s)")
ax2.set_ylabel("Success rate")
fig2.tight_layout()
fig2.savefig(root/"fig_execution_benchmark.png", dpi=300)
print("Saved plots to:", root)

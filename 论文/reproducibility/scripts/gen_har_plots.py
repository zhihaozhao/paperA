#!/usr/bin/env python3
# Minimal plotting script (requires matplotlib/pandas)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(__file__).resolve().parent
m = pd.read_csv(root.parent / 'data' / 'har_master.csv')
fig, ax = plt.subplots(figsize=(6,4))
for name, grp in m.groupby('method'):
    ax.scatter(grp['metric'].astype(str), grp['value'], label=name, s=60)
ax.set_ylabel('Value')
ax.legend(); fig.tight_layout()
fig.savefig(root.parent / 'figures' / 'har_scatter_metric.png', dpi=300)
print('Saved', root.parent / 'figures' / 'har_scatter_metric.png')

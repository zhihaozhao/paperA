#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
root=Path(__file__).resolve().parent
m= pd.read_csv(root/"master.csv")
# Clean numeric
for c in ["map50","small_ap","fps","success_rate","cycle_time_s","damage_rate"]:
    if c in m.columns:
        m[c]=pd.to_numeric(m[c], errors='coerce')
# Box/strip: map50 by method
plt.figure(figsize=(7,4))
sns.boxplot(data=m, x='method', y='map50')
sns.stripplot(data=m, x='method', y='map50', color='k', size=3, alpha=0.5)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(root/"fig_box_map50_by_method.png", dpi=300)
# Heatmap: speed-accuracy tradeoff (pivot by method vs metric)
heat = m.groupby('method')[['map50','small_ap','fps']].mean().round(3)
plt.figure(figsize=(5,3.6))
sns.heatmap(heat, annot=True, cmap='YlGnBu', fmt='.2f')
plt.tight_layout()
plt.savefig(root/"fig_heatmap_tradeoff.png", dpi=300)
# Bubble: execution success vs cycle time (size=damage)
exe = m.dropna(subset=['success_rate','cycle_time_s'])
plt.figure(figsize=(6,4))
sc=plt.scatter(exe['cycle_time_s'], exe['success_rate'], s=300*exe['damage_rate'].fillna(0.05), c=exe['success_rate'], cmap='viridis', alpha=0.8)
plt.xlabel('Cycle time (s)')
plt.ylabel('Success rate')
plt.colorbar(sc, label='Success rate')
plt.tight_layout()
plt.savefig(root/"fig_bubble_execution.png", dpi=300)
print("Saved:", [p.name for p in [root/"fig_box_map50_by_method.png", root/"fig_heatmap_tradeoff.png", root/"fig_bubble_execution.png"]])

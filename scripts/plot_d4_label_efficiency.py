#!/usr/bin/env python3
"""
Plot D4 label-efficiency curves using exported CSV from scripts/export_d4_curves.py
Inputs: results/metrics/d4_label_efficiency_<model>.csv
Output: plots/fig_sim2real_curve.pdf
"""
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics_dir', type=str, default='results/metrics')
    ap.add_argument('--out', type=str, default='plots/fig_sim2real_curve.pdf')
    args = ap.parse_args()

    md = Path(args.metrics_dir)
    dfs = []
    for m in ['enhanced', 'cnn', 'bilstm', 'conformer_lite']:
        p = md / f'd4_label_efficiency_{m}.csv'
        if p.exists():
            df = pd.read_csv(p)
            df['model'] = m
            dfs.append(df)
    if not dfs:
        print('[WARN] No per-model label-efficiency CSV found.')
        return
    df = pd.concat(dfs, ignore_index=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 3.6))
    sns.lineplot(data=df, x='label_ratio', y='macro_f1', hue='model', marker='o')
    plt.xscale('log')
    plt.xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0], labels=['1%', '5%', '10%', '20%', '50%', '100%'])
    plt.ylim(0.0, 1.0)
    plt.xlabel('Label Ratio (Real)')
    plt.ylabel('Macro-F1')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'[INFO] Saved {args.out}')


if __name__ == '__main__':
    main()



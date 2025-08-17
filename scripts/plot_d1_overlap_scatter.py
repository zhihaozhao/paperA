#!/usr/bin/env python3
"""
Plot overlap vs mutual misclassification scatter (D1 synthetic analysis)
Assumes a CSV with columns: class_overlap, mutual_misclass
Outputs: plots/fig_overlap_scatter.pdf
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='results/metrics/summary_cpu.csv')
    ap.add_argument('--out', type=str, default='plots/fig_overlap_scatter.pdf')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'data_params.class_overlap' in df.columns:
        df = df.rename(columns={'data_params.class_overlap': 'class_overlap'})
    if 'metrics.mutual_misclass' in df.columns:
        df = df.rename(columns={'metrics.mutual_misclass': 'mutual_misclass'})
    
    use = df[['class_overlap', 'mutual_misclass']].dropna()
    if use.empty:
        print('[WARN] No overlap/misclass columns found; skipping plot.')
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.2, 3.4))
    sns.regplot(data=use, x='class_overlap', y='mutual_misclass', scatter_kws={'s': 20}, line_kws={'color': 'red'})
    plt.xlabel('Class Overlap')
    plt.ylabel('Mutual Misclassification')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'[INFO] Saved {args.out}')


if __name__ == '__main__':
    main()



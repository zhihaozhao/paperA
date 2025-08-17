#!/usr/bin/env python3
"""
Plot D1 CPU bars: Macro-F1 per model from results/metrics/summary_cpu.csv
Outputs: plots/fig_synth_bars.pdf
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='results/metrics/summary_cpu.csv')
    ap.add_argument('--out', type=str, default='plots/fig_synth_bars.pdf')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Normalize column names if needed
    if 'meta.model' in df.columns:
        df = df.rename(columns={'meta.model': 'model', 'metrics.macro_f1': 'macro_f1', 'metrics.ece_cal': 'ece_cal'})
    
    # Aggregate mean and std by model
    agg = df.groupby('model', as_index=False).agg(macro_f1=('macro_f1', 'mean'))
    order = ['enhanced', 'cnn', 'conformer_lite', 'bilstm']
    agg['model'] = pd.Categorical(agg['model'], categories=order, ordered=True)
    agg = agg.sort_values('model')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 3.2))
    sns.barplot(data=agg, x='model', y='macro_f1', palette='Set2')
    plt.ylim(0.0, 1.0)
    plt.ylabel('Macro-F1')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'[INFO] Saved {args.out}')


if __name__ == '__main__':
    main()



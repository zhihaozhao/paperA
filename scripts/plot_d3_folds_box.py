#!/usr/bin/env python3
"""
Plot D3 LOSO/LORO per-fold Macro-F1 boxplots from aggregated JSONs.
Usage: point to a directory (results/d3/loso or results/d3/loro) and glob *.json.
Outputs: plots/fig_d3_folds_box_[loso|loro].pdf
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_rows(root: Path):
    rows = []
    for p in root.glob('*.json'):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        proto = (obj.get('protocol') or '').lower()
        model = obj.get('model') or obj.get('args', {}).get('model')
        fr = obj.get('fold_results') or []
        for i, r in enumerate(fr):
            rows.append({'protocol': proto, 'model': model, 'fold': i, 'macro_f1': r.get('macro_f1', None)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='results/d3/loso')
    ap.add_argument('--out', type=str, default='plots/fig_d3_folds_box_loso.pdf')
    args = ap.parse_args()

    df = load_rows(Path(args.root))
    if df.empty:
        print('[WARN] no fold results found')
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.8, 3.2))
    sns.boxplot(data=df, x='model', y='macro_f1', hue='protocol', palette='Set3')
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'[INFO] Saved {args.out}')


if __name__ == '__main__':
    main()



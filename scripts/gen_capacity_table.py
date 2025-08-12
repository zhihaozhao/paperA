
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from pathlib import Path

TPL = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Model & Params (M) & Macro-F1 & ECE $\downarrow$ & NLL $\downarrow$ & Brier $\downarrow$ \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{Capacity-matched comparison (±10%% params) on %s (seed=%s). Temperature selected on validation.}
\label{tab:capacity-match}
\end{table}
"""

def fmt_row(row):
    params_m = float(row["params"]) / 1e6
    return f"{row['model']} & {params_m:.2f} & {float(row['macro_f1']):.3f} & {float(row['ece_cal']):.3f} & {float(row['nll_cal']):.3f} & {float(row['brier']):.3f} \\\\"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with rows for Enhanced_small and Baseline_large")
    ap.add_argument("--out_tex", required=True, help="Output tex path (e.g., tables/tab_capacity_match.tex)")
    ap.add_argument("--difficulty", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Optional filters
    if args.difficulty and "difficulty" in df.columns:
        df = df[df["difficulty"] == args.difficulty]
    if args.seed is not None and "seed" in df.columns:
        df = df[df["seed"] == args.seed]

    required = ["model","params","macro_f1","ece_cal","nll_cal","brier"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if len(df) < 2:
        raise ValueError("Need at least two rows to compare")

    # Check capacity match (±10%)
    df = df.sort_values("params").reset_index(drop=True)
    pmin, pmax = df["params"].iloc[0], df["params"].iloc[-1]
    ratio = float(pmax) / max(1.0, float(pmin))
    warn = ""
    if ratio > 1.10:
        warn = f"% WARNING: params ratio {ratio:.3f} exceeds 1.10; not capacity-matched within ±10%\n"

    rows = "\n".join(fmt_row(r) for _, r in df.iterrows())
    difficulty = args.difficulty if args.difficulty else (df["difficulty"].iloc[0] if "difficulty" in df.columns and len(df) else "dataset")
    seed = args.seed if args.seed is not None else (df["seed"].iloc[0] if "seed" in df.columns and len(df) else "NA")
    tex = warn + (TPL % (rows, difficulty, seed))

    outp = Path(args.out_tex)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(tex, encoding="utf-8")
    print(f"Wrote {outp}")

if __name__ == "__main__":
    main()

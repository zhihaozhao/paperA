#
# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# """
# Enhanced plotting of lambda curves (v2):
# - ECE raw and ECE cal simultaneously
# - Multi-seed aggregation (mean + SE/SD), error bars
# - Optional Macro-F1 and Mutual misclass on secondary y-axis
# - Axis and legend tweaks:
#   * Left y-axis (ECE) bottom=0, top with 10% padding
#   * Right y-axis fixed to [0, 1]
#   * Legend at upper left
#   * If Mutual misclass is all zeros, plot as scatter points and add an annotation
#
# Usage (Windows CMD example):
#   python scripts\\plot_lambda_curves.py ^
#       --csv results\\synth_lambda\\metrics_lambda_full.csv ^
#       --out_pdf plots\\fig_lambda_curves_mid_multi_seed.pdf ^
#       --difficulty mid ^
#       --seeds 0,1,2 ^
#       --metric_f1 --metric_mm ^
#       --error se
# """
#
# import argparse
# import math
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--csv", required=True, help="Input aggregated CSV path")
#     p.add_argument("--out_pdf", required=True, help="Output PDF path")
#     p.add_argument("--difficulty", required=True, help="Difficulty to filter, e.g., mid")
#     p.add_argument("--seeds", default=None, help="Comma-separated seeds to include; default: all seeds found")
#     p.add_argument("--metric_f1", action="store_true", help="Also plot Macro-F1")
#     p.add_argument("--metric_mm", action="store_true", help="Also plot Mutual misclass")
#     p.add_argument("--error", choices=["se", "sd", "none"], default="se", help="Error bars: standard error (se), standard deviation (sd), or none")
#     p.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving")
#     p.add_argument("--title", default=None, help="Custom title; default auto")
#     p.add_argument("--palette", choices=["classic", "colorblind"], default="classic")
#     return p.parse_args()
#
#
# def choose_colors(palette):
#     if palette == "colorblind":
#         # Okabe-Ito palette
#         return {
#             "ece_raw": "#0072B2",      # blue
#             "ece_cal": "#D55E00",      # vermillion
#             "macro_f1": "#009E73",     # green
#             "mm": "#CC79A7",           # purple
#         }
#     # classic
#     return {
#         "ece_raw": "#1f77b4",
#         "ece_cal": "#ff7f0e",
#         "macro_f1": "#2ca02c",
#         "mm": "#17becf",
#     }
#
#
# def sem(x: np.ndarray) -> float:
#     x = np.asarray(x, dtype=float)
#     if x.size <= 1:
#         return np.nan
#     return np.nanstd(x, ddof=1) / math.sqrt(np.sum(~np.isnan(x)))
#
#
# def aggregate(df: pd.DataFrame, error_mode: str):
#     # group by lambda; compute mean and error
#     def agg_err(x):
#         if error_mode == "sd":
#             return np.nanstd(x, ddof=1) if len(x) > 1 else np.nan
#         elif error_mode == "se":
#             return sem(x)
#         else:
#             return np.nan
#
#     agg_spec = {
#         "ece_raw": ["mean", agg_err],
#         "ece_cal": ["mean", agg_err],
#     }
#     if "macro_f1" in df.columns:
#         agg_spec["macro_f1"] = ["mean", agg_err]
#     if "mutual_misclass" in df.columns:
#         agg_spec["mutual_misclass"] = ["mean", agg_err]
#
#     grp = df.groupby("lambda", as_index=False).agg(agg_spec)
#
#     # Flatten columns
#     cols = ["lambda"]
#     flat = []
#     for base in ["ece_raw", "ece_cal", "macro_f1", "mutual_misclass"]:
#         if base in grp.columns.get_level_values(0):
#             flat += [f"{base}_mean", f"{base}_err"]
#             cols += [f"{base}_mean", f"{base}_err"]
#     grp.columns = cols
#     grp = grp.sort_values("lambda").reset_index(drop=True)
#     return grp
#
#
# def ensure_columns(df: pd.DataFrame):
#     missing = [c for c in ["lambda", "difficulty", "seed"] if c not in df.columns]
#     if missing:
#         raise ValueError(f"CSV missing required columns: {missing}")
#     # ECE columns: try fallback names if needed
#     if "ece_raw" not in df.columns:
#         if "ece" in df.columns:
#             df = df.rename(columns={"ece": "ece_raw"})
#         else:
#             raise ValueError("CSV missing 'ece_raw' or fallback 'ece' column")
#     if "ece_cal" not in df.columns:
#         df["ece_cal"] = df["ece_raw"]
#     for opt in ["macro_f1", "mutual_misclass"]:
#         if opt not in df.columns:
#             df[opt] = np.nan
#     return df
#
#
# def main():
#     args = parse_args()
#     colors = choose_colors(args.palette)
#
#     df = pd.read_csv(args.csv)
#     df = ensure_columns(df)
#
#     df = df[df["difficulty"] == args.difficulty].copy()
#     if df.empty:
#         raise SystemExit(f"No rows for difficulty={args.difficulty}")
#
#     if args.seeds:
#         seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
#         df = df[df["seed"].isin(seeds)].copy()
#         seed_text = ",".join(map(str, sorted(set(df["seed"]))))
#     else:
#         seed_text = ",".join(map(str, sorted(set(df["seed"]))))
#
#     if df.empty:
#         raise SystemExit("No data after filtering seeds/difficulty")
#
#     agg = aggregate(df, args.error)
#
#     plt.style.use("seaborn-v0_8-whitegrid")
#     fig, ax1 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
#
#     x = agg["lambda"].values
#
#     # ECE curves
#     ece_raw = agg["ece_raw_mean"].values if "ece_raw_mean" in agg.columns else None
#     ece_cal = agg["ece_cal_mean"].values if "ece_cal_mean" in agg.columns else None
#     ece_raw_err = agg["ece_raw_err"].values if "ece_raw_err" in agg.columns and args.error != "none" else None
#     ece_cal_err = agg["ece_cal_err"].values if "ece_cal_err" in agg.columns and args.error != "none" else None
#
#     if ece_raw is not None:
#         ax1.errorbar(x, ece_raw, yerr=ece_raw_err, fmt="-o", color=colors["ece_raw"],
#                      label="ECE (raw)", capsize=3, lw=2, ms=5)
#     if ece_cal is not None:
#         ax1.errorbar(x, ece_cal, yerr=ece_cal_err, fmt="-o", color=colors["ece_cal"],
#                      label="ECE (cal)", capsize=3, lw=2, ms=5)
#
#     # Left axis limits: bottom=0, top with 10% padding based on visible ECE values
#     ece_vals = []
#     if ece_raw is not None:
#         ece_vals.extend(ece_raw[~np.isnan(ece_raw)])
#     if ece_cal is not None:
#         ece_vals.extend(ece_cal[~np.isnan(ece_cal)])
#     ymax = max(ece_vals) if ece_vals else 1.0
#     ax1.set_ylim(bottom=0, top=ymax * 1.10 if ymax > 0 else 1.0)
#
#     ax1.set_xlabel("λ")
#     ax1.set_ylabel("ECE")
#     ax1.tick_params(axis="y", labelcolor="black")
#
#     # Secondary axis for Macro-F1 and Mutual misclass
#     ax2 = None
#     plotted_mm_all_zero = False
#     if args.metric_f1 or args.metric_mm:
#         ax2 = ax1.twinx()
#         ax2.set_ylabel("Macro-F1 / Mutual misclass")
#         ax2.set_ylim(0, 1)
#
#         if args.metric_f1 and "macro_f1_mean" in agg.columns:
#             ax2.errorbar(x, agg["macro_f1_mean"].values,
#                          yerr=(agg["macro_f1_err"].values if "macro_f1_err" in agg.columns and args.error != "none" else None),
#                          fmt="-s", color=colors["macro_f1"], label="Macro-F1", capsize=3, lw=1.8, ms=5)
#
#         if args.metric_mm and "mutual_misclass_mean" in agg.columns:
#             mm = agg["mutual_misclass_mean"].values
#             mm_err = agg["mutual_misclass_err"].values if "mutual_misclass_err" in agg.columns and args.error != "none" else None
#             if np.all(np.nan_to_num(mm) == 0):
#                 # Plot scatter points at 0 with small upward offset for visibility
#                 ax2.scatter(x, np.zeros_like(x), color=colors["mm"], marker="^", s=35, label="Mutual misclass")
#                 plotted_mm_all_zero = True
#             else:
#                 ax2.errorbar(x, mm, yerr=mm_err, fmt="-^", color=colors["mm"], label="Mutual misclass",
#                              capsize=3, lw=1.8, ms=5)
#
#     # Title
#     if args.title:
#         title = args.title
#     else:
#         title = f"difficulty={args.difficulty}, seeds={seed_text}" if ("," in seed_text or seed_text == "" or len(seed_text.split(",")) > 1) else f"difficulty={args.difficulty}, seed={seed_text}"
#     ax1.set_title(title)
#
#     # Legend
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     if ax2:
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         lines = lines1 + lines2
#         labels = labels1 + labels2
#     else:
#         lines, labels = lines1, labels1
#     if lines:
#         ax1.legend(lines, labels, loc="upper left", frameon=True)
#
#     # Annotation if MM all zeros
#     if plotted_mm_all_zero:
#         ax2.annotate("Mutual misclass = 0 for all λ", xy=(0.98, 0.02),
#                      xycoords="axes fraction", ha="right", va="bottom",
#                      fontsize=9, color="#444")
#
#     out_path = Path(args.out_pdf)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path.as_posix(), dpi=args.dpi)
#     print(f"Saved figure to {out_path}")
#
#
# if __name__ == "__main__":
#     main()

# 新脚本: plots/plot_lambda_curves.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('results/synth_lambda/metrics_lambda_full.csv')
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='lambda', y='ece_cal', label='ECE_cal')
sns.lineplot(data=df, x='lambda', y='ece_raw', label='ECE_raw')
sns.lineplot(data=df, x='lambda', y='macro_f1', label='F1')
plt.title('Metrics vs Lambda')
plt.savefig('plots/fig_lambda_curves.pdf')
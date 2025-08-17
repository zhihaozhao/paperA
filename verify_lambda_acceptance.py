
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
verify_lambda_acceptance.py

功能
- 聚合 results/synth_lambda/mid_s0/enhanced_l*.json 为一个 CSV
- 支持嵌套键读取（metrics.*, calibration.*），自动别名映射
- 稳健处理缺失字段/NaN：优先使用 ece_cal，若全为 NaN 则回退 ece_raw
- 计算最佳 λ（使 ECE 最小），并与 λ=0 对比
- 输出两条验收结论：
  A) 最佳点出现在 λ>0
  B) 容量对齐下准确性不劣（Macro‑F1 不降）且校准更好（ECE 降）

使用
  python verify_lambda_acceptance.py \
      --dir results/synth_lambda/mid_s0 \
      --pattern enhanced_l*.json \
      --out_csv results/synth_lambda/metrics_lambda_full.csv \
      --verbose
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=r"results/synth_lambda/mid_s0", help="JSON 目录")
    p.add_argument("--pattern", default="enhanced_l*.json", help="文件通配符")
    p.add_argument("--out_csv", default=r"results/synth_lambda/metrics_lambda_full.csv", help="输出 CSV 路径")
    p.add_argument("--model_name", default="enhanced")
    p.add_argument("--difficulty", default="mid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps_f1", type=float, default=0.0, help="F1 容差，允许轻微下降")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def safe_get_path(d: Dict[str, Any], paths, default=np.nan):
    """
    支持嵌套路径读取：
      paths: 别名列表，每个元素可以是 "a.b.c" 形式
    任一可用即返回；全失败返回 default。
    """
    for p in paths:
        cur = d
        ok = True
        for k in p.split("."):
            if isinstance(cur, dict) and (k in cur):
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def load_rows(dir_path: str, pattern: str, model: str, difficulty: str, seed: int, verbose: bool) -> pd.DataFrame:
    pattern_glob = os.path.join(dir_path, pattern)
    files: List[str] = sorted(glob.glob(pattern_glob))
    if not files:
        # 兼容正反斜杠
        alt = pattern_glob.replace("\\", "/") if "\\" in pattern_glob else pattern_glob.replace("/", "\\")
        files = sorted(glob.glob(alt))
    if not files:
        raise SystemExit(f"未找到文件：{pattern_glob}")

    if verbose:
        print(f"匹配到 {len(files)} 个文件：")
        for f in files:
            print(" -", f)

    rows = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"读取失败，跳过 {f}: {e}", file=sys.stderr)
            continue

        base = os.path.basename(f)
        # 解析 lambda：假设形如 enhanced_l0.08.json 或 enhanced_l0.json
        try:
            lam_str = base.split("enhanced_l")[-1].split(".json")[0]
            lam = float(lam_str)
        except Exception:
            if verbose:
                print(f"警告：无法从文件名解析 lambda，跳过 {f}")
            continue

        # 从 metrics / calibration 等嵌套结构读取
        macro_f1 = safe_get_path(d, ["metrics.macro_f1", "metrics.f1", "macro_f1", "f1"])
        mutual_misclass = safe_get_path(d, ["metrics.mutual_misclass", "mutual_misclass", "metrics.mm", "mm"])
        ece_raw = safe_get_path(d, ["metrics.ece_raw", "metrics.ece", "ece_raw", "ece"])
        ece_cal = safe_get_path(d, ["metrics.ece_cal", "ece_cal"])
        nll_raw = safe_get_path(d, ["metrics.nll_raw", "nll_raw"])
        nll_cal = safe_get_path(d, ["metrics.nll_cal", "nll_cal"])
        temperature = safe_get_path(d, ["calibration.temperature", "metrics.temperature", "temperature", "T"])

        rows.append({
            "model": model,
            "difficulty": difficulty,
            "seed": safe_get_path(d, ["seed"], default=seed),
            "macro_f1": macro_f1,
            "mutual_misclass": mutual_misclass,
            "ece_raw": ece_raw,
            "ece_cal": ece_cal,
            "nll_raw": nll_raw,
            "nll_cal": nll_cal,
            "temperature": temperature,
            "lambda": lam,
            "path": f
        })

    if not rows:
        raise SystemExit("没有可用的数据行，检查文件内容与文件名格式。")

    df = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)

    # 若 ece_cal 全 NaN，则复制 ece_raw 便于导出，但会在判定时告警并回退
    if df["ece_cal"].isna().all() and not df["ece_raw"].isna().all():
        if verbose:
            print("警告：ece_cal 全为 NaN，将用 ece_raw 填充导出（判定时仍会回退使用 ece_raw）。")
        df["ece_cal"] = df["ece_raw"]

    return df


def choose_best_lambda(df: pd.DataFrame, eps_f1: float, verbose: bool):
    g = df.groupby("lambda", as_index=False).agg({
        "ece_cal": "mean",
        "ece_raw": "mean",
        "macro_f1": "mean"
    })

    gc = g.dropna(subset=["ece_cal"]).copy()
    use_cal = not gc.empty

    if use_cal:
        best_row = gc.loc[gc["ece_cal"].idxmin()]
        lam0_rows = gc.loc[gc["lambda"].eq(0)]
        if lam0_rows.empty:
            lam0_rows = g.loc[g["lambda"].eq(0)]  # 兜底
        metric_name = "ECE_cal"
        best_ece = float(best_row["ece_cal"])
        lam0_ece = float(lam0_rows["ece_cal"].values[0])
    else:
        gr = g.dropna(subset=["ece_raw"]).copy()
        if gr.empty:
            raise SystemExit("无法判定：ece_cal 与 ece_raw 均为 NaN。请检查 JSON 的 metrics.* 字段是否存在。")
        print("警告：ece_cal 缺失或全为 NaN，回退使用 ece_raw 进行判定。")
        best_row = gr.loc[gr["ece_raw"].idxmin()]
        lam0_rows = gr.loc[gr["lambda"].eq(0)]
        if lam0_rows.empty:
            raise SystemExit("数据缺少 λ=0 的对照，无法完成验收判定。")
        metric_name = "ECE_raw"
        best_ece = float(best_row["ece_raw"])
        lam0_ece = float(lam0_rows["ece_raw"].values[0])

    if lam0_rows.empty:
        raise SystemExit("数据缺少 λ=0 的对照，无法完成验收判定。")

    best_lam = float(best_row["lambda"])
    best_f1 = float(best_row["macro_f1"])
    lam0_f1 = float(lam0_rows["macro_f1"].values[0])

    if verbose:
        print("\n聚合数据：")
        print(g.sort_values("lambda").to_string(index=False))

    return {
        "metric_name": metric_name,
        "best_lam": best_lam,
        "best_ece": best_ece,
        "best_f1": best_f1,
        "lam0_ece": lam0_ece,
        "lam0_f1": lam0_f1,
        "pass_A": best_lam > 0,
        "pass_B": (best_f1 >= lam0_f1 - eps_f1) and (best_ece < lam0_ece - 1e-12)
    }


def main():
    args = parse_args()

    df = load_rows(args.dir, args.pattern, args.model_name, args.difficulty, args.seed, args.verbose)

    # 保存 CSV（与绘图脚本兼容）
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Saved CSV:", args.out_csv)
    print(df[["lambda", "macro_f1", "ece_raw", "ece_cal", "nll_raw", "nll_cal", "temperature"]]
          .to_string(index=False))

    # 选择最佳 λ 并验收
    res = choose_best_lambda(df, eps_f1=args.eps_f1, verbose=args.verbose)

    print("\nBest lambda (by {}): {}".format(res["metric_name"], res["best_lam"]))
    print("ECE(best) = {:.6f}   vs   ECE(λ=0) = {:.6f}   Δ = {:+.6f}"
          .format(res["best_ece"], res["lam0_ece"], res["best_ece"] - res["lam0_ece"]))
    print("F1(best)  = {:.6f}   vs   F1(λ=0)  = {:.6f}   Δ = {:+.6f}"
          .format(res["best_f1"], res["lam0_f1"], res["best_f1"] - res["lam0_f1"]))

    print("\n验收 A（最佳点出现在 λ>0）:", "通过" if res["pass_A"] else "未通过")
    print("验收 B（容量对齐下不劣且校准更好）:", "通过" if res["pass_B"] else "未通过")

    if not (res["pass_A"] and res["pass_B"]):
        print("\n提示：")
        print("- 若 temperature 恒为 1.0 或 ece_cal 与 ece_raw 完全相等，说明温度校准可能未生效；"
              "请检查评估是否加载了每个 λ 的最佳温度。")
        print("- 可扩展 λ 搜索到更细粒度：0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.12, 0.18")
        print("- 建议多 seed 后用均值±SE 作图，再做验收。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--d5-dir", default=r"..\WiFi-CSI-Sensing-Results\results_gpu\d5")
    p.add_argument("--d6-dir", default=r"..\WiFi-CSI-Sensing-Results\results_gpu\d6")
    p.add_argument("--out", default=r"docs\d5_d6_acceptance_report.txt")
    p.add_argument("--ece-threshold", type=float, default=-1.0, help=">0 launch: each model need ece_cal_mean <= threshold")
    p.add_argument("--min-d6-runs", type=int, default=0, help=">0 launch:  D6  result num >= val")
    return p.parse_args()

def find_result_files(dir_path: str, include_patterns=None) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    files: List[str] = []
    if include_patterns:
        pats = [os.path.join(dir_path, pat) for pat in include_patterns]
    else:
        pats = [os.path.join(dir_path, "*.json"), os.path.join(dir_path, "**", "*.json")]
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    files = [f for f in files if os.path.isfile(f)
             and "summary" not in os.path.basename(f).lower()]
    return sorted(files)
	



def safe_get(d: Dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and (k in cur):
            cur = cur[k]
        else:
            return default
    return cur

def _try_get_number(d: dict, paths: list, default=None):
    for p in paths:
        cur = d
        ok = True
        for k in p:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return default

def _get_path_value(d, path):
    cur = d
    for k in path:
        if isinstance(cur, dict) and isinstance(k, str) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and 0 <= k < len(cur):
            cur = cur[k]
        else:
            return None
    return cur

def _try_number(d, path_options, default=None):
    for p in path_options:
        v = _get_path_value(d, p)
        if isinstance(v, (int, float)):
            return float(v)
        # 兼容字符串数字
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                pass
    return default

def load_results(files: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue

        # 先尝试 aggregate_stats（你的 D6 JSON 有这一结构）
        macro_f1 = _try_number(data, [
            ["aggregate_stats", "macro_f1", "mean"],
            ["aggregate_stats", "f1_macro", "mean"],
        ])
        ece_cal = _try_number(data, [
            ["aggregate_stats", "ece", "mean"],      # 你的键名是 ece
            ["aggregate_stats", "ece_cal", "mean"],
        ])

        # 若没有 aggregate_stats，则回退到第 0 折 或 常见 metrics
        if macro_f1 is None:
            macro_f1 = _try_number(data, [
                ["fold_results", 0, "macro_f1"],
                ["metrics", "macro_f1"],
                ["macro_f1"],
            ])
        if ece_cal is None:
            ece_cal = _try_number(data, [
                ["fold_results", 0, "ece"],
                ["metrics", "ece_cal"],
                ["metrics", "ece_after"],
                ["ece_cal"],
                ["ece_after"],
                ["ece"],
            ])

        if macro_f1 is None:
            print(f"Skip (no macro_f1): {f}")
            continue
        if ece_cal is None:
            ece_cal = float("nan")

        model = (
            _get_path_value(data, ["model"]) or
            _get_path_value(data, ["meta", "model"]) or
            _get_path_value(data, ["args", "model"]) or
            "unknown"
        )
        seed = (
            _get_path_value(data, ["seed"]) or
            _get_path_value(data, ["meta", "args", "seed"]) or
            _get_path_value(data, ["meta", "seed"]) or
            _get_path_value(data, ["args", "seed"]) or
            -1
        )
        try:
            seed = int(seed)
        except Exception:
            seed = -1

        rows.append({
            "path": f,
            "model": str(model),
            "seed": seed,
            "macro_f1": float(macro_f1),
            "ece_cal": float(ece_cal) if isinstance(ece_cal, (int, float)) else float("nan"),
        })
    return rows
	
def mean(values: List[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def std(values: List[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    n = len(vals)
    if n <= 1:
        return 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)  # sample std
    return math.sqrt(var)


def summarize(rows: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    if not rows:
        return [], {}
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    seeds_per_model: Dict[str, set] = defaultdict(set)
    for r in rows:
        by_model[r["model"]].append(r)
        seeds_per_model[r["model"]].add(r["seed"])

    summary: List[Dict] = []
    for model, items in by_model.items():
        f1_list = [x["macro_f1"] for x in items]
        ece_list = [x["ece_cal"] for x in items]
        summary.append({
            "model": model,
            "runs": len(items),
            "macro_f1_mean": mean(f1_list),
            "macro_f1_std": std(f1_list),
            "ece_cal_mean": mean(ece_list),
            "ece_cal_std": std(ece_list),
        })

    seeds_count = {m: len(seeds_per_model[m]) for m in seeds_per_model}
    # sort by model name for stable output
    summary.sort(key=lambda d: d["model"])
    return summary, seeds_count


def fmt_float(v: float) -> str:
    if v != v:  # NaN check
        return "nan"
    return f"{v:.3f}"


def format_summary_table(summary: List[Dict]) -> str:
    if not summary:
        return "(no data)"
    headers = ["model", "runs", "macro_f1_mean", "macro_f1_std", "ece_cal_mean", "ece_cal_std"]
    col_widths = {h: len(h) for h in headers}
    rows_fmt: List[Dict[str, str]] = []
    for r in summary:
        row = {
            "model": str(r["model"]),
            "runs": str(r["runs"]),
            "macro_f1_mean": fmt_float(r["macro_f1_mean"]),
            "macro_f1_std": fmt_float(r["macro_f1_std"]),
            "ece_cal_mean": fmt_float(r["ece_cal_mean"]),
            "ece_cal_std": fmt_float(r["ece_cal_std"]),
        }
        rows_fmt.append(row)
        for k, v in row.items():
            col_widths[k] = max(col_widths[k], len(v))

    def fmt_row(row: Dict[str, str]) -> str:
        return "  ".join(row[h].ljust(col_widths[h]) for h in headers)

    lines = ["  ".join(h.ljust(col_widths[h]) for h in headers)]
    lines.append("-" * (sum(col_widths.values()) + 2 * (len(headers) - 1)))
    for row in rows_fmt:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def build_report(d5_rows: List[Dict], d6_rows: List[Dict], ece_thr: float, min_d6_runs: int) -> str:
    lines: List[str] = []
    lines.append("D5 & D6 ACCEPTANCE SUMMARY")
    lines.append("============================================================")
    lines.append(f"Total runs: {len(d5_rows) + len(d6_rows)} (D5: {len(d5_rows)}, D6: {len(d6_rows)})")

    d5_summary, d5_seeds = summarize(d5_rows)
    d6_summary, d6_seeds = summarize(d6_rows)

    if d5_rows:
        lines.append("")
        lines.append("[D5] per-model summary:")
        lines.append(format_summary_table(d5_summary))
        lines.append(f"Seeds per model: {d5_seeds}")

    if d6_rows:
        lines.append("")
        lines.append("[D6] per-model summary:")
        lines.append(format_summary_table(d6_summary))
        lines.append(f"Seeds per model: {d6_seeds}")

    # Optional checks
    lines.append("")
    lines.append("Acceptance checks (if thresholds provided):")
    if ece_thr > 0:
        def check(summary: List[Dict]) -> Dict[str, bool]:
            return {r['model']: (isinstance(r['ece_cal_mean'], float) and not math.isnan(r['ece_cal_mean']) and r['ece_cal_mean'] <= ece_thr) for r in summary}
        if d5_summary:
            lines.append(f" - D5 ECE <= {ece_thr}: {check(d5_summary)}")
        if d6_summary:
            lines.append(f" - D6 ECE <= {ece_thr}: {check(d6_summary)}")
    else:
        lines.append(" - (no ECE threshold given)")

    if min_d6_runs > 0:
        lines.append(f" - D6 minimum runs >= {min_d6_runs}: {len(d6_rows) >= min_d6_runs}")
    else:
        lines.append(" - (no D6 minimum run requirement)")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    d5_files = find_result_files(args.d5_dir)
    #d6_files = find_result_files(args.d6_dir)
    include_patterns=(["loso_*.json", "loro_*.json"])
    d6_files = find_result_files(args.d6_dir) 
    #d6_files = find_result_files(args.d6_dir, include_patterns=["loso_*.json", "loro_*.json"])
    print(f"Scanning D5 dir: {args.d5_dir}")
    print(f"Found {len(d5_files)} files")
    print(f"Scanning D6 dir: {args.d6_dir}")
    print(f"Found {len(d6_files)} files")

    d5_rows = load_results(d5_files)
    d6_rows = load_results(d6_files)

    report = build_report(d5_rows, d6_rows, args.ece_threshold, args.min_d6_runs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", errors="ignore") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nSaved report to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
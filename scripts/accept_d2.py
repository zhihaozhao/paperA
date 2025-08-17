#!/usr/bin/env python3
"""
Acceptance checker for D2 sweep experiments.

Reads result JSON files (as produced by src/train_eval.py) under a root directory
and verifies basic coverage and metric sanity. Produces:
 - results/metrics/summary_d2.csv  (aggregated rows)
 - results/metrics/d2_acceptance_report.txt (human-readable report)

Exit code:
 - 0 if all checks pass
 - 2 if any acceptance check fails

Usage (examples):
  python scripts/accept_d2.py --root results_gpu/d2 --min_seeds 3 --models enhanced cnn bilstm conformer_lite
  python scripts/accept_d2.py --pattern "results_gpu/d2/**/*.json" --require_by model
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class ResultRow:
    file: str
    model: str
    seed: int
    difficulty: str
    macro_f1: Optional[float]
    ece_cal: Optional[float]
    nll_cal: Optional[float]
    brier: Optional[float]
    class_overlap: Optional[float]
    env_burst_rate: Optional[float]
    gain_drift_std: Optional[float]
    sc_corr_rho: Optional[float]
    label_noise_prob: Optional[float]


def safe_get(dct: Dict[str, Any], *keys: str, default=None):
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_results(paths: List[Path]) -> List[ResultRow]:
    rows: List[ResultRow] = []
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            # Skip unreadable files
            continue

        model = safe_get(obj, "meta", "model", default=safe_get(obj, "args", "model", default=""))
        seed = safe_get(obj, "meta", "seed", default=safe_get(obj, "args", "seed", default=-1))
        difficulty = safe_get(obj, "args", "difficulty", default=safe_get(obj, "meta", "difficulty", default=""))

        macro_f1 = safe_get(obj, "metrics", "macro_f1")
        ece_cal = safe_get(obj, "metrics", "ece_cal")
        nll_cal = safe_get(obj, "metrics", "nll_cal")
        brier = safe_get(obj, "metrics", "brier")

        class_overlap = safe_get(obj, "data_params", "class_overlap", default=safe_get(obj, "args", "class_overlap"))
        env_burst_rate = safe_get(obj, "data_params", "env_burst_rate", default=safe_get(obj, "args", "env_burst_rate"))
        gain_drift_std = safe_get(obj, "data_params", "gain_drift_std", default=safe_get(obj, "args", "gain_drift_std"))
        sc_corr_rho = safe_get(obj, "data_params", "sc_corr_rho", default=safe_get(obj, "args", "sc_corr_rho"))
        label_noise_prob = safe_get(obj, "data_params", "label_noise_prob", default=safe_get(obj, "args", "label_noise_prob"))

        rows.append(ResultRow(
            file=str(p).replace("\\", "/"),
            model=str(model),
            seed=int(seed) if isinstance(seed, int) or (isinstance(seed, str) and seed.isdigit()) else -1,
            difficulty=str(difficulty),
            macro_f1=float(macro_f1) if macro_f1 is not None else None,
            ece_cal=float(ece_cal) if ece_cal is not None else None,
            nll_cal=float(nll_cal) if nll_cal is not None else None,
            brier=float(brier) if brier is not None else None,
            class_overlap=float(class_overlap) if class_overlap is not None else None,
            env_burst_rate=float(env_burst_rate) if env_burst_rate is not None else None,
            gain_drift_std=float(gain_drift_std) if gain_drift_std is not None else None,
            sc_corr_rho=float(sc_corr_rho) if sc_corr_rho is not None else None,
            label_noise_prob=float(label_noise_prob) if label_noise_prob is not None else None,
        ))
    return rows


def write_summary_csv(rows: List[ResultRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "model", "seed", "difficulty",
            "macro_f1", "ece_cal", "nll_cal", "brier",
            "class_overlap", "env_burst_rate", "gain_drift_std", "sc_corr_rho", "label_noise_prob",
        ])
        for r in rows:
            w.writerow([
                r.file, r.model, r.seed, r.difficulty,
                r.macro_f1, r.ece_cal, r.nll_cal, r.brier,
                r.class_overlap, r.env_burst_rate, r.gain_drift_std, r.sc_corr_rho, r.label_noise_prob,
            ])


def aggregate_means(rows: List[ResultRow]) -> Dict[str, Tuple[float, int]]:
    agg: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.model and r.macro_f1 is not None:
            agg[r.model].append(r.macro_f1)
    return {m: (sum(v) / len(v), len(v)) for m, v in agg.items() if v}


def check_coverage(rows: List[ResultRow], required_models: List[str], min_seeds: int, require_by: str) -> List[str]:
    problems: List[str] = []
    if require_by == "model":
        seeds_by_model: Dict[str, set] = defaultdict(set)
        for r in rows:
            seeds_by_model[r.model].add(r.seed)
        for m in required_models:
            if len(seeds_by_model.get(m, set())) < min_seeds:
                problems.append(f"Coverage: model={m} seeds={len(seeds_by_model.get(m, set()))} < {min_seeds}")
        return problems

    # Extended schemas can be added later (e.g., model+difficulty)
    if require_by == "model+difficulty":
        combos: Dict[Tuple[str, str], set] = defaultdict(set)
        for r in rows:
            combos[(r.model, r.difficulty)].add(r.seed)
        for m in required_models:
            keys = [k for k in combos.keys() if k[0] == m]
            for k in keys:
                if len(combos[k]) < min_seeds:
                    problems.append(f"Coverage: model={k[0]} difficulty={k[1]} seeds={len(combos[k])} < {min_seeds}")
        return problems

    return problems


def check_metric_sanity(rows: List[ResultRow]) -> List[str]:
    problems: List[str] = []
    for r in rows:
        if r.macro_f1 is None or not (0.0 <= r.macro_f1 <= 1.0):
            problems.append(f"Sanity: {r.file} macro_f1 out of range: {r.macro_f1}")
        if r.ece_cal is not None and not (0.0 <= r.ece_cal <= 1.0):
            problems.append(f"Sanity: {r.file} ece_cal out of range: {r.ece_cal}")
        if r.nll_cal is not None and not (r.nll_cal >= 0.0):
            problems.append(f"Sanity: {r.file} nll_cal negative: {r.nll_cal}")
        if r.brier is not None and not (0.0 <= r.brier <= 2.0):
            problems.append(f"Sanity: {r.file} brier out of range: {r.brier}")
    return problems


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Acceptance checker for D2 sweep experiments")
    p.add_argument("--root", type=str, default="results_gpu/d2", help="Root directory containing JSON result files")
    p.add_argument("--pattern", type=str, default="**/*.json", help="Glob pattern under root")
    p.add_argument("--models", nargs="*", default=["enhanced", "cnn", "bilstm", "conformer_lite"], help="Required model names")
    p.add_argument("--min_seeds", type=int, default=3, help="Minimum number of distinct seeds per requirement unit")
    p.add_argument("--require_by", type=str, default="model", choices=["model", "model+difficulty"], help="Acceptance granularity")
    p.add_argument("--out_dir", type=str, default="results/metrics", help="Directory to write summary/report")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    root = Path(args.root)
    paths = sorted(root.glob(args.pattern))
    if not paths:
        print(f"No result files matched: root={root} pattern={args.pattern}")
        return 2

    rows = read_results(paths)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary_d2.csv"
    write_summary_csv(rows, summary_csv)

    means = aggregate_means(rows)
    coverage_problems = check_coverage(rows, required_models=args.models, min_seeds=args.min_seeds, require_by=args.require_by)
    sanity_problems = check_metric_sanity(rows)

    report_lines: List[str] = []
    report_lines.append("D2 Acceptance Report")
    report_lines.append(f"Root: {root}")
    report_lines.append(f"Files: {len(paths)}")
    report_lines.append("")
    report_lines.append("Means by model (Macro-F1, count):")
    for m in sorted(means.keys()):
        mu, n = means[m]
        report_lines.append(f"  - {m}: {mu:.6f} ({n})")
    report_lines.append("")
    if coverage_problems:
        report_lines.append("Coverage issues:")
        for s in coverage_problems:
            report_lines.append(f"  - {s}")
        report_lines.append("")
    if sanity_problems:
        report_lines.append("Metric sanity issues:")
        for s in sanity_problems:
            report_lines.append(f"  - {s}")
        report_lines.append("")

    status = "PASS" if not coverage_problems and not sanity_problems else "FAIL"
    report_lines.append(f"Status: {status}")

    report_path = out_dir / "d2_acceptance_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())



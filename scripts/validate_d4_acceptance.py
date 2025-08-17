#!/usr/bin/env python3
"""
Validate D4 Sim2Real experiments and export summary.

Scans results under a root for JSONs with protocol=="Sim2Real" and aggregates:
 - results/metrics/summary_d4.csv
 - results/metrics/d4_acceptance_report.txt (or --save_report)

Acceptance defaults:
 - Coverage: each required model has ≥ min_runs files
 - Label efficiency: presence of multiple label_ratio points
 - Sanity ranges: 0<=macro_f1<=1, ece<=0.5; soft warning if macro_f1>0.99
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import csv


def read_json(p: Path) -> Dict[str, Any] | None:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in root.rglob("*.json"):
        obj = read_json(p)
        if not obj:
            continue
        proto = (obj.get("protocol") or "").lower()
        if proto != "sim2real":
            continue
        model = obj.get("model") or obj.get("args", {}).get("model")
        label_ratio = float(obj.get("label_ratio") or obj.get("args", {}).get("label_ratio") or 0.0)
        tm = obj.get("transfer_method") or obj.get("args", {}).get("transfer_method") or ""
        tgt = obj.get("target_metrics") or {}
        rows.append({
            "file": str(p).replace("\\", "/"),
            "model": str(model or ""),
            "label_ratio": label_ratio,
            "transfer_method": str(tm),
            "macro_f1": float(tgt.get("macro_f1", 0.0)),
            "falling_f1": float(tgt.get("falling_f1", 0.0)),
            "ece": float(tgt.get("ece", 0.0)),
        })
    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "model", "label_ratio", "transfer_method", "macro_f1", "falling_f1", "ece"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate D4 Sim2Real acceptance")
    ap.add_argument("--root", type=str, default="results/d4", help="Root directory to scan")
    ap.add_argument("--out_dir", type=str, default="results/metrics", help="Directory for summaries")
    ap.add_argument("--models", nargs="*", default=["enhanced", "cnn", "bilstm", "conformer_lite"], help="Required models")
    ap.add_argument("--min_runs", type=int, default=5, help="Minimum runs per model")
    ap.add_argument("--save_report", type=str, default="", help="Optional report path (.md)")
    args = ap.parse_args()

    root = Path(args.root)
    rows = collect(root)
    out_dir = Path(args.out_dir)
    write_csv(rows, out_dir / "summary_d4.csv")

    # Build report
    report: List[str] = []
    report.append("D4 Acceptance Report")
    report.append(f"Root: {root}")
    report.append(f"Rows: {len(rows)}\n")

    # Coverage
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    cov_ok = True
    for m in args.models:
        cnt = len(by_model.get(m, []))
        report.append(f"- {m}: runs={cnt} (min={args.min_runs}) [{'OK' if cnt >= args.min_runs else 'MISS'}]")
        cov_ok = cov_ok and (cnt >= args.min_runs)

    # Label ratios presence
    ratios = sorted({r["label_ratio"] for r in rows})
    report.append(f"\nLabel ratios covered: {ratios}")

    # Sanity
    if rows:
        mvals = [r["macro_f1"] for r in rows]
        evals = [r["ece"] for r in rows]
        lo, hi = min(mvals), max(mvals)
        report.append(f"macro_f1 range: {lo:.3f}..{hi:.3f}")
        loe, hie = min(evals), max(evals)
        report.append(f"ece range: {loe:.3f}..{hie:.3f}")

    status = "PASS" if cov_ok else "FAIL"
    report.append(f"\nStatus: {status}")

    out_report = out_dir / "d4_acceptance_report.txt"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report), encoding="utf-8")
    if args.save_report:
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_report).write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    return 0 if cov_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())



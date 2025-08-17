#!/usr/bin/env python3
"""
Validate D3 cross-domain (LOSO/LORO) experiments.

- Scans results recursively and filters JSONs by meta.protocol in {LOSO,LORO}
- Exports summary CSV to results/metrics/summary_d3.csv
- Writes a human-readable report to results/metrics/d3_acceptance_report.txt (and optional --save_report)

Acceptance (default):
 - Coverage: For each protocol, each of required models has ≥ min_runs files
 - Metric sanity: 0 <= macro_f1 <= 1, ece <= 0.5
 - Reasonable ranges: 0.70 <= macro_f1 <= 0.98, ece <= 0.15 (soft fail → WARN)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import csv


def read_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_results(root: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for p in root.rglob("*.json"):
        obj = read_json(p)
        if not obj:
            continue
        proto = (obj.get("protocol") or obj.get("meta", {}).get("protocol") or "").upper()
        if proto not in {"LOSO", "LORO"}:
            continue
        model = obj.get("model") or obj.get("args", {}).get("model") or obj.get("meta", {}).get("model")
        agg = obj.get("aggregate_stats") or {}
        macro = agg.get("macro_f1", {}).get("mean") if isinstance(agg.get("macro_f1"), dict) else agg.get("macro_f1")
        falling = agg.get("falling_f1", {}).get("mean") if isinstance(agg.get("falling_f1"), dict) else agg.get("falling_f1")
        ece = agg.get("ece", {}).get("mean") if isinstance(agg.get("ece"), dict) else agg.get("ece")
        results.append({
            "file": str(p).replace("\\", "/"),
            "protocol": proto,
            "model": str(model or ""),
            "macro_f1": float(macro or 0.0),
            "falling_f1": float(falling or 0.0),
            "ece": float(ece or 0.0),
        })
    return results


def write_summary_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "protocol", "model", "macro_f1", "falling_f1", "ece"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate D3 LOSO/LORO acceptance")
    ap.add_argument("--root", type=str, default="results", help="Root directory to scan for JSON results")
    ap.add_argument("--out_dir", type=str, default="results/metrics", help="Directory to save summaries")
    ap.add_argument("--models", nargs="*", default=["enhanced", "cnn", "bilstm", "conformer_lite"], help="Required models")
    ap.add_argument("--min_runs", type=int, default=3, help="Minimum runs per model per protocol")
    ap.add_argument("--save_report", type=str, default="", help="Optional path to save a markdown report")
    args = ap.parse_args()

    root = Path(args.root)
    rows = collect_results(root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary_d3.csv"
    write_summary_csv(rows, summary_csv)

    # Build simple report
    protos = {r["protocol"] for r in rows}
    by_proto_model: Dict[str, Dict[str, List[Dict[str, Any]]]] = {p: {} for p in protos}
    for r in rows:
        by_proto_model.setdefault(r["protocol"], {}).setdefault(r["model"], []).append(r)

    report_lines: List[str] = []
    report_lines.append("D3 Acceptance Report")
    report_lines.append(f"Scanned root: {root}")
    report_lines.append(f"Summary CSV: {summary_csv}")
    report_lines.append("")
    status_ok = True
    for proto in sorted(protos):
        report_lines.append(f"Protocol: {proto}")
        for m in args.models:
            runs = by_proto_model.get(proto, {}).get(m, [])
            cnt = len(runs)
            cov_ok = cnt >= args.min_runs
            status_ok = status_ok and cov_ok
            report_lines.append(f"  - {m}: runs={cnt} (min={args.min_runs}) [{'OK' if cov_ok else 'MISS'}]")
        report_lines.append("")

    def _soft_check(val: float, lo: float, hi: float) -> str:
        return "OK" if (lo <= val <= hi) else "WARN"

    for proto in sorted(protos):
        report_lines.append(f"Metric ranges for {proto}:")
        proto_rows = [r for r in rows if r["protocol"] == proto]
        if not proto_rows:
            continue
        mvals = [r["macro_f1"] for r in proto_rows]
        evals = [r["ece"] for r in proto_rows]
        if mvals:
            lo, hi = min(mvals), max(mvals)
            report_lines.append(f"  - macro_f1: {lo:.3f}..{hi:.3f} [{_soft_check(lo, 0.70, 1.0)}/{_soft_check(hi, 0.0, 0.98)}]")
        if evals:
            loe, hie = min(evals), max(evals)
            report_lines.append(f"  - ece: {loe:.3f}..{hie:.3f} [{_soft_check(hie, 0.0, 0.15)}]")
        report_lines.append("")

    status = "PASS" if status_ok else "FAIL"
    report_lines.append(f"Status: {status}")

    out_report = out_dir / "d3_acceptance_report.txt"
    out_report.write_text("\n".join(report_lines), encoding="utf-8")

    if args.save_report:
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_report).write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    return 0 if status_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())



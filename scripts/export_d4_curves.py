#!/usr/bin/env python3
"""
Export D4 Sim2Real label-efficiency curves from result JSONs.
Generates a CSV per model with (label_ratio, macro_f1, falling_f1, ece), and a combined CSV.
"""

import argparse
import json
from pathlib import Path
import csv


def find_rows(root: Path):
    rows = []
    for p in root.rglob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (obj.get("protocol") or "").lower() != "sim2real":
            continue
        m = obj.get("model") or obj.get("args", {}).get("model")
        r = float(obj.get("label_ratio") or obj.get("args", {}).get("label_ratio") or 0.0)
        tgt = obj.get("target_metrics") or {}
        rows.append({
            "file": str(p).replace("\\", "/"),
            "model": m,
            "label_ratio": r,
            "macro_f1": float(tgt.get("macro_f1", 0.0)),
            "falling_f1": float(tgt.get("falling_f1", 0.0)),
            "ece": float(tgt.get("ece", 0.0)),
        })
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "model", "label_ratio", "macro_f1", "falling_f1", "ece"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Export D4 label efficiency curves")
    ap.add_argument("--root", type=str, default="results/d4", help="Root with Sim2Real results")
    ap.add_argument("--out_dir", type=str, default="results/metrics", help="Output directory")
    args = ap.parse_args()

    root = Path(args.root)
    rows = find_rows(root)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "d4_label_efficiency_all.csv", rows)

    # Per-model CSVs
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for m, rs in by_model.items():
        rs_sorted = sorted(rs, key=lambda x: x["label_ratio"])  # ordered by ratio
        write_csv(out_dir / f"d4_label_efficiency_{m}.csv", rs_sorted)

    print(f"Exported label-efficiency CSVs to {out_dir}")


if __name__ == "__main__":
    main()



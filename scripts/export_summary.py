import os
import json
import glob
import argparse
import csv


def parse_args():
    ap = argparse.ArgumentParser("Export summary CSV from JSON results")
    ap.add_argument("--pattern", type=str, default="results_*\\paperA_*_hard_*.json",
                    help="Glob pattern to match result JSON files (e.g., results_cpu\\paperA_* or results_gpu\\paperA_*)")
    ap.add_argument("--out_csv", type=str, default="results/metrics/summary_all.csv")
    return ap.parse_args()


KEEP = [
    ("meta", "model"),
    ("meta", "seed"),
    ("meta", "F"),
    ("meta", "T"),
    ("meta", "num_classes"),
    ("meta", "git_commit"),
    ("meta", "git_branch"),
    ("meta", "device"),
    ("meta", "timestamp"),
    ("args", "difficulty"),
    ("metrics", "macro_f1"),
    ("metrics", "falling_f1"),
    ("metrics", "ece_cal"),
    ("metrics", "nll_cal"),
    ("metrics", "brier"),
    ("metrics", "mutual_misclass"),
]


def pick(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    args = parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched: {args.pattern}")
        return
    rows = []
    for p in files:
        try:
            obj = json.load(open(p, 'r', encoding='utf-8'))
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
            continue
        row = {"file": p.replace('\\', '/')}  # normalize
        for sec, key in KEEP:
            row[f"{sec}.{key}"] = pick(obj, (sec, key))
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote summary: {args.out_csv} (rows={len(rows)})")


if __name__ == '__main__':
    main()



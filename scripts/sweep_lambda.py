
# Minimally patched sweep_lambda.py based on your provided code.
# Key changes (minimal):
# - Added argparse flags for --n_samples, --epochs, --batch (with defaults) to parse_args().
# - In run_one(), replaced hardcoded values with args.n_samples, args.epochs, args.batch in cmd.
# - No other changes; retained all your existing logic, including hardcoded noise params, --difficulty "hard", and cmd additions.
# Integration: Replace your sweep_lambda.py with this, then rerun your original command (it will now recognize and use the args).

import os
import json
import subprocess
import sys
import csv
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

DEFAULT_TRAIN_REL = "src/train_eval.py"
OUT_ROOT = Path("results/synth_lambda")
DIFFS_DEFAULT = ["easy", "mid", "hard"]
LAMBDAS_DEFAULT = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18]


def parse_lambda_list(spec: Optional[str]) -> List[float]:
    if not spec:
        return LAMBDAS_DEFAULT[:]
    spec = spec.strip()
    if ":" in spec and "," not in spec:
        a, b, c = spec.split(":")
        lo, hi, step = float(a), float(b), float(c)
        vals, x = [], lo
        for _ in range(100000):
            if x > hi + 1e-12: break
            vals.append(round(x, 10))
            x += step
        return sorted(set(vals))
    else:
        vals = [float(x) for x in spec.split(",") if x.strip() != ""]
        return sorted(set(vals))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default=DEFAULT_TRAIN_REL)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--difficulties", type=str, default=",".join(DIFFS_DEFAULT))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--lambdas", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    # 新增：温度参数
    ap.add_argument("--temp_mode", type=str, default="none", choices=["none", "logspace", "learnable"])
    ap.add_argument("--temp_min", type=float, default=0.5)
    ap.add_argument("--temp_max", type=float, default=5.0)
    ap.add_argument("--temp_steps", type=int, default=40)
    ap.add_argument("--fixed_temp", type=float, default=None,
                    help="If set, skip temperature search and use this fixed temperature in train_eval.")
    ap.add_argument("--val_split", type=float, default=0.5,
                    help="Fraction of test set to use as validation for temperature scaling.")
    # NEW: Added flags for training params with defaults
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)

    return ap.parse_args()


def parse_list_int(spec: str) -> List[int]:
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def parse_list_str(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip() != ""]


def filter_combos(only: Optional[str], diffs: List[str], seeds: List[int]) -> List[Tuple[str, int]]:
    if not only:
        return [(d, s) for d in diffs for s in seeds]
    pairs: List[Tuple[str, int]] = []
    parts = [x.strip() for x in only.split(";")]
    for p in parts:
        if ":" not in p:
            raise ValueError(f"--only format: diff:seeds, got {p}")
        diff, s_str = p.split(":", 1)
        diff = diff.strip()
        if diff not in diffs:
            raise ValueError(f"unknown difficulty: {diff}")
        if s_str.strip().lower() == "all":
            for s in seeds:
                pairs.append((diff, s))
        else:
            for tok in s_str.split(","):
                pairs.append((diff, int(tok)))
    return pairs


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def out_path_for(diff: str, seed: int, lam: float) -> Path:
    safe_lam = str(lam).rstrip("0").rstrip(".") if "." in str(lam) else str(lam)
    return OUT_ROOT / f"{diff}_s{seed}" / f"enhanced_l{safe_lam}.json"


def json_exists_and_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            _ = json.load(f)
        return True
    except Exception:
        return False


def run_one(diff: str, seed: int, lam: float, out_path: Path, args) -> int:
    root = project_root()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fallback: Explicitly add src/ to sys.path before running
    cmd = [
              sys.executable, "-c",
              "import sys; sys.path.append('" + str(root / 'src') + "'); import train_eval; train_eval.main()"
          ] + [
              "--model", "enhanced",
              "--difficulty", "hard",  # Increased to hard for more challenge
              "--seed", str(seed),
              "--n_samples", str(args.n_samples),  # NEW: Use args value instead of hardcoded
              "--epochs", str(args.epochs),        # NEW: Use args value instead of hardcoded
              "--batch", str(args.batch),          # NEW: Use args value instead of hardcoded
              "--T", "128",
              "--F", "10",  # Reduced features for added difficulty
              "--early_metric", "macro_f1",
              "--patience", "10",
              "--lambda_val", str(lam),
              "--out_json", str(out_path),
              # 透传温度参数
              "--temp_mode", args.temp_mode,
              "--temp_min", str(args.temp_min),
              "--temp_max", str(args.temp_max),
              "--temp_steps", str(args.temp_steps),
              "--val_split", str(args.val_split),
              # Increased noise parameters
              "--class_overlap", "0.8",
              "--gain_drift_std", "0.6",
              "--sc_corr_rho", "0.5",
              "--env_burst_rate", "0.2",
          ]
    if args.fixed_temp is not None:
        cmd += ["--fixed_temp", str(args.fixed_temp)]
    cmd += ["--label_noise_prob", "0.1", "--num_classes", "8"]
    print("RUN:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(root))
    return r.returncode


def load_metrics(path: Path, difficulty: str, seed: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    m = j.get("metrics", {})
    return {
        "model": j.get("meta", {}).get("model", "enhanced"),
        "difficulty": difficulty,
        "seed": str(seed),
        "macro_f1": m.get("macro_f1", None),
        "falling_f1": m.get("falling_f1", None),
        "mutual_misclass": m.get("mutual_misclass", None),
        "ece": m.get("ece", m.get("ece_cal", None)),
        "brier": m.get("brier", None),
        "overlap_stat": m.get("overlap_stat", None),
        "ece_raw": m.get("ece_raw", None),
        "ece_cal": m.get("ece_cal", None),
        "nll_raw": m.get("nll_raw", None),
        "nll_cal": m.get("nll_cal", None),
        "temperature": m.get("temperature", None),
        "lambda": j.get("args", {}).get("logit_l2", None),
        "path": str(path),
    }


def main():
    args = parse_args()
    if args.force and args.resume:
        raise SystemExit("Use either --force or --resume, not both.")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    diffs = parse_list_str(args.difficulties)
    seeds = parse_list_int(args.seeds)
    lambdas = parse_lambda_list(args.lambdas)
    combos = filter_combos(args.only, diffs, seeds) if args.only else [(d, s) for d in diffs for s in seeds]

    successes, failures = [], []

    for (d, s) in combos:
        for lam in lambdas:
            out_path = out_path_for(d, s, lam)
            if args.force:
                skip = False
            elif args.resume:
                skip = json_exists_and_valid(out_path)
            else:
                skip = False

            if skip:
                print(f"SKIP (exists): {out_path}")
            else:
                code = run_one(d, s, lam, out_path, args)
                if code != 0:
                    print(f"FAILED (code={code}): diff={d}, seed={s}, lambda={lam}")
                    failures.append({"difficulty": d, "seed": s, "lambda": lam, "out": str(out_path), "code": code})
                    continue
            try:
                row = load_metrics(out_path, d, s)
                successes.append(row)
            except Exception as e:
                print(f"LOAD FAILED: {out_path} ({e})")
                failures.append({"difficulty": d, "seed": s, "lambda": lam, "out": str(out_path), "code": "load_error"})

    if successes:
        csv_path = OUT_ROOT / "metrics_lambda_full.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(successes[0].keys()))
            w.writeheader()
            w.writerows(successes)
        print("Wrote", csv_path)
    else:
        print("No successful results to write.")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f)


if __name__ == "__main__":
    main()

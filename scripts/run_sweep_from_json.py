import os
import sys
import json
import argparse
import itertools
import subprocess
from datetime import datetime
from typing import Dict, Any, Iterable, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run train_eval.py over a JSON-defined sweep")
    ap.add_argument("--spec", required=True, help="Path to sweep spec JSON")
    ap.add_argument("--resume", action="store_true", help="Skip runs whose out_json already exists")
    ap.add_argument("--dry_run", action="store_true", help="Only print planned commands")
    ap.add_argument("--max", type=int, default=0, help="Optional cap on number of runs (0 = no cap)")
    return ap.parse_args()


def product_grid(grid: Dict[str, Iterable[Any]]):
    keys = sorted(grid.keys())
    values = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_cmd(spec: Dict[str, Any], model: str, seed: int, fixed: Dict[str, Any], combo: Dict[str, Any], out_dir: str) -> List[str]:
    exe = sys.executable  # Use current Python
    args = [exe, os.path.join("src", "train_eval.py"), "--model", model]
    # Basic/fixed fields
    for k, v in fixed.items():
        # Handle boolean CLI flags correctly: present flag when True, omit when False
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        else:
            args.extend([f"--{k}", str(v)])
    # Grid/overrides
    for k, v in combo.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        else:
            args.extend([f"--{k}", str(v)])
    # Seed and out
    args.extend(["--seed", str(seed)])
    diff = str(fixed.get("difficulty", "hard"))
    # Suffix describing key grid params for traceability
    tag_parts = []
    for k in sorted(combo.keys()):
        val = combo[k]
        safe = str(val).replace(".", "p").replace("-", "m")
        tag_parts.append(f"{k[:3]}{safe}")
    tag = ("_" + "_".join(tag_parts)) if tag_parts else ""
    out_name = f"paperA_{model}_{diff}_s{seed}{tag}.json"
    out_path = os.path.join(out_dir, out_name)
    args.extend(["--out_json", out_path])
    return args


def main():
    ns = parse_args()
    with open(ns.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    models = spec.get("models", ["enhanced"])  # type: List[str]
    seeds = spec.get("seeds", [0])              # type: List[int]
    fixed = spec.get("fixed", {})               # type: Dict[str, Any]
    grid = spec.get("grid", {})                 # type: Dict[str, Any]
    out_dir = spec.get("output_dir", "results_gpu")
    os.makedirs(out_dir, exist_ok=True)

    planned: List[List[str]] = []
    for m in models:
        for s in seeds:
            for combo in product_grid(grid):
                cmd = build_cmd(spec, m, int(s), fixed, combo, out_dir)
                if ns.resume:
                    try:
                        idx = cmd.index("--out_json")
                        out_json = cmd[idx + 1]
                    except Exception:
                        out_json = ""
                    if out_json and os.path.exists(out_json):
                        print(f"[skip] exists: {out_json}")
                        continue
                planned.append(cmd)

    if ns.max and len(planned) > ns.max:
        planned = planned[: ns.max]

    print(f"[plan] total runs = {len(planned)}")
    for cmd in planned:
        print(" ", " ".join(cmd))

    if ns.dry_run:
        return

    # Execute sequentially
    ok, fail = 0, 0
    for i, cmd in enumerate(planned, 1):
        print(f"[run {i}/{len(planned)}] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"[error] code={e.returncode} cmd={' '.join(cmd)}")
            fail += 1

    # Write manifest
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "spec": ns.spec,
        "total_planned": len(planned),
        "ok": ok,
        "fail": fail,
        "timestamp": ts,
    }
    man_path = os.path.join("results", "metrics", f"sweep_manifest_{ts}.json")
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[summary] {manifest}")


if __name__ == "__main__":
    main()



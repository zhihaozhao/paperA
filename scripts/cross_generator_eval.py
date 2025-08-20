import os
import sys
import json
import argparse
from typing import List, Dict, Any

import torch
import numpy as np

# Make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_synth import get_synth_loaders
from src.metrics import aggregate_classification_metrics
from src.models import build_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Cross-generator evaluation: train seed -> test on new generator seeds")
    ap.add_argument("--from_json", required=True, help="Path to a result JSON produced by src/train_eval.py")
    ap.add_argument("--test_seeds", type=str, default="1,2,3", help="Comma-separated test seeds, e.g., 1,2,3")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out_csv", type=str, default="")
    return ap.parse_args()


def _collect_logits_labels(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            logits_all.append(logits.cpu())
            y_all.append(yb.cpu())
    return torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0)


def main():
    args = parse_args()
    with open(args.from_json, "r", encoding="utf-8") as f:
        ref = json.load(f)

    # Reconstruct settings
    mname: str = ref["meta"]["model"]
    F: int = int(ref["meta"].get("F", ref["args"].get("F", 52)))
    T: int = int(ref["meta"].get("T", ref["args"].get("T", 128)))
    num_classes: int = int(ref["meta"].get("num_classes", ref["args"].get("num_classes", 8)))
    difficulty: str = ref["args"].get("difficulty", "hard")

    # Physical params
    sc_corr_rho = float(ref["args"].get("sc_corr_rho", 0.5))
    env_burst_rate = float(ref["args"].get("env_burst_rate", 0.2))
    gain_drift_std = float(ref["args"].get("gain_drift_std", 0.6))
    class_overlap = float(ref["args"].get("class_overlap", 0.8))
    label_noise_prob = float(ref["args"].get("label_noise_prob", 0.0))

    # Calibration temperature from source run (kept for cross evaluation)
    cal_T = None
    if "calibration" in ref and isinstance(ref["calibration"], dict):
        cal_T = ref["calibration"].get("temperature", None)

    ckpt = ref.get("best_ckpt", None)
    if not ckpt or not os.path.exists(ckpt):
        # Try to resolve relative to project root
        alt = os.path.join(os.getcwd(), ckpt or "")
        if os.path.exists(alt):
            ckpt = alt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(mname, F, num_classes, T=T).to(device)
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)

    test_seed_list: List[int] = [int(s) for s in args.test_seeds.split(',') if s.strip() != ""]

    rows: List[Dict[str, Any]] = []
    for tseed in test_seed_list:
        # Build only test loader from new generator seed
        _, _, test_loader = get_synth_loaders(
            batch=args.batch,
            difficulty=difficulty,
            seed=tseed,
            n=ref["args"].get("n_samples", 20000),
            T=T,
            F=F,
            sc_corr_rho=sc_corr_rho,
            env_burst_rate=env_burst_rate,
            gain_drift_std=gain_drift_std,
            class_overlap=class_overlap,
            label_noise_prob=label_noise_prob,
            num_classes=num_classes,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=bool(torch.cuda.is_available()),
        )

        logits, labels = _collect_logits_labels(model, test_loader, device)
        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()
        m = aggregate_classification_metrics(logits_np, labels_np, temperature=cal_T, num_classes=num_classes)
        rows.append({
            "source_json": os.path.relpath(args.from_json),
            "model": mname,
            "src_seed": int(ref["meta"].get("seed", ref["args"].get("seed", -1))),
            "test_seed": tseed,
            "macro_f1": m.get("macro_f1"),
            "ece_raw": m.get("ece_raw"),
            "ece_cal": m.get("ece_cal"),
            "nll_raw": m.get("nll_raw"),
            "nll_cal": m.get("nll_cal"),
            "brier": m.get("brier"),
            "mutual_misclass": m.get("mutual_misclass"),
            "temperature": cal_T,
            "difficulty": difficulty,
        })

    # Write CSV
    import csv
    out_csv = args.out_csv
    if not out_csv:
        base = os.path.splitext(os.path.basename(args.from_json))[0]
        out_csv = os.path.join("results", "metrics", f"crossgen_{base}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote cross-generator CSV: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()



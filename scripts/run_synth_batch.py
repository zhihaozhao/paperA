import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
import csv

def run_one(model, difficulty, seed, epochs=10):
    out_dir = Path("results/synth")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{model}_{difficulty}_s{seed}.json"

    cmd = [
        "python", "-m", "src.train_eval",
        "--model", model,
        "--difficulty", difficulty,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--out", str(out_json)
    ]
    print(f"[RUN] {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=True, text=True)
    print(ret.stdout)
    if ret.returncode != 0:
        print(ret.stderr)
        raise RuntimeError(f"Run failed: {model}-{difficulty}-s{seed}")

    if not out_json.exists():
        raise FileNotFoundError(f"Missing output: {out_json}")

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return out_json, data

def extract_row(out_json, data):
    args = data.get("args", {})
    metrics = data.get("metrics", {})
    meta = data.get("meta", {})
    row = {
        "time_end": meta.get("time_end", ""),
        "model": args.get("model", ""),
        "difficulty": args.get("difficulty", ""),
        "seed": args.get("seed", ""),
        "epochs": args.get("epochs", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "falling_f1": metrics.get("falling_f1", ""),
        "mutual_misclass": metrics.get("mutual_misclass", ""),
        "ece_reported": metrics.get("ece", ""),
        "brier": metrics.get("brier", ""),
        "temperature": metrics.get("temperature", ""),
        "ece_raw": metrics.get("ece_raw", ""),
        "ece_cal": metrics.get("ece_cal", ""),
        "nll_raw": metrics.get("nll_raw", ""),
        "nll_cal": metrics.get("nll_cal", ""),
        "out_json": str(out_json.as_posix()),
    }
    return row

def main():
    models = ["enhanced", "lstm", "tcn", "txf"]
    difficulties = ["easy", "mid", "hard"]
    seeds = [0, 1, 2]
    epochs = 10

    summary_path = Path("results/synth/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "time_end", "model", "difficulty", "seed", "epochs",
        "macro_f1", "falling_f1", "mutual_misclass",
        "ece_reported", "brier",
        "temperature", "ece_raw", "ece_cal", "nll_raw", "nll_cal",
        "out_json"
    ]

    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for model in models:
            for difficulty in difficulties:
                for seed in seeds:
                    try:
                        out_json, data = run_one(model, difficulty, seed, epochs=epochs)
                        row = extract_row(out_json, data)
                        writer.writerow(row)
                        csvfile.flush()
                        print(f"[OK] {model}-{difficulty}-s{seed} -> {out_json}")
                    except Exception as e:
                        print(f"[FAIL] {model}-{difficulty}-s{seed}: {e}")

if __name__ == "__main__":
    main()

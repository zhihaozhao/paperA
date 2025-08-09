from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Any

REGISTRY_PATH = Path("results/registry.csv")

def append_run_registry(row: Dict[str, Any]):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not REGISTRY_PATH.exists()
    # 扁平化简单字段
    flat = {k: v for k, v in row.items()}
    keys = [
        "time", "run_id", "phase", "exp", "ver",
        "model", "dataset", "seed",
        "difficulty", "epochs",
        "macro_f1", "falling_f1", "mutual_misclass",
        "ece", "brier", "overlap",
        "out_json"
    ]
    with open(REGISTRY_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        w.writerow({k: flat.get(k, "") for k in keys})
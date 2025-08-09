from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from .logger import dump_json_safe

class ExpRecorder:
    def __init__(self, out_dir: str | Path = "results", run_id: Optional[str] = None):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.history: List[Dict[str, Any]] = []

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        row = {"epoch": int(epoch)}
        row.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
        self.history.append(row)

    def save_history_csv(self, path: str | Path):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        if not self.history:
            return
        keys = sorted({k for r in self.history for k in r.keys()})
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.history:
                w.writerow(r)

    def set_final(self, args: Dict[str, Any], metrics: Dict[str, Any], extras: Optional[Dict[str, Any]] = None):
        payload = {
            "run_id": self.run_id,
            "args": args,
            "metrics": metrics,
        }
        if extras:
            payload.update(extras)
        self._final_payload = payload

    def save_final_json(self, path: str | Path):
        if not hasattr(self, "_final_payload"):
            raise RuntimeError("final payload is empty; call set_final() first")
        dump_json_safe(self._final_payload, path)
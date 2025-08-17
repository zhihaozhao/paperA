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

def append_run_registry(run_data: Dict[str, Any], registry_path: str = "registry.csv"):
    """
    Append run data to a global registry CSV file
    """
    registry_file = Path(registry_path)
    
    # Create header if file doesn't exist
    if not registry_file.exists():
        with open(registry_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=run_data.keys())
            writer.writeheader()
    
    # Append data
    with open(registry_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run_data.keys())
        writer.writerow(run_data)

def init_run(phase: str, exp: str, ver: str, desc: str = "", args: Optional[Dict[str, Any]] = None):
    """
    Initialize a new experiment run
    Returns logger and metadata
    """
    from datetime import datetime
    import uuid
    from .logger import setup_logger
    
    # Generate unique run ID
    run_id = f"{phase}-{exp}-{ver}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
    
    # Setup logger
    logger = setup_logger(run_id)
    
    # Create metadata
    meta = {
        "run_id": run_id,
        "phase": phase,
        "exp": exp,
        "ver": ver,
        "desc": desc,
        "time_start": datetime.now().isoformat(timespec="seconds"),
        "args": args or {}
    }
    
    logger.info(f"Initialized run: {run_id}")
    logger.info(f"Description: {desc}")
    
    return logger, meta
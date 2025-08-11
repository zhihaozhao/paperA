from __future__ import annotations
import os
import sys
import json
import socket
import logging
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Optional, Dict, Any

def _ensure_dir(p: str | Path) -> Path:
    pp = Path(p); pp.mkdir(parents=True, exist_ok=True); return pp

def get_git_hash(short: bool = True) -> str:
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def build_run_id(phase="P1", exp="E1", ver="V1", model="model", dataset="synth") -> str:
    return f"{phase}-{exp}-{ver}-{model}-{dataset}-{now_str()}-{get_git_hash(True)}"

def collect_env() -> Dict[str, Any]:
    info = {
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "pid": os.getpid(),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        })
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info

def dump_json_safe(obj: dict, path: str | Path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def setup_logger(run_id: str,
                 logs_dir: str | Path = "results/logs",
                 level: int = logging.INFO) -> logging.Logger:
    _ensure_dir(logs_dir)
    logger = logging.getLogger(run_id)
    logger.setLevel(level)
    logger.propagate = False
    # 清理旧 handlers（notebook 多次执行时避免重复）
    for h in list(logger.handlers):
        logger.removeHandler(h)
    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    # 文件
    log_path = Path(logs_dir) / f"{run_id}.txt"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger

def init_run(phase="P1", exp="E1", ver="V1",
             model="model", dataset="synth",
             logs_dir="results/logs",
             cli_args: Optional[dict] = None) -> tuple[logging.Logger, dict]:
    run_id = build_run_id(phase, exp, ver, model, dataset)
    logger = setup_logger(run_id, logs_dir)
    env = collect_env()
    meta = {
        "run_id": run_id,
        "phase": phase, "exp": exp, "ver": ver,
        "model": model, "dataset": dataset,
        "git": get_git_hash(True),
        "time_start": datetime.now().isoformat(timespec="seconds"),
        "env": env,
        "args": cli_args or {},
    }
    dump_json_safe(meta, Path(logs_dir) / f"{run_id}.meta.json")
    logger.info(f"Initialized run: {run_id}")
    logger.info(f"Env: {env}")
    if cli_args:
        logger.info(f"Args: {cli_args}")
    return logger, meta
from pathlib import Path
import json
import random
import numpy as np

def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not available

def dump_json_safe(obj, path: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def savefig_safe(path: str, plt_obj=None, bbox_inches="tight"):
    from matplotlib import pyplot as plt
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    (plt_obj or plt).savefig(p.as_posix(), bbox_inches=bbox_inches)

from pathlib import Path
import json

def dump_json_safe(obj, path: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def savefig_safe(path: str, plt_obj=None, bbox_inches="tight"):
    from matplotlib import pyplot as plt
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    (plt_obj or plt).savefig(p.as_posix(), bbox_inches=bbox_inches)

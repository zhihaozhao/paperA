import inspect
import json
import importlib

def check_get_synth_loaders():
    m = importlib.import_module("src.data_synth")
    fn = getattr(m, "get_synth_loaders", None)
    assert fn is not None, "get_synth_loaders not found"
    sig = inspect.signature(fn)
    required = ["batch", "difficulty", "seed", "n", "T", "F", "sc_corr_rho", "env_burst_rate", "gain_drift_std", "class_overlap"]
    for r in required:
        assert r in sig.parameters, f"missing param: {r}"
    print("[OK] get_synth_loaders signature aligned")

def check_out_json(path="results/synth/out_default.json"):
    try:
        meta = json.load(open(path, "r"))
    except FileNotFoundError:
        print(f"[WARN] {path} not found (run training first)")
        return
    for k in ["meta", "metrics", "best_ckpt"]:
        assert k in meta, f"missing key in out.json: {k}"
    print("[OK] out.json keys present")

if __name__ == "__main__":
    check_get_synth_loaders()
    check_out_json()

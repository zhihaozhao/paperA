import argparse
import json
import numpy as np
import torch

# from src.models import build_model
from src.train_eval import build_model  # 复用占位 build_model
from src.data_synth import get_synth_loaders


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def read_meta(out_json: str):
    meta = json.load(open(out_json, "r"))
    F = int(meta.get("meta", {}).get("F", 30))
    T = int(meta.get("meta", {}).get("T", 128))
    num_classes = int(meta.get("meta", {}).get("num_classes", 4))
    model = meta.get("meta", {}).get("model", "enhanced")
    temperature = float(meta.get("metrics", {}).get("temperature", 1.0))
    return dict(F=F, T=T, num_classes=num_classes, model=model, temperature=temperature)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--split", type=str, default="te", choices=["tr", "te"])
    ap.add_argument("--save_npz", type=str, default=None)
    args = ap.parse_args()

    meta = read_meta(args.out_json)
    F = meta["F"]; T = meta["T"]; num_classes = meta["num_classes"]; model_name = meta["model"]; temperature = meta["temperature"]

    # 数据与模型（按 out.json 元数据复原）
    tr_loader, te_loader = get_synth_loaders(F=F, T=T, n=2000, difficulty="mid")
    loader = te_loader if args.split == "te" else tr_loader

    model = build_model(model_name, F=F, num_classes=num_classes)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd); model.eval()

    logits_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits, _ = model(xb)
            logits_all.append(logits.numpy())
            labels_all.append(yb.numpy())

    logits = np.concatenate(logits_all, axis=0)
    logits_cal = logits / max(temperature, 1e-6)
    probs_cal = softmax_np(logits_cal)
    preds = probs_cal.argmax(axis=1)
    labels = np.concatenate(labels_all, axis=0)

    if args.save_npz:
        np.savez(args.save_npz,
                 logits_raw=logits,
                 logits_cal=logits_cal,
                 probs_cal=probs_cal,
                 preds=preds,
                 labels=labels)
        print(f"Saved {args.save_npz}")


if __name__ == "__main__":
    main()

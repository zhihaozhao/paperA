import argparse, json, os
import numpy as np, torch
from torch.optim import Adam
from src.models import build_model
from src.data_synth import get_synth_loaders
from src.metrics import compute_metrics
from src.calibration import ece, brier
def to_py(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    return o

def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_one_epoch(model, loader, opt, device):
    model.train(); total=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        _, loss = model(x,y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

def eval_model(model, loader, device, num_classes=4):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            logits,_ = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            ys.append(y.numpy()); ps.append(prob)
    y = np.concatenate(ys); p = np.vstack(ps)
    m = compute_metrics(y, p, num_classes=num_classes, positive_class=1)
    m["ece"] = ece(p, y, n_bins=15); m["brier"] = brier(p, y, num_classes=num_classes)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="enhanced")
    ap.add_argument("--logit_l2", type=float, default=0.05)
    ap.add_argument("--difficulty", type=str, default="mid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=str, default="results/synth/out.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    tr, te = get_synth_loaders(difficulty=args.difficulty, seed=args.seed)
    x, y = next(iter(tr))
    model = build_model(args.model, input_dim=x.shape[-1], num_classes=4, logit_l2=args.logit_l2).to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    best = None
    for ep in range(args.epochs):
        train_one_epoch(model, tr, opt, device)
        metr = eval_model(model, te, device, num_classes=4)
        if best is None or metr["macro_f1"] > best["macro_f1"]:
            best = metr
    # with open(args.out, "w") as f: json.dump(dict(args=vars(args), metrics=best), f, indent=2)
    payload = to_py(dict(args=vars(args), metrics=best))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
if __name__ == "__main__":
    main()

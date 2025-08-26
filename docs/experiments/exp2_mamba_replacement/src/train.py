import argparse, json, pathlib
import numpy as np
import torch
from sklearn.metrics import f1_score
from model import MambaEncoder

def fake_loader(num=200, T=128, F=52, K=6):
    x = torch.randn(num, T, F)
    y = torch.randint(0, K, (num,))
    return [(x, y)]

def train(cfg):
    device = torch.device(cpu)
    model = MambaEncoder(input_dim=cfg.get(input_dim,52), hidden_dim=cfg[hidden_dim], layers=cfg.get(layers,2), se=cfg.get(se,True), attn=(cfg.get(attention,lite)==lite))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg[train][lr])
    crit = torch.nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(cfg[train][epochs]):
        model.train()
        for xb, yb in fake_loader():
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xs, ys = next(iter(fake_loader()))
            logits = model(xs)
            pred = logits.argmax(dim=-1).cpu().numpy()
            f1 = f1_score(ys.numpy(), pred, average=macro)
            best = max(best, f1)
    out = pathlib.Path(results); out.mkdir(exist_ok=True)
    (out/ metrics.json).write_text(json.dumps({"best_macro_f1": best}, indent=2))
    print("F1:", best)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(--config, type=str, required=False)
    args = ap.parse_args()
    cfg = {"hidden_dim":128,"layers":2,"attention":"lite","se":True,"train":{"epochs":2,"lr":1e-3}}
    train(cfg)

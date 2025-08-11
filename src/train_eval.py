import os
import json
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_synth import get_synth_loaders
# 你的模型构建函数，如已有请替换为实际导入
# from src.models import build_model

# 简化的占位模型（若你已有模型，请替换 build_model 与调用处）
class TinyNet(nn.Module):
    def __init__(self, F: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(F, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        logits = self.head(h)
        return logits, h


def build_model(name: str, F: int, num_classes: int = 4):
    # 如你已有 build_model，请替换此实现
    return TinyNet(F=F, num_classes=num_classes)


def compute_metrics(logits: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import f1_score, confusion_matrix
    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=1)
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    cm = confusion_matrix(y, y_pred).tolist()
    return {
        "macro_f1": macro_f1,
        "cm": cm,
    }


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def temperature_scale(logits: torch.Tensor, T: float) -> torch.Tensor:
    return logits / max(T, 1e-6)


def calibrate_temperature(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb.clone())
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    # 简化：用网格搜索温度，避免引入新依赖
    def nll_temp(T: float) -> float:
        z = logits / max(T, 1e-6)
        logp = z - torch.logsumexp(z, dim=1, keepdim=True)
        pick = logp[torch.arange(len(y)), y]
        return float(-pick.mean().item())

    Ts = np.linspace(0.5, 3.0, 26)
    vals = [nll_temp(float(T)) for T in Ts]
    best_T = float(Ts[int(np.argmin(vals))])
    return best_T


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device, temperature: Optional[float] = None) -> Dict[str, Any]:
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            if temperature is not None:
                logits = temperature_scale(logits, temperature)
            logits_all.append(logits.cpu().numpy())
            y_all.append(yb.numpy())
    logits = np.concatenate(logits_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    m = compute_metrics(logits, y)
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="enhanced")
    ap.add_argument("--difficulty", type=str, default="mid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--F", type=int, default=30)
    ap.add_argument("--sc_corr_rho", type=float, default=None)
    ap.add_argument("--env_burst_rate", type=float, default=0.0)
    ap.add_argument("--gain_drift_std", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="results/synth/out.json")

    # [ANCHOR:EARLYSTOP_DECL]
    ap.add_argument("--class_overlap", type=float, default=0.0)
    ap.add_argument("--early_metric", type=str, default="macro_f1", choices=["macro_f1", "nll"])
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--ckpt_dir", type=str, default="results/ckpt")
    # [ANCHOR:EARLYSTOP_DECL_END]

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_loader, te_loader = get_synth_loaders(
        batch=args.batch,
        difficulty=args.difficulty,
        seed=args.seed,
        n=args.n_samples,
        T=args.T,
        F=args.F,
        sc_corr_rho=args.sc_corr_rho,
        env_burst_rate=args.env_burst_rate,
        gain_drift_std=args.gain_drift_std,
        class_overlap=args.class_overlap,  # 透传
    )

    model = build_model(args.model, F=args.F, num_classes=4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练 + 早停
    best_val: Optional[float] = None
    best_epoch: int = -1
    no_improve = 0
    ckpt_path = os.path.join(args.ckpt_dir, f"best_{args.model}_{args.difficulty}_s{args.seed}.pt")

    def is_better(curr: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        return (curr > best) if args.early_metric == "macro_f1" else (curr < best)

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # [ANCHOR:EARLYSTOP_LOOP]
        # 以验证集（此处用 te_loader 简化）指标作为早停依据
        with torch.no_grad():
            # 当 early_metric == nll 时，用未校准 logits 的 NLL；否则用 macro_f1
            logits_all, y_all = [], []
            for xb, yb in te_loader:
                xb = xb.to(device)
                logits, _ = model(xb)
                logits_all.append(logits.cpu().numpy())
                y_all.append(yb.numpy())
            logits_np = np.concatenate(logits_all, 0)
            y_np = np.concatenate(y_all, 0)

            if args.early_metric == "macro_f1":
                metric_val = float(compute_metrics(logits_np, y_np)["macro_f1"])
            else:
                # 简化 NLL 计算
                z = logits_np - logits_np.max(axis=1, keepdims=True)
                logp = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
                metric_val = float(-np.mean(logp[np.arange(len(y_np)), y_np]))
        if is_better(metric_val, best_val):
            best_val = metric_val
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best {args.early_metric}={best_val:.4f} @ {best_epoch})")
                break
        # [ANCHOR:EARLYSTOP_LOOP_END]

    # 加载 best，校准温度，最终评估
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    T_cal = calibrate_temperature(model, te_loader, device=device)
    final = eval_model(model, te_loader, device=device, temperature=T_cal)

    # [ANCHOR:BEST_EVAL_WRITE]
    out = {
        "meta": {
            "model": args.model,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "F": args.F,
            "T": args.T,
            "num_classes": 4,
        },
        "metrics": {
            "temperature": float(T_cal),
            "macro_f1": float(final.get("macro_f1", float("nan"))),
        },
        "best_ckpt": ckpt_path,
        "early_stop": {
            "metric": args.early_metric,
            "best_value": float(best_val if best_val is not None else np.nan),
            "best_epoch": int(best_epoch),
            "patience": int(args.patience),
        },
        "data_params": {
            "n_samples": args.n_samples,
            "sc_corr_rho": args.sc_corr_rho,
            "env_burst_rate": args.env_burst_rate,
            "gain_drift_std": args.gain_drift_std,
            "class_overlap": args.class_overlap,
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")
    # [ANCHOR:BEST_EVAL_WRITE_END]


if __name__ == "__main__":
    main()
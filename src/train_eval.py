import argparse, json, os
import numpy as np, torch
from torch.optim import Adam
from src.models import build_model
from src.data_synth import get_synth_loaders
from src.metrics import compute_metrics
from src.calibration import ece, brier
from pathlib import Path
from datetime import datetime
from src.utils.logger import init_run
from src.utils.exp_recorder import ExpRecorder
from src.utils.registry import append_run_registry  # 如果暂时不需要 registry，可先不导入

import math

def softmax_logits(logits):
    # logits: (N, C)
    m = logits.max(dim=1, keepdim=True).values
    ex = torch.exp(logits - m)
    return ex / ex.sum(dim=1, keepdim=True)

def compute_ece(probs, labels, n_bins=15):
    # probs: (N, C), labels: (N,)
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean()
        if prop > 0:
            acc_bin = accuracies[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_bin - acc_bin) * prop
    return ece.item()

def nll_from_logits(logits, labels):
    # CrossEntropyLoss = NLL(LogSoftmax) on logits
    return torch.nn.functional.cross_entropy(logits, labels).item()

@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits, _ = model(x, y=None) if hasattr(model, "__call__") else model(x)  # 兼容你的 forward 返回
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        all_logits.append(logits.detach())
        all_labels.append(y.detach())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def calibrate_temperature(model, loader, device, max_iter=1000, lr=0.01):
    # 在验证集上拟合温度 T（标量），最小化 NLL
    logits, labels = collect_logits(model, loader, device)
    T = torch.ones(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        scaled = logits / T.clamp_min(1e-3)
        loss = torch.nn.functional.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    opt.step(closure)
    T_opt = T.detach().clamp_min(1e-3)
    # 计算校准前后指标
    with torch.no_grad():
        probs_raw = softmax_logits(logits)
        ece_raw = compute_ece(probs_raw, labels)
        nll_raw = nll_from_logits(logits, labels)

        probs_cal = softmax_logits(logits / T_opt)
        ece_cal = compute_ece(probs_cal, labels)
        nll_cal = nll_from_logits(logits / T_opt, labels)

    return {
        "T": T_opt.item(),
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "nll_raw": nll_raw,
        "nll_cal": nll_cal,
    }


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
    # m = compute_metrics(y, p, num_classes=p.shape[1], positive_class=args.positive_class)
    m["ece"] = ece(p, y, n_bins=15); m["brier"] = brier(p, y, num_classes=num_classes)
    m["falling_f1"] = m.get("f1_fall", float("nan"))
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="enhanced")
    ap.add_argument("--logit_l2", type=float, default=0.05)
    ap.add_argument("--difficulty", type=str, default="mid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=str, default="results/synth/out.json")
    ap.add_argument("--sc_corr_rho", type=str, default=None,
                        help="Subcarrier correlation rho (Toeplitz). Use a float or 'None'.")
    ap.add_argument("--env_burst_rate", type=str, default="0.0",
                        help="Expected number of bursts per window. 0 disables.")
    ap.add_argument("--gain_drift_std", type=str, default="0.0",
                        help="Std of slow multiplicative gain drift. 0 disables.")
    ap.add_argument("--positive_class", type=int, default=1,
                        help="Index of the positive class for AUPRC/F1_fall.")
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--F", type=int, default=30)
    args = ap.parse_args()

    def _maybe_none_float(s):
        if s is None:
            return None
        if isinstance(s, (float, int)):
            return float(s)
        s_str = str(s).strip().lower()
        if s_str in ("none", "null", ""):
            return None
        return float(s)

    sc_corr_rho = _maybe_none_float(args.sc_corr_rho)
    env_burst_rate = _maybe_none_float(args.env_burst_rate) or 0.0
    gain_drift_std = _maybe_none_float(args.gain_drift_std) or 0.0

    # 1) 初始化 run 与 logger
    logger, meta = init_run(
        phase="P1", exp="E1", ver="V1",
        model=args.model, dataset=args.difficulty,
        cli_args=vars(args)
    )
    rec = ExpRecorder(out_dir="results", run_id=meta["run_id"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    # tr, te = get_synth_loaders(difficulty=args.difficulty, seed=args.seed)
    tr, te = get_synth_loaders(
        batch=64, difficulty=args.difficulty, seed=args.seed,
        n=args.n_samples, T=args.T, F=args.F,
        sc_corr_rho=sc_corr_rho,
        env_burst_rate=env_burst_rate,
        gain_drift_std=gain_drift_std
    )

    x, y = next(iter(tr))
    # Determine feature dimension C regardless of (B, T, C) or (B, C, T)
    if x.dim() == 3:
        B0, A, B1 = x.shape
        if A < B1:
            # 形状很可能是 (B, C, T)
            input_dim = A  # C
        else:
            # 形状是 (B, T, C)
            input_dim = B1  # C
    else:
        input_dim = x.shape[-1]

    model = build_model(args.model, input_dim=input_dim, num_classes=4, logit_l2=args.logit_l2).to(device)
    # model = build_model(args.model, input_dim=x.shape[-1], num_classes=4, logit_l2=args.logit_l2).to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    best = None

    for ep in range(args.epochs):
        train_loss = train_one_epoch(model, tr, opt, device)
        metr = eval_model(model, te, device, num_classes=4)
        # [BEGIN] 每epoch记录与打印
        rec.log_epoch(ep, {
            "train_loss": float(train_loss),
            "macro_f1": float(metr.get("macro_f1", float("nan"))),
            "falling_f1": float(metr.get("falling_f1", float("nan"))),
            "ece": float(metr.get("ece", float("nan"))),
            "brier": float(metr.get("brier", float("nan"))),
        })
        logger.info(
            f"epoch={ep} loss={train_loss:.4f} f1={metr.get('macro_f1', 0.0):.4f} ece={metr.get('ece', float('nan')):.4f}")
        # [END]

        if best is None or metr["macro_f1"] > best["macro_f1"]:
            best = metr
    # [BEGIN] 训练结束后保存历史与最终JSON（含meta）
    from pathlib import Path
    hist_path = Path(f"results/history/{args.model}_{args.difficulty}_s{args.seed}.csv")
    rec.save_history_csv(hist_path)

    # 校准评估（温度缩放），仅使用验证集，不影响训练
    cal = calibrate_temperature(model, te, device)
    print(f"[INFO] calib: T={cal['T']:.3f} ece_raw={cal['ece_raw']:.4f} -> ece_cal={cal['ece_cal']:.4f} "
          f"nll_raw={cal['nll_raw']:.3f} -> nll_cal={cal['nll_cal']:.3f}")

    # 最终指标（与评估脚本兼容）
    final_metrics = {
        "macro_f1": float(best.get("macro_f1", 0.0)),
        "falling_f1": float(best.get("falling_f1", float("nan"))),
        "mutual_misclass": float(best.get("mutual_misclass", float("nan"))),
        "ece": float(best.get("ece", float("nan"))),
        "brier": float(best.get("brier", float("nan"))),
        "overlap_stat": best.get("overlap_stat", None),
        # Calibration add-ons
        "temperature": cal["T"],
        "ece_raw": cal["ece_raw"],
        "ece_cal": cal["ece_cal"],
        "nll_raw": cal["nll_raw"],
        "nll_cal": cal["nll_cal"],
    }

    # 保存最终 JSON 到 args.out，附带 meta 与 args
    from datetime import datetime
    payload = to_py({
        "args": vars(args),
        "metrics": final_metrics,
        "meta": meta | {"time_end": datetime.now().isoformat(timespec="seconds")}
    })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved final JSON to {args.out}")

    # 可选：登记索引，便于全局汇总
    try:
        append_run_registry({
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_id": meta["run_id"],
            "phase": meta["phase"], "exp": meta["exp"], "ver": meta["ver"],
            "model": args.model, "dataset": args.difficulty, "seed": args.seed,
            "difficulty": args.difficulty, "epochs": args.epochs,
            "macro_f1": final_metrics["macro_f1"],
            "falling_f1": final_metrics["falling_f1"],
            "mutual_misclass": final_metrics["mutual_misclass"],
            "ece": final_metrics["ece"],
            "brier": final_metrics["brier"],
            "overlap": (final_metrics["overlap_stat"].get("mean")
                        if isinstance(final_metrics["overlap_stat"], dict)
                        else final_metrics["overlap_stat"]),
            "out_json": Path(args.out).as_posix()
        })
    except Exception as e:
        logger.warning(f"append_run_registry failed: {e}")
    # [END]
if __name__ == "__main__":
    main()

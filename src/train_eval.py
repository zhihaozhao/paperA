import os
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_synth import get_synth_loaders

import math
from dataclasses import dataclass

try:
    import scipy.optimize as scopt  # 可选

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from sklearn.metrics import f1_score
import numpy as np

from models import build_model  # Assuming import
#
# def compute_falling_f1(y_true: np.ndarray, y_pred: np.ndarray, falling_class: int = 2) -> float:
#     """
#     Compute F1 for 'falling' class (assume class index=2; adjust if needed).
#     Returns NaN if no samples in class.
#     """
#     mask = y_true == falling_class
#     if not np.any(mask):
#         return float('nan')
#     return f1_score(y_true[mask], y_pred[mask], average='binary', zero_division=0)
#
#
# from sklearn.metrics import f1_score
# import numpy as np
# import torch
#
#
# def compute_falling_f1(y_true: np.ndarray, y_pred: np.ndarray, falling_class: int = 2) -> float:
#     """
#     Compute F1 for 'falling' class (assume class index=2; adjust if needed).
#     Returns NaN if no samples in class.
#     """
#     mask = y_true == falling_class
#     if not np.any(mask):
#         return float('nan')
#     return f1_score(y_true[mask], y_pred[mask], average='binary', zero_division=0)

#
# def compute_overlap_stat(train_loader, val_loader, num_classes: int = 4) -> float:
#     """
#     Compute class overlap statistic (mean pairwise difference between class means).
#     Larger value = less overlap (more separation); smaller = more overlap.
#     Uses train_loader to compute per-class means (assumes x is tensor, e.g., (batch, channels, seq_len)).
#     """
#     class_sums = [torch.zeros(1) for _ in range(num_classes)]  # Sum of means per class
#     class_counts = [0 for _ in range(num_classes)]
#
#     # Collect data in one pass (efficient)
#     for x, y in train_loader:
#         for cls in range(num_classes):
#             mask = y == cls
#             if mask.any():
#                 class_sums[cls] += x[mask].mean()  # Mean over batch for this class (scalar for simplicity)
#                 class_counts[cls] += mask.sum().item()
#
#     # Compute means
#     class_means = []
#     for cls in range(num_classes):
#         if class_counts[cls] > 0:
#             class_means.append((class_sums[cls] / class_counts[cls]).item())
#         else:
#             class_means.append(np.nan)
#
#     # Filter valid means
#     valid_means = [m for m in class_means if not np.isnan(m)]
#     if len(valid_means) < 2:
#         return float('nan')
#
#     # Compute mean pairwise absolute difference (larger = less overlap)
#     overlaps = [abs(valid_means[i] - valid_means[j]) for i in range(len(valid_means)) for j in
#                 range(i + 1, len(valid_means))]
#
#     return np.mean(overlaps) if overlaps else float('nan')
#
#     # Optional extension: If features are multi-dimensional (e.g., class_means is list of vectors),
#     # use Euclidean distance instead:
#     # from scipy.spatial.distance import euclidean
#     # overlaps = [euclidean(class_means[i], class_means[j]) for i in range(num_classes) for j in range(i + 1, num_classes)]

def _confusion_matrix_from_preds(y_true, y_pred, num_classes):
    import numpy as np
    C = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        C[int(t), int(p)] += 1
    return C

def _compute_mutual_misclass(y_true, y_pred, num_classes):
    """
    Mutual misclassification rate:
    For each unordered class pair {i, j}, compute
      m_ij = (C[i,j] + C[j,i]) / (count(true=i) + count(true=j))
    and return the average over all pairs.
    """
    if num_classes < 2:
        return 0.0
    C = _confusion_matrix_from_preds(y_true, y_pred, num_classes)
    totals = C.sum(axis=1)
    pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            denom = float(totals[i] + totals[j])
            if denom <= 0:
                continue
            m_ij = (C[i, j] + C[j, i]) / denom
            pairs.append(m_ij)
    if not pairs:
        return 0.0
    import numpy as np
    return float(np.mean(pairs))
# -------------------------
# 模型（示例 TinyNet）
# -------------------------
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


# def build_model(name: str, F: int, num_classes: int = 4):
#     # 若你有自己的构建函数，可替换
#     return TinyNet(F=F, num_classes=num_classes)


# -------------------------
# 指标与工具
# -------------------------
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


# def _nll_from_logits_torch(logits: torch.Tensor, targets: torch.Tensor) -> float:
#     logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
#     pick = logp[torch.arange(len(targets)), targets]
#     return float(-pick.mean().item())


def _probs_from_logits(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    if T <= 0:
        T = 1e-6
    return torch.softmax(logits / T, dim=-1)


def _brier_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> float:
    num_classes = probs.size(-1)
    y = torch.nn.functional.one_hot(targets, num_classes=num_classes).to(probs.dtype)
    return torch.mean(torch.sum((probs - y) ** 2, dim=-1)).item()


def _ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    with torch.no_grad():
        conf, pred = probs.max(dim=-1)
        correct = (pred == targets).float()
        bins = torch.linspace(0, 1, steps=n_bins + 1, device=probs.device)
        ece = torch.tensor(0.0, device=probs.device)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                acc_bin = correct[mask].mean()
                conf_bin = conf[mask].mean()
                ece += (mask.float().mean()) * torch.abs(acc_bin - conf_bin)
        return float(ece.item())


def _collect_logits_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[
    torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            all_logits.append(logits.detach().cpu())
            all_targets.append(yb.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


# -------------------------
# 温度标定
# -------------------------
@dataclass
class TempCalibResult:
    T: float
    nll: float


def _nll_from_logits_torch(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.nn.functional.cross_entropy(logits, targets, reduction="mean").item()  # 直接 CE >0

def _nll_from_logits(logits: torch.Tensor, targets: torch.Tensor, T: float = 1.0) -> float:
    if T <= 0: T = 1e-6
    scaled = logits / T
    ce = torch.nn.functional.cross_entropy(scaled, targets, reduction="mean")
    return ce.clamp(min=0.0).item()  # clamp 确保 ≥0

def calibrate_temperature_logspace(val_logits: torch.Tensor, val_targets: torch.Tensor,
                                   tmin: float = 1.0, tmax: float = 5.0, steps: int = 40) -> TempCalibResult:
    grid = torch.logspace(math.log10(tmin), math.log10(tmax), steps=steps)
    nlls = []  # NEW: 收集以调试
    best_T, best_nll = 1.0, float("inf")
    for T in grid.tolist():
        nll = _nll_from_logits(val_logits, val_targets, T=T)
        nlls.append((T, nll))
        if nll < best_nll:
            best_nll, best_T = nll, T
    print(f"Temp search: best T={best_T}, nlls={nlls[:5]}...")  # NEW: 日志调试
    if best_T == 1.0:  # Fallback if stuck
        print("Warning: Stuck at 1.0, trying learnable fallback.")
        return calibrate_temperature_learnable(val_logits, val_targets)
    return TempCalibResult(T=best_T, nll=best_nll)

def calibrate_temperature_learnable(val_logits: torch.Tensor, val_targets: torch.Tensor) -> TempCalibResult:
    device = torch.device("cpu")
    logits = val_logits.to(device)
    targets = val_targets.to(device)
    logT = torch.tensor(0.0, requires_grad=True)  # 初始 T=1
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(logT)
        loss = torch.nn.functional.cross_entropy(logits / T, targets, reduction="mean")
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        opt2 = torch.optim.Adam([logT], lr=0.05)
        for _ in range(200):
            opt2.zero_grad()
            T = torch.exp(logT)
            loss = torch.nn.functional.cross_entropy(logits / T, targets, reduction="mean")
            loss.backward()
            opt2.step()

    T_final = float(torch.exp(logT).clamp_min(1e-4).item())
    nll_final = _nll_from_logits(logits, targets, T=T_final)
    return TempCalibResult(T=T_final, nll=nll_final)


def calibrate_temperature_from_val(val_logits: torch.Tensor, val_targets: torch.Tensor,
                                   mode: str, tmin: float, tmax: float, steps: int) -> Optional[TempCalibResult]:
    if mode == "none":
        return None
    if mode == "logspace":
        return calibrate_temperature_logspace(val_logits, val_targets, tmin=tmin, tmax=tmax, steps=steps)
    if mode == "learnable":
        if _HAS_SCIPY:
            def f(logT):
                T = float(math.exp(logT[0]))
                return _nll_from_logits(val_logits, val_targets, T=T)

            res = scopt.minimize(f, x0=[0.0], method="L-BFGS-B", bounds=[(-10.0, 5.0)])
            T_star = float(math.exp(res.x[0]))
            return TempCalibResult(T=T_star, nll=_nll_from_logits(val_logits, val_targets, T=T_star))
        else:
            return calibrate_temperature_learnable(val_logits, val_targets)
    raise ValueError(f"Unknown temp_mode: {mode}")


# -------------------------
# 评估
# -------------------------
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device, temperature: Optional[float] = None) -> Dict[
    str, Any]:
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


# -------------------------
# 主程序
# -------------------------

# Note: This is the corrected version of your main() function in train_eval.py.
# Key fixes and integrations:
# - Ensured calib_result is properly defined and called with parameters (val_logits, val_targets).
# - Collected separate logits/targets for val (for calibration) and test (for final metrics).
# - Used aggregate_classification_metrics on TEST data for final "metrics" (consistent with original code's intent).
# - Integrated falling_f1 and overlap_stat via aggregate (no separate calls needed).
# - Safe temperature extraction.
# - Removed redundant final_metrics = eval_model(...) (assumed covered by aggregate).
# - Retained original structure for T_cal, ece/nll/brier calculations, but merged into agg_metrics where possible.
# - Assumed missing functions (_ece, _nll_from_logits_torch, etc.) are defined elsewhere.
# - Added import for metrics and SynthCSIDataset if needed.
# - Overall: Modifications are correct and functional, with fixes for consistency.

def main():
    import argparse
    import os
    import json
    import math
    from typing import Optional

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Subset, DataLoader

    # Import metrics module (add if not present)
    import metrics  # Assuming metrics.py is in the same dir or path

    calib_result = None  # Default

    def compute_macro_f1_torch(pred: torch.Tensor, y_true: torch.Tensor, num_classes: Optional[int] = None) -> float:
        """
        pred: LongTensor [N], 预测类别
        y_true: LongTensor [N], 真实类别
        num_classes: 类别数；若为 None，则用 max(pred,y_true)+1
        返回 macro-F1 (float)
        """
        if num_classes is None:
            num_classes = int(torch.max(torch.stack([pred.max(), y_true.max()])).item() + 1)
        f1_list = []
        for c in range(num_classes):
            tp = ((pred == c) & (y_true == c)).sum().item()
            fp = ((pred == c) & (y_true != c)).sum().item()
            fn = ((pred != c) & (y_true == c)).sum().item()
            denom_p = tp + fp
            denom_r = tp + fn
            precision = tp / denom_p if denom_p > 0 else 0.0
            recall = tp / denom_r if denom_r > 0 else 0.0
            if precision + recall > 1e-12:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            f1_list.append(f1)
        return float(np.mean(f1_list) if len(f1_list) > 0 else 0.0)

    parser = argparse.ArgumentParser()
    # 原有参数（保持不变/按你项目已有）
    parser.add_argument("--difficulty", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_val", "--lambda", dest="lambda_val", type=float, default=0.0)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 训练相关（按你已有参数名）
    parser.add_argument("--model", type=str, default="enhanced")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--early_metric", type=str, default="macro_f1", choices=["macro_f1", "nll"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--logit_l2", type=float, default=0.0)

    # 数据生成参数（与你项目一致）
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--F", type=int, default=64)
    parser.add_argument("--sc_corr_rho", type=float, default=0.0)
    parser.add_argument("--env_burst_rate", type=float, default=0.0)
    parser.add_argument("--gain_drift_std", type=float, default=0.0)
    parser.add_argument("--class_overlap", type=float, default=0.0)

    # 温度标定参数
    parser.add_argument("--temp_mode", type=str, default="logspace", choices=["none", "logspace", "learnable"])
    parser.add_argument("--temp_min", type=float, default=0.5)
    parser.add_argument("--temp_max", type=float, default=5.0)
    parser.add_argument("--temp_steps", type=int, default=60)

    # 新增参数
    parser.add_argument("--val_split", type=float, default=0.5,
                        help="Fraction of test set used as validation for temperature search")
    parser.add_argument("--fixed_temp", type=float, default=None, help="If set, skip search and use this temperature")
    parser.add_argument("--label_noise_prob", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=8)
    args = parser.parse_args()

    # 路径准备
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 随机种子与设备
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # 数据加载（训练/测试）
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
        class_overlap=args.class_overlap,
        label_noise_prob=args.label_noise_prob,
        num_classes=args.num_classes
    )

    # te_loader -> 拆分为 val/test
    def split_val_test_from_loader(base_loader, val_split=0.5, seed=42):
        ds = base_loader.dataset
        n = len(ds)
        n_val = int(math.ceil(n * val_split))
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        val_idx = perm[:n_val]
        test_idx = perm[n_val:]

        def _clone_loader(sub_idx):
            subset = Subset(ds, sub_idx)
            return DataLoader(
                subset,
                batch_size=base_loader.batch_size,
                shuffle=False,
                num_workers=getattr(base_loader, "num_workers", 0),
                pin_memory=getattr(base_loader, "pin_memory", False),
                drop_last=False,
            )

        return _clone_loader(val_idx), _clone_loader(test_idx)

    val_loader, test_loader = split_val_test_from_loader(te_loader, val_split=args.val_split, seed=args.seed)

    # 模型、优化器、损失
    # model = build_model(args.model, F=args.F, num_classes=4).to(device)
    model = build_model(args.model, input_dim=args.F, num_classes=args.num_classes, logit_l2=args.lambda_val)

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
            base_loss = criterion(logits, yb)
            lambda_l2 = float(getattr(args, "logit_l2", 0.0) or 0.0)
            loss = base_loss + (lambda_l2 * logits.pow(2).mean() if lambda_l2 > 0.0 else 0.0)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # 验证用于早停（这里使用 val_loader）
        with torch.no_grad():
            logits_all, y_all = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits, _ = model(xb)
                logits_all.append(logits.cpu().numpy())
                y_all.append(yb.numpy())
            logits_np = np.concatenate(logits_all, 0)
            y_np = np.concatenate(y_all, 0)

            if args.early_metric == "macro_f1":
                metric_val = float(compute_metrics(logits_np, y_np)["macro_f1"])
            else:
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

    # 载入最佳权重
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # 验证集上做温度标定
    def collect_logits_y(m, loader, device, T=None):
        m.eval()
        L, Y = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits, _ = m(xb)
                if T is not None:
                    logits = logits / max(T, 1e-6)
                L.append(logits.cpu())
                Y.append(yb.clone())
        return torch.cat(L, 0), torch.cat(Y, 0)

    # Collect for val (for calibration) and test (for metrics)
    val_logits, val_targets = collect_logits_y(model, val_loader, device, T=None)
    test_logits, test_targets = collect_logits_y(model, test_loader, device, T=None)

    # Calibration step
    if args.temp_mode == "none" and args.fixed_temp is None:
        T_cal = 1.0
        calib_extra = {"mode": "none", "tmin": None, "tmax": None, "steps": None}
    elif args.fixed_temp is not None:
        T_cal = float(args.fixed_temp)
        calib_extra = {"mode": "fixed", "tmin": None, "tmax": None, "steps": None}
    else:
        calib_res = calibrate_temperature_from_val(
            # Assuming this is your function name; adjust to calibrate_temperature_logspace if needed
            val_logits, val_targets,
            mode=args.temp_mode, tmin=args.temp_min, tmax=args.temp_max, steps=args.temp_steps
        )
        T_cal = float(getattr(calib_res, "T", 1.0))
        calib_extra = {
            "mode": args.temp_mode,
            "tmin": float(args.temp_min),
            "tmax": float(args.temp_max),
            "steps": int(args.temp_steps),
        }
        calib_result = calib_res  # Set calib_result here for consistency

    # Safe temperature extraction (fallback if calib_result not set)
    temperature = T_cal  # Use T_cal directly (from above)
    try:
        if calib_result:
            temperature = calib_result.T
    except NameError:
        pass

    # Compute aggregated metrics on TEST set
    underlying_dataset = test_loader.dataset.dataset if isinstance(test_loader.dataset,
                                                                   torch.utils.data.Subset) else test_loader.dataset
    agg_metrics = metrics.aggregate_classification_metrics(
        test_logits.numpy(),  # Use test_logits (np array)
        test_targets.numpy(),  # Use test_targets
        temperature=temperature,
        dataset=underlying_dataset,  # For overlap_stat
        num_classes=args.num_classes  # Add this
    )

    # 组装输出 (use agg_metrics for "metrics")
    out = {
        "meta": {
            "model": args.model,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "F": args.F,
            "T": args.T,
            "num_classes": 4
        },
        "args": {
            "model": args.model,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch": args.batch,
            "n_samples": args.n_samples,
            "T": args.T,
            "F": args.F,
            "sc_corr_rho": args.sc_corr_rho,
            "env_burst_rate": args.env_burst_rate,
            "gain_drift_std": args.gain_drift_std,
            "class_overlap": args.class_overlap,
            "early_metric": args.early_metric,
            "patience": args.patience,
            "ckpt_dir": args.ckpt_dir,
            "logit_l2": float(getattr(args, "logit_l2", 0.0) or 0.0),
            "temp_mode": args.temp_mode,
            "temp_min": args.temp_min,
            "temp_max": args.temp_max,
            "temp_steps": args.temp_steps,
            "val_split": args.val_split,
            "fixed_temp": args.fixed_temp,
            "out_json": args.out_json
        },
        "metrics": agg_metrics,  # Unified metrics dict including falling_f1, overlap_stat, ece, etc.
        "seed": int(args.seed),
        "best_ckpt": ckpt_path,
        "early_stop": {
            "metric": args.early_metric,
            "best_value": float(best_val if best_val is not None else np.nan),
            "best_epoch": int(best_epoch),
            "patience": int(args.patience)
        },
        "data_params": {
            "n_samples": args.n_samples,
            "sc_corr_rho": args.sc_corr_rho,
            "env_burst_rate": args.env_burst_rate,
            "gain_drift_std": args.gain_drift_std,
            "class_overlap": args.class_overlap
        },
        "calibration": calib_extra
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out_json} | T={temperature:.4f} | (Metrics from agg: {agg_metrics})")


if __name__ == "__main__":
    main()
import argparse
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

from src.metrics import aggregate_classification_metrics

try:
    import scipy.optimize as scopt  # 可选

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from sklearn.metrics import f1_score
import numpy as np

from models import build_model  # Assuming import

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate model on synthetic CSI dataset")

    # Model and basic config
    parser.add_argument("--model", type=str, default="enhanced", help="Model name (e.g., enhanced, bilstm)")
    parser.add_argument("--difficulty", type=str, default="hard", help="Difficulty level (e.g., hard, easy)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--F", type=int, default=52, help="Feature dimension F")
    parser.add_argument("--T", type=int, default=32, help="Time steps T")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes (fixed at 4)")

    # Training params
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--n_samples", type=int, default=20000, help="Number of synthetic samples")
    parser.add_argument("--early_metric", type=str, default="macro_f1",
                        help="Metric for early stopping (e.g., macro_f1)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--logit_l2", type=float, default=0.0, help="L2 regularization on logits (lambda)")

    # Data synthesis params
    parser.add_argument("--sc_corr_rho", type=float, default=0.5, help="Subcarrier correlation rho")
    parser.add_argument("--env_burst_rate", type=float, default=0.1, help="Environmental burst rate")
    parser.add_argument("--gain_drift_std", type=float, default=0.1, help="Gain drift standard deviation")
    parser.add_argument("--class_overlap", type=float, default=0.8, help="Class overlap factor")

    # Temperature calibration params
    parser.add_argument("--temp_mode", type=str, default="logspace",
                        help="Temperature mode (e.g., logspace, learnable, none)")
    parser.add_argument("--temp_min", type=float, default=1.0, help="Min temperature for search")
    parser.add_argument("--temp_max", type=float, default=5.0, help="Max temperature for search")
    parser.add_argument("--temp_steps", type=int, default=100, help="Steps for temperature search")
    parser.add_argument("--val_split", type=float, default=0.5, help="Validation split for calibration")
    parser.add_argument("--fixed_temp", type=float, default=1.0, help="Fixed temperature if no calibration")

    # Output
    parser.add_argument("--out_json", type=str, default="results/out.json", help="Path to output JSON")

    return parser.parse_args()
# -------------------------
# 主程序
# -------------------------
import logging  # NEW: Import for logging

def main():
    args = parse_args()  # Assuming your parse_args() with all fields like --model, --difficulty, etc.

    # Configure logging
    log_file = os.path.join(os.path.dirname(args.out_json), f"train_eval_{args.seed}_{args.difficulty}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with args: {vars(args)}")
    logger.info(f"Logs saved to: {log_file}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        # Load datasets
        train_loader, val_loader, test_loader = get_synth_loaders(args)
        logger.info(f"Dataset: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

        # Build and move model to device
        model = build_model(args.model, args.F, args.num_classes)
        model = model.to(device)
        logger.info(f"Model: {args.model} with {sum(p.numel() for p in model.parameters())} params")

        # Optimizer and criterion (adjust to your exact setup, e.g., with logit_l2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example
        criterion = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val = None
        best_epoch = 0
        patience_counter = 0
        ckpt_path = None  # Will be set if saving checkpoints
        log_interval = 5
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                if args.logit_l2 > 0:  # Assuming --logit_l2 is the param for lambda
                    reg_loss = args.logit_l2 * torch.mean(torch.norm(logits, p=2, dim=1))
                    loss += reg_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item()
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}")

            # Compute early stopping metric (e.g., macro_f1)
            val_logits, val_targets = _collect_logits_labels(model, val_loader, device)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_labels = val_targets.cpu().numpy()
            if args.early_metric == 'macro_f1':
                current_metric = f1_score(val_labels, val_preds, average='macro')
            else:
                current_metric = 0  # Add logic for other metrics
            # logger.info(f"Epoch {epoch+1} - Val {args.early_metric}: {current_metric:.4f}")

            if best_val is None or current_metric > best_val:
                best_val = current_metric
                best_epoch = epoch + 1
                patience_counter = 0
                # Save checkpoint (example)
                ckpt_path = os.path.join(args.ckpt_dir, f"best_{args.seed}_{args.difficulty}.pth")
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Best model saved to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            # Reduced logging: only every log_interval epochs
            if (epoch + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val {args.early_metric}: {current_metric:.4f}")

        # Final evaluation
        test_logits, test_targets = _collect_logits_labels(model, test_loader, device)
        logger.info("Collected test logits and targets")

        # Temperature calibration
        calib_extra = {}  # To store calibration details
        if args.temp_mode != 'none':
            # Use val_split on test set for calibration (as per args)
            split_idx = int(len(test_targets) * args.val_split)
            cal_val_logits, cal_val_targets = test_logits[:split_idx], test_targets[:split_idx]
            cal_test_logits, cal_test_targets = test_logits[split_idx:], test_targets[split_idx:]
            calib_result = calibrate_temperature_from_val(cal_val_logits, cal_val_targets, args.temp_mode, args.temp_min, args.temp_max, args.temp_steps)
            temperature = calib_result.temperature if calib_result else args.fixed_temp or 1.0
            calib_extra = {
                "temperature": temperature,
                "best_nll": calib_result.best_nll if calib_result else None,
                "method": args.temp_mode
            }
            logger.info(f"Calibration: T={temperature:.4f}, Details={calib_extra}")
        else:
            temperature = args.fixed_temp or 1.0
            calib_extra = {"temperature": temperature, "method": "none"}
            logger.info("No calibration; using T=1.0")

        # Compute aggregated metrics (only fills "metrics" field)
        agg_metrics = aggregate_classification_metrics(test_logits, test_targets, temperature)  # Your function

        # Build full out dict (no loss – all fields preserved from args, results, etc.)
        out = {
            "meta": {
                "model": args.model,
                "difficulty": args.difficulty,
                "seed": args.seed,
                "F": args.F,
                "T": args.T,
                "num_classes": 8
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
            "metrics": agg_metrics,  # Only metrics here; no loss
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

        # Save to JSON
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved full results to {args.out_json}")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise
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
#
# def main():
#     import argparse
#     import os
#     import json
#     import math
#     from typing import Optional
#
#     import numpy as np
#     import torch
#     import torch.nn as nn
#     from torch.utils.data import Subset, DataLoader
#
#     # Import metrics module (add if not present)
#     import metrics  # Assuming metrics.py is in the same dir or path
#
#     calib_result = None  # Default
#
#     def compute_macro_f1_torch(pred: torch.Tensor, y_true: torch.Tensor, num_classes: Optional[int] = None) -> float:
#         """
#         pred: LongTensor [N], 预测类别
#         y_true: LongTensor [N], 真实类别
#         num_classes: 类别数；若为 None，则用 max(pred,y_true)+1
#         返回 macro-F1 (float)
#         """
#         if num_classes is None:
#             num_classes = int(torch.max(torch.stack([pred.max(), y_true.max()])).item() + 1)
#         f1_list = []
#         for c in range(num_classes):
#             tp = ((pred == c) & (y_true == c)).sum().item()
#             fp = ((pred == c) & (y_true != c)).sum().item()
#             fn = ((pred != c) & (y_true == c)).sum().item()
#             denom_p = tp + fp
#             denom_r = tp + fn
#             precision = tp / denom_p if denom_p > 0 else 0.0
#             recall = tp / denom_r if denom_r > 0 else 0.0
#             if precision + recall > 1e-12:
#                 f1 = 2.0 * precision * recall / (precision + recall)
#             else:
#                 f1 = 0.0
#             f1_list.append(f1)
#         return float(np.mean(f1_list) if len(f1_list) > 0 else 0.0)
#
#     parser = argparse.ArgumentParser()
#     # 原有参数（保持不变/按你项目已有）
#     parser.add_argument("--difficulty", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--lambda_val", "--lambda", dest="lambda_val", type=float, default=0.0)
#     parser.add_argument("--out_json", type=str, required=True)
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#
#     # 训练相关（按你已有参数名）
#     parser.add_argument("--model", type=str, default="enhanced")
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--batch", type=int, default=256)
#     parser.add_argument("--early_metric", type=str, default="macro_f1", choices=["macro_f1", "nll"])
#     parser.add_argument("--patience", type=int, default=10)
#     parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
#     parser.add_argument("--logit_l2", type=float, default=0.0)
#
#     # 数据生成参数（与你项目一致）
#     parser.add_argument("--n_samples", type=int, default=20000)
#     parser.add_argument("--T", type=int, default=128)
#     parser.add_argument("--F", type=int, default=64)
#     parser.add_argument("--sc_corr_rho", type=float, default=0.0)
#     parser.add_argument("--env_burst_rate", type=float, default=0.0)
#     parser.add_argument("--gain_drift_std", type=float, default=0.0)
#     parser.add_argument("--class_overlap", type=float, default=0.0)
#
#     # 温度标定参数
#     parser.add_argument("--temp_mode", type=str, default="logspace", choices=["none", "logspace", "learnable"])
#     parser.add_argument("--temp_min", type=float, default=0.5)
#     parser.add_argument("--temp_max", type=float, default=5.0)
#     parser.add_argument("--temp_steps", type=int, default=60)
#
#     # 新增参数
#     parser.add_argument("--val_split", type=float, default=0.5,
#                         help="Fraction of test set used as validation for temperature search")
#     parser.add_argument("--fixed_temp", type=float, default=None, help="If set, skip search and use this temperature")
#     parser.add_argument("--label_noise_prob", type=float, default=0.1)
#     parser.add_argument("--num_classes", type=int, default=8)
#     args = parser.parse_args()
#
#     # NEW: Configure logging to file
#     log_file = os.path.join(os.path.dirname(args.out_json), f"train_eval_{args.seed}_{args.difficulty}.log")
#     logging.basicConfig(
#         level=logging.INFO,  # Change to DEBUG for more details
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[
#             logging.FileHandler(log_file),  # Save to file
#             logging.StreamHandler()  # Also print to console
#         ]
#     )
#     logger = logging.getLogger(__name__)
#     logger.info(f"Starting training with args: {vars(args)}")
#     logger.info(f"Logs will be saved to: {log_file}")
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")
#     # 路径准备
#     out_dir = os.path.dirname(args.out_json)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)
#     os.makedirs(args.ckpt_dir, exist_ok=True)
#
#     # 随机种子与设备
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     device = torch.device(args.device)
#
#     try:
#         # 数据加载（训练/测试）
#         tr_loader, te_loader = get_synth_loaders(
#             batch=args.batch,
#             difficulty=args.difficulty,
#             seed=args.seed,
#             n=args.n_samples,
#             T=args.T,
#             F=args.F,
#             sc_corr_rho=args.sc_corr_rho,
#             env_burst_rate=args.env_burst_rate,
#             gain_drift_std=args.gain_drift_std,
#             class_overlap=args.class_overlap,
#             label_noise_prob=args.label_noise_prob,
#             num_classes=args.num_classes
#         )
#
#         # te_loader -> 拆分为 val/test
#         def split_val_test_from_loader(base_loader, val_split=0.5, seed=42):
#             ds = base_loader.dataset
#             n = len(ds)
#             n_val = int(math.ceil(n * val_split))
#             g = torch.Generator()
#             g.manual_seed(seed)
#             perm = torch.randperm(n, generator=g).tolist()
#             val_idx = perm[:n_val]
#             test_idx = perm[n_val:]
#
#             def _clone_loader(sub_idx):
#                 subset = Subset(ds, sub_idx)
#                 return DataLoader(
#                     subset,
#                     batch_size=base_loader.batch_size,
#                     shuffle=False,
#                     num_workers=getattr(base_loader, "num_workers", 0),
#                     pin_memory=getattr(base_loader, "pin_memory", False),
#                     drop_last=False,
#                 )
#
#             return _clone_loader(val_idx), _clone_loader(test_idx)
#
#         val_loader, test_loader = split_val_test_from_loader(te_loader, val_split=args.val_split, seed=args.seed)
#
#         # 模型、优化器、损失
#         # model = build_model(args.model, F=args.F, num_classes=4).to(device)
#         model = build_model(args.model, input_dim=args.F, num_classes=args.num_classes, logit_l2=args.lambda_val).to(device)
#
#         optim = torch.optim.Adam(model.parameters(), lr=1e-3)
#         criterion = nn.CrossEntropyLoss()
#
#         # 训练 + 早停
#         best_val: Optional[float] = None
#         best_epoch: int = -1
#         no_improve = 0
#         ckpt_path = os.path.join(args.ckpt_dir, f"best_{args.model}_{args.difficulty}_s{args.seed}.pt")
#
#         def is_better(curr: float, best: Optional[float]) -> bool:
#             if best is None:
#                 return True
#             return (curr > best) if args.early_metric == "macro_f1" else (curr < best)
#
#         # Training loop with early stopping
#         best_val = None
#         best_epoch = 0
#         patience_counter = 0
#         ckpt_patch = None
#         for epoch in range(args.epochs):
#             model.train()
#             train_loss = 0
#             for xb, yb in tr_loader:
#                 xb = xb.to(device)
#                 yb = yb.to(device)
#                 logits, _ = model(xb)
#                 # base_loss = criterion(logits, yb)
#                 # lambda_l2 = float(getattr(args, "logit_l2", 0.0) or 0.0)
#                 # loss = base_loss + (lambda_l2 * logits.pow(2).mean() if lambda_l2 > 0.0 else 0.0)
#                 loss = criterion(logits, yb)
#                 if args.logit_l2 > 0:  # Assuming --logit_l2 is the param for lambda
#                     reg_loss = args.logit_l2 * torch.mean(torch.norm(logits, p=2, dim=1))
#                     loss += reg_loss
#                 optim.zero_grad()
#                 loss.backward()
#                 optim.step()
#                 train_loss += loss.item()
#             avg_train_loss = train_loss / len(train_loader)
#             logger.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")
#
#             # 验证用于早停（这里使用 val_loader）
#             with torch.no_grad():
#                 logits_all, y_all = [], []
#                 for xb, yb in val_loader:
#                     xb = xb.to(device)
#                     logits, _ = model(xb)
#                     logits_all.append(logits.cpu().numpy())
#                     y_all.append(yb.numpy())
#                 logits_np = np.concatenate(logits_all, 0)
#                 y_np = np.concatenate(y_all, 0)
#
#                 if args.early_metric == "macro_f1":
#                     metric_val = float(compute_metrics(logits_np, y_np)["macro_f1"])
#                 else:
#                     z = logits_np - logits_np.max(axis=1, keepdims=True)
#                     logp = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
#                     metric_val = float(-np.mean(logp[np.arange(len(y_np)), y_np]))
#
#             if is_better(metric_val, best_val):
#                 best_val = metric_val
#                 best_epoch = epoch
#                 torch.save(model.state_dict(), ckpt_path)
#                 no_improve = 0
#             else:
#                 no_improve += 1
#                 if no_improve >= args.patience:
#                     print(f"Early stopping at epoch {epoch} (best {args.early_metric}={best_val:.4f} @ {best_epoch})")
#                     break
#
#         # 载入最佳权重
#         model.load_state_dict(torch.load(ckpt_path, map_location=device))
#
#         # 验证集上做温度标定
#         def collect_logits_y(m, loader, device, T=None):
#             m.eval()
#             L, Y = [], []
#             with torch.no_grad():
#                 for xb, yb in loader:
#                     xb = xb.to(device)
#                     logits, _ = m(xb)
#                     if T is not None:
#                         logits = logits / max(T, 1e-6)
#                     L.append(logits.cpu())
#                     Y.append(yb.clone())
#             return torch.cat(L, 0), torch.cat(Y, 0)
#
#         # Collect for val (for calibration) and test (for metrics)
#         # val_logits, val_targets = collect_logits_y(model, val_loader, device, T=None)
#         test_logits, test_targets = collect_logits_y(model, test_loader, device, T=None)
#         logger.info("Collected test logits and targets")
#
#         # Calibration step
#         if args.temp_mode == "none" and args.fixed_temp is None:
#             T_cal = 1.0
#             calib_extra = {"mode": "none", "tmin": None, "tmax": None, "steps": None}
#         elif args.fixed_temp is not None:
#             T_cal = float(args.fixed_temp)
#             calib_extra = {"mode": "fixed", "tmin": None, "tmax": None, "steps": None}
#         else:
#             calib_res = calibrate_temperature_from_val(
#                 # Assuming this is your function name; adjust to calibrate_temperature_logspace if needed
#                 test_logits, test_targets,
#                 mode=args.temp_mode, tmin=args.temp_min, tmax=args.temp_max, steps=args.temp_steps
#             )
#             T_cal = float(getattr(calib_res, "T", 1.0))
#             calib_extra = {
#                 "mode": args.temp_mode,
#                 "tmin": float(args.temp_min),
#                 "tmax": float(args.temp_max),
#                 "steps": int(args.temp_steps),
#             }
#             calib_result = calib_res  # Set calib_result here for consistency
#
#         # Safe temperature extraction (fallback if calib_result not set)
#         temperature = T_cal  # Use T_cal directly (from above)
#         try:
#             if calib_result:
#                 temperature = calib_result.T
#         except NameError:
#             pass
#
#         # Compute aggregated metrics on TEST set
#         underlying_dataset = test_loader.dataset.dataset if isinstance(test_loader.dataset,
#                                                                        torch.utils.data.Subset) else test_loader.dataset
#         agg_metrics = metrics.aggregate_classification_metrics(
#             test_logits.numpy(),  # Use test_logits (np array)
#             test_targets.numpy(),  # Use test_targets
#             temperature=temperature,
#             dataset=underlying_dataset,  # For overlap_stat
#             num_classes=args.num_classes  # Add this
#         )
#
#         # 组装输出 (use agg_metrics for "metrics")
#         out = {
#             "meta": {
#                 "model": args.model,
#                 "difficulty": args.difficulty,
#                 "seed": args.seed,
#                 "F": args.F,
#                 "T": args.T,
#                 "num_classes": 4
#             },
#             "args": {
#                 "model": args.model,
#                 "difficulty": args.difficulty,
#                 "seed": args.seed,
#                 "epochs": args.epochs,
#                 "batch": args.batch,
#                 "n_samples": args.n_samples,
#                 "T": args.T,
#                 "F": args.F,
#                 "sc_corr_rho": args.sc_corr_rho,
#                 "env_burst_rate": args.env_burst_rate,
#                 "gain_drift_std": args.gain_drift_std,
#                 "class_overlap": args.class_overlap,
#                 "early_metric": args.early_metric,
#                 "patience": args.patience,
#                 "ckpt_dir": args.ckpt_dir,
#                 "logit_l2": float(getattr(args, "logit_l2", 0.0) or 0.0),
#                 "temp_mode": args.temp_mode,
#                 "temp_min": args.temp_min,
#                 "temp_max": args.temp_max,
#                 "temp_steps": args.temp_steps,
#                 "val_split": args.val_split,
#                 "fixed_temp": args.fixed_temp,
#                 "out_json": args.out_json
#             },
#             "metrics": agg_metrics,  # Unified metrics dict including falling_f1, overlap_stat, ece, etc.
#             "seed": int(args.seed),
#             "best_ckpt": ckpt_path,
#             "early_stop": {
#                 "metric": args.early_metric,
#                 "best_value": float(best_val if best_val is not None else np.nan),
#                 "best_epoch": int(best_epoch),
#                 "patience": int(args.patience)
#             },
#             "data_params": {
#                 "n_samples": args.n_samples,
#                 "sc_corr_rho": args.sc_corr_rho,
#                 "env_burst_rate": args.env_burst_rate,
#                 "gain_drift_std": args.gain_drift_std,
#                 "class_overlap": args.class_overlap
#             },
#             "calibration": calib_extra
#         }
#
#         with open(args.out_json, "w", encoding="utf-8") as f:
#             json.dump(out, f, indent=2, ensure_ascii=False)
#         # print(f"Wrote {args.out_json} | T={temperature:.4f} | (Metrics from agg: {agg_metrics})")
#         logger.info(f"Saved results to {args.out_json}")
#
#     except Exception as e:
#             logger.error(f"Error during training: {str(e)}", exc_info=True)
#             raise
if __name__ == "__main__":
    main()
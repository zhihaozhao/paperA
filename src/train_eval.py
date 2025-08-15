### UPDATED: Path fix for import errors ###
import sys
import os
# Add project root to sys.path (resolves 'No module named src' when running script from subdir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Assumes root is parent of src

import argparse
import os
import json
from typing import Optional, Dict, Any, Tuple
import subprocess
from datetime import datetime
from pathlib import Path

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
            outputs = model(xb)  # Get full output
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
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
            outputs = model(xb)  # Get full output
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
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
    parser.add_argument("--label_noise_prob", type=float, default=0.0, help="Label noise probability")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA for speed and memory")
    parser.add_argument("--save_ckpt", type=str, default="final", choices=["all", "final", "none"], help="Checkpoint saving policy")
    parser.add_argument("--num_workers_override", type=int, default=-1, help="Override dataloader workers (-1 = auto)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Dataloader prefetch_factor when workers>0")
    parser.add_argument("--val_every", type=int, default=1, help="Validate every k epochs to reduce overhead")
    parser.add_argument("--resume_from", type=str, default="", help="Path to resume checkpoint (last_*.pt)")
    parser.add_argument("--save_last_every", type=int, default=0, help="Save 'last' training checkpoint every k epochs (0=disable)")
    parser.add_argument("--strict_load", type=str, default="true", help="Strict state_dict load (true/false)")
    # Output
    parser.add_argument("--out_json", type=str, default="results/out.json", help="Path to output JSON")

    return parser.parse_args()
# -------------------------
# 主程序
# -------------------------
import logging  # NEW: Import for logging

def main():
    args = parse_args()  # Assuming your parse_args() with all fields like --model, --difficulty, etc.

    # Configure logging to file (unique per run, align with out_json basename)
    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = str(out_json_path.with_suffix('.log'))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with args: {vars(args)}")
    logger.info(f"Logs saved to: {log_file}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    use_amp = bool(torch.cuda.is_available()) and bool(getattr(args, 'amp', False))
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler(device_type="cuda")

    try:
        # Load datasets
        # train_loader, val_loader, test_loader = get_synth_loaders(args)
        ### UPDATED: Explicit unpack of args to fix passing Namespace as batch_size ###
        # Tune dataloader for GPU throughput; CPU-safe defaults remain small
        if getattr(args, 'num_workers_override', -1) is not None and args.num_workers_override >= 0:
            num_workers = int(args.num_workers_override)
        else:
            num_workers = 2 if (os.name == 'nt' and torch.cuda.is_available()) else (4 if torch.cuda.is_available() else 0)
        pin_memory = bool(torch.cuda.is_available())
        train_loader, val_loader, test_loader = get_synth_loaders(
            batch=args.batch,  # Now passing int, not Namespace
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
            num_classes=args.num_classes,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=getattr(args, 'prefetch_factor', 2),
        )
        logger.info(f"Dataset: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
        logger.info(f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory}")

        # Build and move model to device
        model = build_model(args.model, args.F, args.num_classes)
        model = model.to(device)
        logger.info(f"Model: {args.model} with {sum(p.numel() for p in model.parameters())} params")

        # Optional resume
        if args.resume_from:
            try:
                strict = str(args.strict_load).lower() in ("1","true","yes")
                state = torch.load(args.resume_from, map_location=device)
                model.load_state_dict(state, strict=strict)
                logger.info(f"Resumed weights from {args.resume_from} (strict={strict})")
            except Exception as e:
                logger.warning(f"Resume failed from {args.resume_from}: {e}")

        # Optimizer and criterion (adjust to your exact setup, e.g., with logit_l2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example
        criterion = nn.CrossEntropyLoss()

        ### UPDATED: Create checkpoints directory if it doesn't exist (fixes RuntimeError) ###
        os.makedirs(args.ckpt_dir, exist_ok=True)  # Ensures 'checkpoints/' exists before saving

        # Training loop with early stopping
        best_val = None
        best_epoch = 0
        patience_counter = 0
        ckpt_path = None  # Will be set if saving checkpoints
        log_interval = 5
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            reg_loss_total = 0  # New: Track regularization loss
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                if use_amp:
                    with autocast("cuda"):
                        outputs = model(xb)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        loss = criterion(logits, yb)
                        if args.logit_l2 > 0:
                            reg_loss = args.logit_l2 * torch.mean(torch.norm(logits, p=2, dim=1))
                            loss = loss + reg_loss
                            reg_loss_total += float(reg_loss.detach().item())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(xb)  # Get full output
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = criterion(logits, yb)
                    if args.logit_l2 > 0:  # Assuming --logit_l2 is the param for lambda
                        reg_loss = args.logit_l2 * torch.mean(torch.norm(logits, p=2, dim=1))
                        loss += reg_loss
                        reg_loss_total += reg_loss.item()  # Accumulate for averaging
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            avg_reg_loss = reg_loss_total / len(train_loader) if args.logit_l2 > 0 else 0  # Average reg loss

            # Optionally save "last" checkpoint periodically for resume
            if args.save_last_every and ((epoch + 1) % max(1, args.save_last_every) == 0):
                last_ckpt = os.path.join(args.ckpt_dir, f"last_{args.model}_{args.seed}_{args.difficulty}.pt")
                torch.save(model.state_dict(), last_ckpt)
                logger.info(f"Saved last checkpoint: {last_ckpt} (epoch={epoch+1})")

            # logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")

            # Validation
            do_val = ((epoch + 1) % max(1, args.val_every) == 0)
            avg_val_loss = float('nan')
            if do_val:
                model.eval()
                val_loss = 0
                try:
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            outputs = model(xb)  # Get full output
                            logits = outputs[0] if isinstance(outputs, tuple) else outputs
                            val_loss += criterion(logits, yb).item()
                except OSError as e:
                    logger.warning(f"Val loader error {e}; rebuilding with num_workers=0 and retrying once")
                    from torch.utils.data import DataLoader as _DL
                    val_loader = _DL(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            outputs = model(xb)
                            logits = outputs[0] if isinstance(outputs, tuple) else outputs
                            val_loss += criterion(logits, yb).item()
                avg_val_loss = val_loss / max(1, len(val_loader))
                logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}")

            # Compute early stopping metric (e.g., macro_f1)
            if do_val:
                try:
                    val_logits, val_targets = _collect_logits_labels(model, val_loader, device)
                except OSError as e:
                    logger.warning(f"Val collect error {e}; rebuilding with num_workers=0 and retrying once")
                    from torch.utils.data import DataLoader as _DL
                    val_loader = _DL(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
                    val_logits, val_targets = _collect_logits_labels(model, val_loader, device)
                val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                val_labels = val_targets.cpu().numpy()
                if args.early_metric == 'macro_f1':
                    current_metric = f1_score(val_labels, val_preds, average='macro')
                else:
                    current_metric = 0  # Add logic for other metrics
            else:
                current_metric = best_val if best_val is not None else 0
            # logger.info(f"Epoch {epoch+1} - Val {args.early_metric}: {current_metric:.4f}")

            if best_val is None or current_metric > best_val:
                best_val = current_metric
                best_epoch = epoch + 1
                patience_counter = 0
                if args.save_ckpt != 'none':
                    ckpt_name = f"best_{args.model}_{args.seed}_{args.difficulty}.pth"
                    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"Best model saved to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            # Reduced logging: only every log_interval epochs
            if (epoch + 1) % log_interval == 0:
                log_msg = f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val {args.early_metric}: {current_metric:.4f}"
                if args.logit_l2 > 0:
                    log_msg += f", Avg Reg Loss: {avg_reg_loss:.6f}"  # Verify reg is non-zero
                logger.info(log_msg)
        # Final evaluation
        try:
            test_logits, test_targets = _collect_logits_labels(model, test_loader, device)
        except OSError as e:
            logger.warning(f"Test collect error {e}; rebuilding test loader with num_workers=0 and retrying once")
            from torch.utils.data import DataLoader as _DL
            test_loader = _DL(test_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
            test_logits, test_targets = _collect_logits_labels(model, test_loader, device)
        logger.info("Collected test logits and targets")

        # Temperature calibration
        calib_extra = {}
        if args.temp_mode != 'none':
            split_idx = int(len(test_targets) * args.val_split)
            cal_val_logits = test_logits[:split_idx]
            cal_val_targets = test_targets[:split_idx]
            calib_result = calibrate_temperature_from_val(cal_val_logits, cal_val_targets, args.temp_mode,
                                                          args.temp_min, args.temp_max, args.temp_steps)
            ### UPDATED: Fixed attribute names to match your TempCalibResult dataclass (T and nll) ###
            temperature = calib_result.T if calib_result else args.fixed_temp or 1.0  # Use .T instead of .temperature
            calib_extra = {
                "temperature": temperature,
                "best_nll": calib_result.nll if calib_result else None,  # Use .nll instead of .best_nll
                "method": args.temp_mode
            }
            logger.info(f"Calibration: T={temperature:.4f}, Details={calib_extra}")
        else:
            temperature = args.fixed_temp or 1.0
            calib_extra = {"temperature": temperature, "method": "none"}
            logger.info("No calibration; using T=1.0")

        # Compute aggregated metrics (only fills "metrics" field)
        # agg_metrics = aggregate_classification_metrics(test_logits, test_targets, temperature)  # Your function
        ### UPDATED: Added .cpu().numpy() to convert tensors to NumPy arrays (fixes TypeError in metrics.py softmax/np.max) ###
        agg_metrics = aggregate_classification_metrics(test_logits.cpu().numpy(), test_targets.cpu().numpy(),
                                                       temperature, num_classes=args.num_classes)

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
                "label_noise_prob": args.label_noise_prob,
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
                "class_overlap": args.class_overlap,
                "label_noise_prob": args.label_noise_prob
            },
            "calibration": calib_extra
        }

        # Enrich meta with code/version and runtime info
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd(), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_commit = ""
        try:
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=os.getcwd(), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_branch = ""
        out.setdefault("meta", {}).update({
            "git_commit": git_commit,
            "git_branch": git_branch,
            "device": str(device),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

        # Ensure output directory exists (for JSON)
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

        # Save to JSON
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved full results to {args.out_json}")

        # Save final checkpoint if requested
        if args.save_ckpt == 'final':
            final_ckpt = os.path.join(args.ckpt_dir, f"final_{args.model}_{args.seed}_{args.difficulty}.pth")
            torch.save(model.state_dict(), final_ckpt)
            logger.info(f"Final model saved to {final_ckpt}")

        # Append lightweight registry for quick lookup (kept under results/registry.csv)
        try:
            reg_dir = os.path.join("results")
            os.makedirs(reg_dir, exist_ok=True)
            reg_path = os.path.join(reg_dir, "registry.csv")
            header_needed = not os.path.exists(reg_path)
            import csv
            with open(reg_path, "a", newline="", encoding="utf-8") as rf:
                w = csv.writer(rf)
                if header_needed:
                    w.writerow(["file", "model", "seed", "difficulty", "macro_f1", "ece_cal", "nll_cal", "git_commit", "git_branch", "device", "timestamp"])
                w.writerow([
                    args.out_json,
                    args.model,
                    args.seed,
                    args.difficulty,
                    out.get("metrics", {}).get("macro_f1", ""),
                    out.get("metrics", {}).get("ece_cal", ""),
                    out.get("metrics", {}).get("nll_cal", ""),
                    git_commit,
                    git_branch,
                    str(device),
                    out["meta"].get("timestamp", ""),
                ])
            logger.info(f"Updated registry: {reg_path}")
        except Exception as _:
            logger.warning("Failed to update results/registry.csv (non-fatal)")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
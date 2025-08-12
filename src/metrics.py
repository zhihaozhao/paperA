# -*- coding: utf-8 -*-
"""
Metrics utilities.

This module provides:
- confusion_matrix_from_preds
- compute_macro_f1
- compute_ece (expected calibration error; simple binning)
- compute_brier
- compute_nll  # Added in your code
- compute_mutual_misclass  <-- new metric
- compute_falling_f1  # NEW: Added for D1 verification
- compute_overlap_stat  # NEW: Added for D2 overlap regression
- aggregate_classification_metrics: one-stop aggregator to compute all metrics given logits/preds/labels

If you already have similar functions elsewhere, you can:
- copy only compute_mutual_misclass and confusion_matrix_from_preds
- or replace aggregate_classification_metrics with your project-specific aggregator

Author: D3 task patch
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import f1_score  # NEW: Import for falling_f1

# Assuming SynthCSIDataset is importable (from data_synth.py); adjust path if needed
from data_synth import SynthCSIDataset  # NEW: Import for overlap_stat; change if in different file


def confusion_matrix_from_preds(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build a KxK confusion matrix C where C[i, j] = count(true=i, pred=j).
    """
    C = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        C[int(t), int(p)] += 1
    return C


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Simple macro-F1 from confusion matrix.
    """
    C = confusion_matrix_from_preds(y_true, y_pred, num_classes)
    f1s = []
    for k in range(num_classes):
        tp = C[k, k]
        fp = C[:, k].sum() - tp
        fn = C[k, :].sum() - tp
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / float(tp + fp)
        rec = tp / float(tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
            continue
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def compute_ece(
        logits: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 15,
        temperature: Optional[float] = None,
) -> float:
    """
    Expected calibration error with equal-width confidence bins on [0,1].
    Uses top-class confidence.

    If 'temperature' is provided, applies logits/temperature before softmax.
    """
    if temperature is not None:
        probs = softmax(logits / float(temperature), axis=1)
    else:
        probs = softmax(logits, axis=1)

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        acc_bin = float(np.mean(correct[mask]))
        conf_bin = float(np.mean(conf[mask]))
        ece += (np.sum(mask) / len(conf)) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_brier(
        logits: np.ndarray,
        y_true: np.ndarray,
        num_classes: int,
        temperature: Optional[float] = None,
) -> float:
    """
    Multiclass Brier score (mean squared error between one-hot and probs).
    """
    if temperature is not None:
        probs = softmax(logits / float(temperature), axis=1)
    else:
        probs = softmax(logits, axis=1)

    one_hot = np.eye(num_classes)[y_true]
    mse = np.mean((probs - one_hot) ** 2)
    return float(mse)


def compute_nll(
        logits: np.ndarray,
        y_true: np.ndarray,
        temperature: Optional[float] = None,
) -> float:
    """
    Negative log-likelihood (cross-entropy) using temperature-scaled logits if provided.
    """
    if temperature is not None:
        logits = logits / float(temperature)
    # stable log-softmax
    z = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    nll = -np.mean(log_probs[np.arange(len(y_true)), y_true])
    return float(nll)


def compute_mutual_misclass(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Mutual misclassification rate:
    For each unordered class pair {i, j}, compute
      m_ij = (C[i,j] + C[j,i]) / (count(true=i) + count(true=j))
    and return the average over all pairs.

    Returns float in [0, 1]. If num_classes < 2 or denominators are 0, returns 0.0.
    """
    if num_classes < 2:
        return 0.0
    C = confusion_matrix_from_preds(y_true, y_pred, num_classes)
    totals = C.sum(axis=1)  # samples per true class
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
    return float(np.mean(pairs))


# NEW: Added compute_falling_f1
def compute_falling_f1(y_true: np.ndarray, y_pred: np.ndarray, falling_class: int = 2) -> float:
    """
    Compute F1 for 'falling' class (assume class index=2; adjust if needed).
    Returns NaN if no samples in class.
    """
    mask = y_true == falling_class
    if not np.any(mask):
        return float('nan')
    # y_true_filt = y_true[mask]
    # y_pred_filt = y_pred[mask]
    # # return f1_score(y_true[mask], y_pred[mask], average='binary', zero_division=0)
    # return f1_score(y_true_filt, y_pred_filt, average='macro', zero_division=0)  # Changed from 'binary' to 'macro'
    falling_classes = [5, 6, 7]  # Epileptic Fall, Elderly Fall, Can't Get Up
    y_true_bin = np.isin(y_true, falling_classes).astype(int)
    y_pred_bin = np.isin(y_pred, falling_classes).astype(int)
    return f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0, pos_label=1)

# This is a complete, standalone version of the compute_overlap_stat function.
# Key features:
# - Includes necessary imports (numpy, torch, Dataset/Subset) to avoid NameError.
# - Handles Subset objects by recursively unpacking to the underlying base dataset.
# - Accesses 'class_overlap' from the base dataset (assumes it's an attribute like in SynthCSIDataset).
# - Computes a placeholder statistic (mean overlap per class) – replace with your actual computation logic if different.
# - Fallback: If no 'class_overlap' attribute, returns 0.0 with a warning.
# - You can copy this directly into your metrics.py, replacing the existing def compute_overlap_stat.
# - Usage: Call as overlap_stat = compute_overlap_stat(dataset, num_classes=num_classes)

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def compute_overlap_stat(dataset: Dataset, num_classes: int) -> float:
    """
    Compute overlap statistic from dataset parameters.
    Handles Subset by unpacking to underlying dataset.

    Args:
        dataset (Dataset): The dataset (or Subset) to compute overlap from.
        num_classes (int): Number of classes in the dataset.

    Returns:
        float: The computed overlap statistic.
    """
    # Unpack Subset recursively to get base dataset
    underlying = dataset
    while isinstance(underlying, Subset):
        underlying = underlying.dataset

    # Now access attributes from base dataset
    try:
        overlap = underlying.class_overlap  # Assumes this attribute exists in base dataset (e.g., SynthCSIDataset)
    except AttributeError:
        print("Warning: Base dataset has no 'class_overlap' attribute; defaulting to 0.0")
        overlap = 0.0  # Fallback value; adjust as needed

    # Compute the statistic (placeholder logic – replace with your actual computation)
    # Example: Mean overlap impact per class
    if overlap > 0:
        stat = np.mean([overlap * i for i in range(num_classes)])
    else:
        stat = 0.0

    return float(stat)


# This is an updated patch for your metrics.py (integrate into the full file).
# Key changes:
# - Added optional 'num_classes' param to aggregate_classification_metrics (defaults to logits.shape[1]).
# - Passed num_classes to compute_falling_f1 and compute_macro_f1 (if they use it).
# - Updated compute_falling_f1 to handle multiclass with average='macro' (averages F1 across all classes).
#   - If falling_f1 is binary (e.g., falling vs non-falling), use the binarize alternative (commented; set pos_label to your falling class ID, e.g., 1).
# - Retained dataset for overlap_stat and other logic.
# - Integration: In train_eval.py (~line 597), call with num_classes=args.num_classes (optional, but recommended for consistency).

import numpy as np
from sklearn.metrics import f1_score
from typing import Dict, Optional


# ... (other imports, including SynthCSIDataset if needed)

# Assume these are your existing functions; add if missing
def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_falling_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    # Multiclass version: Use 'macro' average for F1 across all classes
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Alternative: If falling_f1 is binary (e.g., only for 'falling' class vs others), binarize:
    # pos_label = 1  # Replace with your actual 'falling' class ID (e.g., 1)
    # y_true_bin = (y_true == pos_label).astype(int)
    # y_pred_bin = (y_pred == pos_label).astype(int)
    # return f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0, pos_label=1)


# ... (your other functions: compute_ece, compute_nll, compute_mutual_misclass, compute_brier, compute_overlap_stat)

def aggregate_classification_metrics(
        logits: np.ndarray,
        y_true: np.ndarray,
        temperature: Optional[float] = None,
        n_bins_ece: int = 15,
        dataset: Optional[SynthCSIDataset] = None,  # NEW: Optional param for overlap_stat
        num_classes: Optional[int] = None,  # NEW: Optional param for multiclass handling
) -> Dict[str, float]:
    """
    Compute a standard set of metrics given logits and labels.
    - Returns macro_f1, ece_raw, ece_cal (if temperature), nll_raw, nll_cal, brier, mutual_misclass, temperature
    - NEW: Adds falling_f1 and overlap_stat (if dataset provided)
    """
    if num_classes is None:
        num_classes = logits.shape[1]  # Infer if not provided

    preds_raw = np.argmax(logits, axis=1)

    macro_f1 = compute_macro_f1(y_true, preds_raw, num_classes)  # Pass num_classes
    ece_raw = compute_ece(logits, y_true, n_bins=n_bins_ece, temperature=None)
    nll_raw = compute_nll(logits, y_true, temperature=None)

    if temperature is not None:
        ece_cal = compute_ece(logits, y_true, n_bins=n_bins_ece, temperature=temperature)
        nll_cal = compute_nll(logits, y_true, temperature=temperature)
    else:
        ece_cal = ece_raw
        nll_cal = nll_raw

    # Mutual uses argmax labels; calibration won't change argmax
    mutual_misclass = compute_mutual_misclass(y_true, preds_raw, num_classes)

    brier = compute_brier(logits, y_true, num_classes, temperature=temperature)

    metrics = {
        "macro_f1": macro_f1,
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "nll_raw": nll_raw,
        "nll_cal": nll_cal,
        "brier": brier,
        "mutual_misclass": mutual_misclass,
        "temperature": float(temperature) if temperature is not None else None,
    }

    # NEW: Add falling_f1
    falling_f1 = compute_falling_f1(y_true, preds_raw, num_classes)  # Pass num_classes
    metrics["falling_f1"] = falling_f1 if not np.isnan(falling_f1) else None

    # NEW: Add overlap_stat if dataset provided
    if dataset is not None:
        overlap_stat = compute_overlap_stat(dataset, num_classes=num_classes)
        metrics["overlap_stat"] = overlap_stat if not np.isnan(overlap_stat) else None

    return metrics

# ... (rest of metrics.py, including definitions for compute_ece, compute_nll, etc.)


# # -*- coding: utf-8 -*-
# """
# Metrics utilities.
#
# This module provides:
# - confusion_matrix_from_preds
# - compute_macro_f1
# - compute_ece (expected calibration error; simple binning)
# - compute_brier
# - compute_mutual_misclass  <-- new metric
# - aggregate_classification_metrics: one-stop aggregator to compute all metrics given logits/preds/labels
#
# If you already have similar functions elsewhere, you can:
# - copy only compute_mutual_misclass and confusion_matrix_from_preds
# - or replace aggregate_classification_metrics with your project-specific aggregator
#
# Author: D3 task patch
# """
#
# from __future__ import annotations
# import numpy as np
# from typing import Dict, Optional, Tuple
#
#
# def confusion_matrix_from_preds(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
#     """
#     Build a KxK confusion matrix C where C[i, j] = count(true=i, pred=j).
#     """
#     C = np.zeros((num_classes, num_classes), dtype=np.int64)
#     for t, p in zip(y_true, y_pred):
#         C[int(t), int(p)] += 1
#     return C
#
#
# def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
#     """
#     Simple macro-F1 from confusion matrix.
#     """
#     C = confusion_matrix_from_preds(y_true, y_pred, num_classes)
#     f1s = []
#     for k in range(num_classes):
#         tp = C[k, k]
#         fp = C[:, k].sum() - tp
#         fn = C[k, :].sum() - tp
#         if tp + fp == 0 or tp + fn == 0:
#             f1s.append(0.0)
#             continue
#         prec = tp / float(tp + fp)
#         rec = tp / float(tp + fn)
#         if prec + rec == 0:
#             f1s.append(0.0)
#             continue
#         f1s.append(2 * prec * rec / (prec + rec))
#     return float(np.mean(f1s)) if f1s else 0.0
#
#
# def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
#     x = x - np.max(x, axis=axis, keepdims=True)
#     ex = np.exp(x)
#     return ex / np.sum(ex, axis=axis, keepdims=True)
#
#
# def compute_ece(
#     logits: np.ndarray,
#     y_true: np.ndarray,
#     n_bins: int = 15,
#     temperature: Optional[float] = None,
# ) -> float:
#     """
#     Expected calibration error with equal-width confidence bins on [0,1].
#     Uses top-class confidence.
#
#     If 'temperature' is provided, applies logits/temperature before softmax.
#     """
#     if temperature is not None:
#         probs = softmax(logits / float(temperature), axis=1)
#     else:
#         probs = softmax(logits, axis=1)
#
#     conf = np.max(probs, axis=1)
#     pred = np.argmax(probs, axis=1)
#     correct = (pred == y_true).astype(np.float32)
#
#     bins = np.linspace(0.0, 1.0, n_bins + 1)
#     ece = 0.0
#     for b in range(n_bins):
#         lo, hi = bins[b], bins[b + 1]
#         mask = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
#         if not np.any(mask):
#             continue
#         acc_bin = float(np.mean(correct[mask]))
#         conf_bin = float(np.mean(conf[mask]))
#         ece += (np.sum(mask) / len(conf)) * abs(acc_bin - conf_bin)
#     return float(ece)
#
#
# def compute_brier(
#     logits: np.ndarray,
#     y_true: np.ndarray,
#     num_classes: int,
#     temperature: Optional[float] = None,
# ) -> float:
#     """
#     Multiclass Brier score (mean squared error between one-hot and probs).
#     """
#     if temperature is not None:
#         probs = softmax(logits / float(temperature), axis=1)
#     else:
#         probs = softmax(logits, axis=1)
#
#     one_hot = np.eye(num_classes)[y_true]
#     mse = np.mean((probs - one_hot) ** 2)
#     return float(mse)
#
#
# def compute_nll(
#     logits: np.ndarray,
#     y_true: np.ndarray,
#     temperature: Optional[float] = None,
# ) -> float:
#     """
#     Negative log-likelihood (cross-entropy) using temperature-scaled logits if provided.
#     """
#     if temperature is not None:
#         logits = logits / float(temperature)
#     # stable log-softmax
#     z = logits - np.max(logits, axis=1, keepdims=True)
#     log_probs = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
#     nll = -np.mean(log_probs[np.arange(len(y_true)), y_true])
#     return float(nll)
#
#
# def compute_mutual_misclass(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
#     """
#     Mutual misclassification rate:
#     For each unordered class pair {i, j}, compute
#       m_ij = (C[i,j] + C[j,i]) / (count(true=i) + count(true=j))
#     and return the average over all pairs.
#
#     Returns float in [0, 1]. If num_classes < 2 or denominators are 0, returns 0.0.
#     """
#     if num_classes < 2:
#         return 0.0
#     C = confusion_matrix_from_preds(y_true, y_pred, num_classes)
#     totals = C.sum(axis=1)  # samples per true class
#     pairs = []
#     for i in range(num_classes):
#         for j in range(i + 1, num_classes):
#             denom = float(totals[i] + totals[j])
#             if denom <= 0:
#                 continue
#             m_ij = (C[i, j] + C[j, i]) / denom
#             pairs.append(m_ij)
#     if not pairs:
#         return 0.0
#     return float(np.mean(pairs))
#
#
# def aggregate_classification_metrics(
#     logits: np.ndarray,
#     y_true: np.ndarray,
#     temperature: Optional[float] = None,
#     n_bins_ece: int = 15,
# ) -> Dict[str, float]:
#     """
#     Compute a standard set of metrics given logits and labels.
#     - Returns macro_f1, ece_raw, ece_cal (if temperature), nll_raw, nll_cal, brier, mutual_misclass, temperature
#     """
#     num_classes = logits.shape[1]
#     preds_raw = np.argmax(logits, axis=1)
#
#     macro_f1 = compute_macro_f1(y_true, preds_raw, num_classes)
#     ece_raw = compute_ece(logits, y_true, n_bins=n_bins_ece, temperature=None)
#     nll_raw = compute_nll(logits, y_true, temperature=None)
#
#     if temperature is not None:
#         ece_cal = compute_ece(logits, y_true, n_bins=n_bins_ece, temperature=temperature)
#         nll_cal = compute_nll(logits, y_true, temperature=temperature)
#     else:
#         ece_cal = ece_raw
#         nll_cal = nll_raw
#
#     # Mutual uses argmax labels; calibration won't change argmax
#     mutual_misclass = compute_mutual_misclass(y_true, preds_raw, num_classes)
#
#     brier = compute_brier(logits, y_true, num_classes, temperature=temperature)
#
#     return {
#         "macro_f1": macro_f1,
#         "ece_raw": ece_raw,
#         "ece_cal": ece_cal,
#         "nll_raw": nll_raw,
#         "nll_cal": nll_cal,
#         "brier": brier,
#         "mutual_misclass": mutual_misclass,
#         "temperature": float(temperature) if temperature is not None else None,
#     }

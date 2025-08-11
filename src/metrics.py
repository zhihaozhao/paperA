import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score

def compute_metrics(y_true, y_prob, num_classes=4, positive_class=1):
    """
    Enhanced but backward-compatible:
    - zero_division=0 to avoid warnings when a class is missing
    - robust AP when no positives/negatives (returns NaN)
    - returns f1_fall (F1 for positive_class)
    """
    y_pred = y_prob.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    y_true_bin = (y_true == positive_class).astype(int)
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        auprc = float("nan")
    else:
        auprc = average_precision_score(y_true_bin, y_prob[:, positive_class])

    f1_fall = per_class_f1[positive_class] if 0 <= positive_class < len(per_class_f1) else float("nan")
    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "f1_fall": f1_fall,
        "cm": cm,
        "auprc": auprc
    }
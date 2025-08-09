import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score

def compute_metrics(y_true, y_prob, num_classes=4, positive_class=1):
    y_pred = y_prob.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    auprc = average_precision_score((y_true==positive_class).astype(int), y_prob[:, positive_class])
    return {"macro_f1": macro_f1, "per_class_f1": per_class_f1, "cm": cm, "auprc": auprc}

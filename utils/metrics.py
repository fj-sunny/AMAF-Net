"""分类指标：Acc、召回、Macro-F1、AUC、ECE、Brier、NLL。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def confusion_and_metrics(cm: np.ndarray):
    n = cm.shape[0]
    acc = np.diag(cm).sum() / (cm.sum() + 1e-12)
    recall = np.zeros(n)
    for c in range(n):
        denom = cm[c, :].sum()
        recall[c] = (cm[c, c] / denom) if denom > 0 else 0.0
    macro_f1 = 0.0
    for c in range(n):
        prec_denom = cm[:, c].sum()
        prec = (cm[c, c] / prec_denom) if prec_denom > 0 else 0.0
        rec = recall[c]
        macro_f1 += 2 * prec * rec / (prec + rec + 1e-12)
    macro_f1 /= max(n, 1)
    balanced_acc = float(recall.mean()) if n else 0.0
    return float(acc), recall, float(macro_f1), balanced_acc


def precision_specificity_from_cm(cm: np.ndarray, n_classes: int):
    precision_per_class = np.zeros(n_classes)
    specificity_per_class = np.zeros(n_classes)
    for c in range(n_classes):
        pred_pos = cm[:, c].sum()
        precision_per_class[c] = (cm[c, c] / pred_pos) if pred_pos > 0 else 0.0
        tn = 0.0
        fp = 0.0
        for i in range(n_classes):
            for j in range(n_classes):
                if i != c and j != c:
                    tn += cm[i, j]
                elif i != c and j == c:
                    fp += cm[i, j]
        neg = tn + fp
        specificity_per_class[c] = (tn / neg) if neg > 0 else 0.0
    return (
        precision_per_class.tolist(),
        float(np.mean(precision_per_class)),
        specificity_per_class.tolist(),
        float(np.mean(specificity_per_class)),
    )


def format_metrics_line(m: dict, num_classes: int, idx2class: Dict[int, str], prefix: str = "") -> str:
    parts = [
        f"ACC={m['acc']:.4f}",
        f"AUC={m['auc_macro']:.4f}",
        f"MacroF1={m['macro_f1']:.4f}",
        f"BalancedAcc(Recall)={m['balanced_acc']:.4f}",
    ]
    rec = m.get("recall_per_class")
    if rec is not None and len(rec) == num_classes:
        parts.append("Recall[" + ",".join(f"{idx2class[c]}={rec[c]:.3f}" for c in range(num_classes)) + "]")
    return prefix + " | ".join(parts)


def multiclass_ovr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, np.ndarray]:
    n_classes = y_prob.shape[1]
    aucs = []
    for c in range(n_classes):
        try:
            aucs.append(roc_auc_score((y_true == c).astype(int), y_prob[:, c]))
        except Exception:
            aucs.append(np.nan)
    macro = float(np.nanmean(aucs)) if aucs else 0.0
    return macro, np.array(aucs)


def macro_average_precision_score(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    if n_classes <= 2:
        aps = []
        for c in range(n_classes):
            try:
                aps.append(average_precision_score((y_true == c).astype(int), y_prob[:, c]))
            except Exception:
                aps.append(np.nan)
        return float(np.nanmean(aps)) if aps else 0.0
    try:
        return float(average_precision_score(y_true, y_prob, average="macro"))
    except Exception:
        return 0.0


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (conf >= low) & (conf < high if i < n_bins - 1 else conf <= high)
        if in_bin.sum() == 0:
            continue
        acc_bin = (y_true[in_bin] == pred[in_bin]).mean()
        conf_bin = conf[in_bin].mean()
        ece += in_bin.sum() * np.abs(acc_bin - conf_bin)
    return float(ece / len(y_true))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def nll_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    clipped = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def summarize_batch_metrics(cm: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    acc, recall, macro_f1, balanced_acc = confusion_and_metrics(cm)
    auc_macro, auc_per = multiclass_ovr_auc(y_true, y_prob)
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "recall_per_class": recall.tolist(),
        "auc_macro": auc_macro,
        "auc_per_class": auc_per,
        "cm": cm,
        "all_prob": y_prob,
        "all_true": y_true,
    }


__all__ = [
    "confusion_and_metrics",
    "precision_specificity_from_cm",
    "format_metrics_line",
    "multiclass_ovr_auc",
    "macro_average_precision_score",
    "compute_ece",
    "brier_score",
    "nll_score",
    "weighted_f1",
    "summarize_batch_metrics",
]

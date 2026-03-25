from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_curves(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Returns PR curve arrays and summary AUCs for binary classification.
    """
    y_true = y_true.astype(int)
    roc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    return {"roc_auc": float(roc), "average_precision": float(ap), "precision": prec, "recall": rec, "thresholds": thr}


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple = (0, 1, 2),
    label_names: tuple = ("normal", "benign", "malicious"),
) -> Dict[str, dict]:
    """Compute precision, recall, F1 per class."""
    result = {}
    for i, name in zip(labels, label_names):
        mask_true = y_true == i
        mask_pred = y_pred == i
        tp = ((y_true == i) & (y_pred == i)).sum()
        result[name] = {
            "precision": float(precision_score(mask_true, mask_pred, zero_division=0)),
            "recall": float(recall_score(mask_true, mask_pred, zero_division=0)),
            "support": int(mask_true.sum()),
        }
        p, r = result[name]["precision"], result[name]["recall"]
        result[name]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return result


def malicious_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Malicious recall, FPR on benign, FPR on normal (relevant for IDS)."""
    mal = 2
    ben = 1
    nor = 0
    mal_recall = recall_score(y_true == mal, y_pred == mal, zero_division=0)
    fpr_ben = 1 - recall_score(y_true == ben, y_pred != mal, zero_division=0)
    fpr_nor = 1 - recall_score(y_true == nor, y_pred != mal, zero_division=0)
    return {
        "malicious_recall": float(mal_recall),
        "fpr_on_benign": float(fpr_ben),
        "fpr_on_normal": float(fpr_nor),
    }


def per_attack_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    attack_families: list,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """Per-attack-family detection recall (y_pred==malicious) on true-malicious windows."""
    result = {}
    families = sorted(set(f for f in attack_families if f))
    af = np.asarray(attack_families)
    for fam in families:
        mal_fam = (af == fam) & (y_true == 2)
        if mal_fam.sum() == 0:
            continue
        ys = y_score[mal_fam]
        if y_pred is not None:
            det = (y_pred[mal_fam] == 2).astype(int)
            mal_rec = float(det.mean())
        else:
            yt = np.ones(mal_fam.sum(), dtype=int)
            mal_rec = float(recall_score(yt, (ys > 0.5).astype(int), zero_division=0))
        result[fam] = {
            "support": int(mal_fam.sum()),
            "malicious_recall": mal_rec,
        }
        # ROC-AUC needs both classes in the slice; per-family malicious-only slices are often degenerate.
        if y_pred is None and len(np.unique(ys)) >= 2:
            try:
                yt_bin = np.ones(mal_fam.sum(), dtype=int)
                result[fam]["roc_auc"] = float(roc_auc_score(yt_bin, ys))
            except ValueError:
                pass
    return result


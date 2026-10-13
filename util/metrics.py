"""Detection and classification metric helpers."""

from collections.abc import Iterable
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(y_true, positive_scores) -> tuple[float, float]:
    """Compute Equal Error Rate and decision threshold."""
    fpr, tpr, thresholds = roc_curve(
        np.asarray(y_true), np.asarray(positive_scores), pos_label=1
    )
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_text_exact_match(
    target_texts: Iterable[str], prediction_texts: Iterable[str]
) -> float:
    """Fraction of predictions exactly matching targets, case-insensitive."""
    target_list = [str(item or "").strip().lower() for item in target_texts]
    pred_list = [str(item or "").strip().lower() for item in prediction_texts]
    if not target_list:
        return 0.0
    matches = [
        int(target == pred) for target, pred in zip(target_list, pred_list)
    ]
    return float(sum(matches) / len(matches))


def compute_classification_metrics(
    y_true,
    y_pred,
    score_matrix=None,
) -> dict[str, Optional[float]]:
    """Compute task metrics from label IDs and optional score vectors."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred))
        if len(y_true)
        else None,
        "precision_fake": float(
            precision_score(
                y_true, y_pred, average="binary", pos_label=1, zero_division=0
            )
        )
        if len(y_true)
        else None,
        "recall_fake": float(
            recall_score(
                y_true, y_pred, average="binary", pos_label=1, zero_division=0
            )
        )
        if len(y_true)
        else None,
        "precision_real": float(
            precision_score(
                y_true, y_pred, average="binary", pos_label=0, zero_division=0
            )
        )
        if len(y_true)
        else None,
        "recall_real": float(
            recall_score(
                y_true, y_pred, average="binary", pos_label=0, zero_division=0
            )
        )
        if len(y_true)
        else None,
        "f1": float(
            f1_score(
                y_true, y_pred, average="binary", pos_label=1, zero_division=0
            )
        )
        if len(y_true)
        else None,
        "roc_auc": None,
    }
    if score_matrix is not None and len(np.unique(y_true)) == 2:
        score_matrix = np.asarray(score_matrix)
        if score_matrix.ndim == 2 and score_matrix.shape[1] >= 2:
            try:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, score_matrix[:, 1])
                )
            except ValueError:
                metrics["roc_auc"] = None
    return metrics

"""Evaluation metrics for the CICIoT2023 classifiers.

CLAUDE.md is explicit that bare accuracy is not the headline metric here -- with
DDoS/Mirai dominating and Brute Force/Web-based as thin slices (~130x imbalance),
a model can hit 95%+ accuracy while never learning the rare classes. macro-F1
(unweighted mean across classes) and per-class recall are what this module reports;
accuracy is kept only as a secondary reference number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from src.data.label_map import CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASSES)))
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    return {
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
        "per_class_recall": {c: float(r) for c, r in zip(CLASSES, per_class_recall)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=CLASSES, zero_division=0),
    }

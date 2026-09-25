"""Shared evaluation metrics for all three CICIoT2023 training lanes.

``CLAUDE.md`` is explicit that bare accuracy is not the headline metric here:
DDoS and Mirai dominate the data while Brute Force and Web-based are thin
slices. A model can therefore look accurate while never learning the rare
classes. Every public helper in this module keeps macro-F1 and per-class recall
available; accuracy is included only as a secondary reference number.

There are two result shapes on purpose:

* :func:`compute_metrics` returns the rich, nested structure written to the
  centralized and federated JSON reports.
* :func:`summarize` returns scalar-only, flat values because Flower's
  ``MetricRecord`` cannot carry nested dictionaries.

Keeping both shapes in this one module prevents the centralized and federated
branches from quietly calculating or naming the same metric differently.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.data.label_map import CLASSES


def _as_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one-dimensional arrays and reject mismatched inputs early."""
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}"
        )
    return true, pred


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Recall for each of the eight classes, keyed by the canonical class name.

    Classes absent from ``y_true`` still appear with recall 0.0. A non-IID
    client shard may legitimately contain none of a rare class, and silently
    dropping that class would hide exactly the failure mode this metric exists
    to catch.
    """
    true, pred = _as_arrays(y_true, y_pred)
    recalls = recall_score(
        true,
        pred,
        labels=list(range(len(CLASSES))),
        average=None,
        zero_division=0,
    )
    return {name: float(value) for name, value in zip(CLASSES, recalls)}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Build the full report schema persisted by every training lane."""
    true, pred = _as_arrays(y_true, y_pred)
    labels = list(range(len(CLASSES)))
    recalls = per_class_recall(true, pred)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(
            f1_score(true, pred, average="macro", labels=labels, zero_division=0)
        ),
        "per_class_recall": recalls,
        "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
        "report": classification_report(
            true,
            pred,
            labels=labels,
            target_names=CLASSES,
            zero_division=0,
        ),
    }


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a flat scalar-only metric dictionary suitable for Flower."""
    metrics = compute_metrics(y_true, y_pred)
    flat = {
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
    }
    for name, value in metrics["per_class_recall"].items():
        flat[f"recall/{name}"] = float(value)
    return flat


def format_report(y_true: np.ndarray, y_pred: np.ndarray, digits: int = 3) -> str:
    """Return the full sklearn text report for logs and the write-up."""
    true, pred = _as_arrays(y_true, y_pred)
    return classification_report(
        true,
        pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
        digits=digits,
    )


def format_summary_line(metrics: dict[str, float]) -> str:
    """Format headline macro-F1 plus the two most fragile class recalls."""
    parts = [f"macro_f1={metrics.get('macro_f1', float('nan')):.4f}"]
    for name in ("Brute Force", "Web-based"):
        key = f"recall/{name}"
        if key in metrics:
            parts.append(f"recall[{name}]={metrics[key]:.3f}")
    if "accuracy" in metrics:
        parts.append(f"acc={metrics['accuracy']:.4f}")
    return "  ".join(parts)

"""Metrics for the 8-class task.

CLAUDE.md is explicit that bare accuracy is not a result: DDoS and Mirai
dominate the dataset while Brute Force and Web-based are thin slices, so a
model that ignores the rare classes still scores 95%+ accuracy. Every helper
here therefore returns macro-F1 and per-class recall together, and accuracy
only ever rides along as a secondary number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score

from src.data.label_map import CLASSES


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Recall for each of the 8 classes, keyed by class name.

    Classes absent from y_true still appear, with recall 0.0 — a client shard
    under non-IID partitioning may legitimately hold none of a rare class, and
    silently dropping it from the report would hide exactly the failure mode
    the metric exists to catch.
    """
    recalls = recall_score(
        y_true, y_pred, labels=list(range(len(CLASSES))), average=None, zero_division=0
    )
    return {name: float(value) for name, value in zip(CLASSES, recalls)}


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Flat metric dict: macro-F1, accuracy, and one recall entry per class.

    Flat (rather than nested) because Flower's MetricRecord only carries
    scalars, so this is what crosses the wire from client to server.
    """
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for name, value in per_class_recall(y_true, y_pred).items():
        metrics[f"recall/{name}"] = value
    return metrics


def format_report(y_true: np.ndarray, y_pred: np.ndarray, digits: int = 3) -> str:
    """Full sklearn text report, for logs and the write-up."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
        digits=digits,
    )


def format_summary_line(metrics: dict[str, float]) -> str:
    """One-line round summary: headline macro-F1 plus the rare-class recalls.

    Brute Force and Web-based are the classes most likely to collapse under
    federated averaging with skewed shards, so they are surfaced every round
    rather than only in the final report.
    """
    parts = [f"macro_f1={metrics.get('macro_f1', float('nan')):.4f}"]
    for name in ("Brute Force", "Web-based"):
        key = f"recall/{name}"
        if key in metrics:
            parts.append(f"recall[{name}]={metrics[key]:.3f}")
    if "accuracy" in metrics:
        parts.append(f"acc={metrics['accuracy']:.4f}")
    return "  ".join(parts)

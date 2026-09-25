"""Loss functions for the 8-class CICIoT2023 classification task.

Plain `nn.CrossEntropyLoss` treats every misclassification equally, which is a poor
fit here: DDoS (~206k train rows) outnumbers Brute Force (~1.5k) by ~130x
(CLAUDE.md's imbalance warning), so an unweighted CE-trained model can reach a
deceptively low loss by nailing DDoS/Mirai and never learning Brute Force/Web-based
at all -- exactly the "95% accuracy, 0% recall on rare classes" failure mode
CLAUDE.md calls out.

Two ways to correct for that, both implemented here so the training script can pick
between them via `--loss`:

- Class-weighted cross-entropy (`weighted_ce`, the default): scales each class's
  loss contribution by the inverse of its training-set frequency, so a Brute Force
  mistake counts for much more than a DDoS mistake. Simple, well-understood, and
  the standard first move for tabular class imbalance.
- Focal loss (`focal`): down-weights *easy, already-confident* examples (via the
  (1 - p_t)^gamma factor) regardless of class, letting loss concentrate on hard
  examples -- originally from dense object detection (Lin et al., 2017) where
  background examples overwhelm foreground ones, an analogous imbalance to
  DDoS/Mirai overwhelming Brute Force/Web-based here. Combined with class weights
  (`alpha`), it's a strict superset of weighted CE (gamma=0 reduces exactly to it).

Both operate on raw logits (see architectures.py) for numerical stability.

Per Lecture 2's "Afternote: KL Divergence" slide, categorical cross-entropy is
itself just KL divergence between the one-hot label distribution and the model's
softmax output (plus the label distribution's own entropy, which is 0 for a
one-hot label) -- both weighted CE and focal loss below are refinements of that
same KL-divergence-minimization objective, not a different objective entirely.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.data.label_map import CLASSES


def class_weights_from_counts(
    counts: dict[str, int], classes: list[str]
) -> torch.Tensor:
    """Inverse-frequency weights, normalized to mean 1 across classes."""
    freqs = torch.tensor([counts[c] for c in classes], dtype=torch.float32)
    if torch.any(freqs <= 0):
        missing = [c for c, frequency in zip(classes, freqs) if frequency <= 0]
        raise ValueError(f"cannot weight classes with zero rows: {missing}")
    weights = 1.0 / freqs
    weights = weights * (len(classes) / weights.sum())
    return weights


def sqrt_class_weights_from_counts(
    counts: dict[str, int],
    classes: list[str],
) -> torch.Tensor:
    """Softer inverse-frequency weighting using 1 / sqrt(class frequency)."""
    freqs = torch.tensor(
        [counts[c] for c in classes],
        dtype=torch.float32,
    )
    if torch.any(freqs <= 0):
        missing = [c for c, frequency in zip(classes, freqs) if frequency <= 0]
        raise ValueError(f"cannot weight classes with zero rows: {missing}")

    weights = 1.0 / torch.sqrt(freqs)
    weights = weights * (len(classes) / weights.sum())

    return weights


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017), operating on raw logits."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(log_p, target, weight=self.alpha, reduction="none")
        p_t = log_p.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        focal_term = (1 - p_t) ** self.gamma
        return (focal_term * ce).mean()


def build_criterion(
    loss_name: str,
    class_counts: dict[str, int],
) -> torch.nn.Module:
    if loss_name == "ce":
        return torch.nn.CrossEntropyLoss()

    if loss_name == "weighted_ce":
        weights = class_weights_from_counts(
            class_counts,
            CLASSES,
        )
        return torch.nn.CrossEntropyLoss(weight=weights)

    if loss_name == "sqrt_weighted_ce":
        weights = sqrt_class_weights_from_counts(
            class_counts,
            CLASSES,
        )
        return torch.nn.CrossEntropyLoss(weight=weights)

    if loss_name == "focal":
        weights = class_weights_from_counts(
            class_counts,
            CLASSES,
        )
        return FocalLoss(
            alpha=weights,
            gamma=2.0,
        )

    raise ValueError(f"unknown loss: {loss_name}")

"""FedLC adaptation: client-specific logit calibration inside the training loss.

Reference: Zhang et al., "Federated Learning with Label Distribution Skew via
Logits Calibration", ICML 2022, https://proceedings.mlr.press/v162/zhang22p.html.
Each client trains on calibrated logits z_j - tau * n_{c,j}**(-1/4), where n_{c,j}
is client c's count of class j. Rare local classes get the largest margin, so the
model must raise their raw logits to fit them. Inference uses unadjusted logits.

Adaptation choices (not the paper's exact recipe):
- counts are each client's whole fit population, not the current minibatch;
- a class the client never sees uses count_floor instead of zero (the inverse
  root is undefined at zero; the authors' code effectively drops such classes);
- the calibrated logits feed the project's existing criterion, so FedLC can sit
  on top of the frozen sqrt-weighted CE or on ordinary CE (see plan.json).

This module needs only NumPy and PyTorch. The original FedAvg and shared-Adam
loops are reused unchanged by passing them ClientRoutedData and FedLCCriterion.
"""
import math

import numpy as np
import torch

from src.data.label_map import CLASSES


def require(condition, message):
    # Same contract as src.federated.shared_adam.require, without importing it here.
    if not condition:
        raise ValueError(message)


def validate_settings(tau, count_floor):
    require(isinstance(tau, (int, float)) and not isinstance(tau, bool)
            and math.isfinite(tau) and tau >= 0, 'tau must be finite and nonnegative')
    require(type(count_floor) is int and count_floor >= 1, 'count_floor must be a positive integer')


def fedlc_margins(counts, tau, count_floor=1):
    """Per-class margins tau * max(count, floor)**(-1/4), computed in float64."""
    validate_settings(tau, count_floor)
    counts = np.asarray(counts)
    require(counts.shape == (len(CLASSES),) and np.issubdtype(counts.dtype, np.integer)
            and bool((counts >= 0).all()), 'One nonnegative integer count per class required')
    floored = np.maximum(counts.astype(np.float64), float(count_floor))
    margins = torch.from_numpy(tau * floored**-0.25)
    require(bool(torch.isfinite(margins).all()), 'Nonfinite margins')
    return margins


def client_class_counts(fit_labels, parts):
    """Each client's fit-population class counts, in canonical CLASSES order."""
    counts = []
    for rows in parts:
        y = np.asarray(fit_labels[rows])
        require(y.ndim == 1 and len(y) > 0, 'Every client needs fit rows')
        counts.append(np.bincount(y, minlength=len(CLASSES)).astype(np.int64))
    return counts


class FedLCCriterion:
    """Wrap a criterion so each call uses the selected client's margins.

    select(client) must precede every call, and the selection is cleared after
    use. A loop that computed a loss without first drawing that client's batch
    therefore fails instead of silently using another client's margins.
    """

    def __init__(self, base, client_counts, tau, count_floor=1):
        validate_settings(tau, count_floor)
        require(len(client_counts) > 0, 'Client counts required')
        self.base, self.tau, self.count_floor = base, float(tau), count_floor
        self.margins = [fedlc_margins(c, tau, count_floor) for c in client_counts]
        self.client = None

    def select(self, client):
        require(type(client) is int and 0 <= client < len(self.margins), 'Invalid client index')
        self.client = client

    def __call__(self, logits, target):
        require(self.client is not None, 'No client selected for this loss evaluation')
        client, self.client = self.client, None
        require(logits.ndim == 2 and logits.shape[1] == len(CLASSES), 'Batch-by-class logits required')
        if self.tau == 0:
            # Exact control path: the base criterion sees the untouched tensor.
            return self.base(logits, target)
        margins = self.margins[client].to(dtype=logits.dtype, device=logits.device)
        return self.base(logits - margins, target)


class ClientRoutedData:
    """Read-only view that tells the criterion which client owns each batch.

    The original loops draw every training batch from exactly one client's rows
    and call data.batch(rows) immediately before computing that batch's loss.
    This view identifies the owner and selects its margins. It changes no rows,
    order or sampling, so the sample chain is identical to the unwrapped data.
    """

    def __init__(self, data, criterion):
        self.data, self.criterion = data, criterion
        self.parts, self.arrays, self.manifest = data.parts, data.arrays, data.manifest
        owner = np.full(len(data.arrays['fit'][1]), -1, dtype=np.int64)
        for client, rows in enumerate(data.parts):
            require(bool((owner[rows] == -1).all()), 'Overlapping client ownership')
            owner[rows] = client
        require(bool((owner >= 0).all()), 'Client ownership must cover every fit row')
        self.owner = owner

    def batch(self, rows):
        clients = np.unique(self.owner[np.asarray(rows)])
        require(len(clients) == 1, 'A training batch must come from exactly one client')
        self.criterion.select(int(clients[0]))
        return self.data.batch(rows)

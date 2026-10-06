"""Shared training loop for the centralized-heavy and centralized-light MLP lanes.

Optimizer: Adam (Kingma & Ba, 2015), which Lecture 2's "Optimizers" sequence builds
up in three steps -- Momentum (an exponential moving average of gradients, to
smooth out oscillations that would otherwise force a smaller learning rate),
RMSprop (an "approximate second-order method": also normalize by an EMA of the
*squared* gradients, i.e. per-parameter adaptive step size), and a bias-correction
term on both EMAs (since starting the average from t=0 biases early estimates
toward zero). Adam is that combination. Its per-parameter adaptive learning rates
are a good default for a feature set this heterogeneous even after standardization
(see dataset.py), and it needs no manual LR schedule tuning to get a reasonable
first result -- appropriate given a semester-timeline project has little room for
hyperparameter search. (PyTorch's `torch.optim.Adam` implements all of the above
internally; nothing here hand-rolls it.)

Regularization: Lecture 2's "Weight Regularisation" slide contrasts L2 (smooth
shrinkage of all weights) against L1 (drives many weights to exactly zero /
sparsity, geometrically because the L1 ball's corners are where it tends to meet
the loss contours). Both are available here -- `weight_decay` below is Adam's
built-in L2 penalty; `l1_lambda` adds an explicit L1 penalty term. L2 is the
default regularizer for this project: all 46 flow-statistic features are already a
hand-engineered, domain-meaningful set (CLAUDE.md), so there's no motivation to
prune features toward zero the way L1's sparsity would -- smooth shrinkage that
keeps every feature contributing a little is the better fit. `l1_lambda` is exposed
mainly so the report can show the trade-off was considered, not because this
project expects to train with it on by default.

Early stopping and checkpointing are keyed on *validation macro-F1*, not validation
loss or accuracy -- per CLAUDE.md, a model that minimizes loss by ignoring rare
classes is exactly the failure case this project is designed to catch, and macro-F1
(unlike loss or accuracy) would surface it.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.label_map import CLASSES


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        l1_lambda: float = 0.0,
        patience: int = 5,
        device: str | None = None,
        proximal_mu: float = 0.0,
        proximal_reference: dict[str, torch.Tensor] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.l1_lambda = l1_lambda
        self.patience = patience
        if not np.isfinite(proximal_mu) or proximal_mu < 0:
            raise ValueError("proximal_mu must be finite and nonnegative")
        self.proximal_mu = proximal_mu
        self.proximal_reference = {}
        if proximal_mu > 0:
            parameters = {k: p for k, p in self.model.named_parameters() if p.requires_grad}
            if proximal_reference is None or parameters.keys() != proximal_reference.keys():
                raise ValueError("FedProx requires a reference for every trainable parameter")
            for name, parameter in parameters.items():
                anchor = proximal_reference[name]
                if anchor.shape != parameter.shape or not torch.isfinite(anchor).all():
                    raise ValueError(f"Invalid proximal reference: {name}")
                # The round-start global model is a fixed target, not a second
                # trainable model. Clone to prevent aliasing with local updates.
                self.proximal_reference[name] = anchor.detach().to(
                    device=self.device, dtype=parameter.dtype
                ).clone()
        self.last_epoch_proximal_penalty = 0.0

    def _proximal_penalty(self) -> torch.Tensor:
        """mu/2 * sum ||local - round_start_global||^2 (not a parameter mean).

        Includes biases and trainable normalization parameters; excludes buffers.
        This is separate from Adam's existing weight decay and the task loss.
        """
        return (self.proximal_mu / 2) * sum(
            (parameter - self.proximal_reference[name]).square().sum()
            for name, parameter in self.model.named_parameters() if parameter.requires_grad
        )

    def _l1_penalty(self) -> torch.Tensor:
        """L1 norm of every Linear weight matrix (`p.dim() > 1` excludes biases and
        BatchNorm's 1-D affine params, which aren't the "weights" the lecture's L1/L2
        regularization slide is about)."""
        return sum(
            p.abs().sum()
            for p in self.model.parameters()
            if p.requires_grad and p.dim() > 1
        )

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        proximal_total = 0.0
        all_preds, all_targets = [], []
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                task_loss = self.criterion(logits, y)
                if train:
                    loss = (
                        task_loss + self.l1_lambda * self._l1_penalty()
                        if self.l1_lambda > 0
                        else task_loss
                    )
                    # Skip even the arithmetic when mu=0, preserving FedAvg's
                    # original optimizer trajectory and RNG consumption exactly.
                    if self.proximal_mu > 0:
                        penalty = self._proximal_penalty()
                        loss = loss + penalty
                        proximal_total += penalty.detach().item() * X.size(0)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                # Logged/returned loss is always the task loss alone (no L1 term), so
                # train/val loss stay comparable to each other and across l1_lambda settings.
                # This legacy value averages batch-normalized task losses by row
                # count; it is NOT the unweighted validation CE in task.predict.
                # New research runs explicitly label it and share task.predict
                # for validation loss across centralized and federated lanes.
                # FedProx is also excluded here and is logged separately below.
                total_loss += task_loss.item() * X.size(0)
                all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
                all_targets.append(y.detach().cpu().numpy())
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        self.last_epoch_proximal_penalty = proximal_total / len(targets)
        macro_f1 = f1_score(
            targets,
            preds,
            labels=list(range(len(CLASSES))),
            average="macro",
            zero_division=0,
        )
        return total_loss / len(targets), macro_f1

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> list[dict]:
        best_val_f1 = -1.0
        best_state = None
        epochs_without_improvement = 0
        history: list[dict] = []

        for epoch in range(1, epochs + 1):
            train_loss, train_f1 = self._run_epoch(train_loader, train=True)
            val_loss, val_f1 = self._run_epoch(val_loader, train=False)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_macro_f1": train_f1,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                }
            )
            print(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_f1={train_f1:.4f}  "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if self.patience and epochs_without_improvement >= self.patience:
                    print(
                        f"Early stopping at epoch {epoch} (best val_macro_f1={best_val_f1:.4f})"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for X, y in loader:
                logits = self.model(X.to(self.device))
                all_logits.append(logits.cpu().numpy())
                all_targets.append(y.numpy())
        return np.concatenate(all_logits), np.concatenate(all_targets)

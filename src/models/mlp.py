"""Backward-compatible access to the canonical light MLP.

The original federated branch defined a second ``LightMLP`` here. Its layers
did not include the BatchNorm and default Dropout used by centralized-light,
so comparing their results was not an apples-to-apples centralized-versus-
federated experiment. The implementation now delegates to
``src.models.architectures.MLPClassifier``; this module remains as a friendly
import path for the teaching notebook and any teammate scripts.

Keeping one architecture matters especially for FedAvg: every client must
instantiate tensors with identical structure, and federated-light must have
the same capacity as centralized-light for the decision layer to attribute a
difference to training setting rather than model design.
"""

from __future__ import annotations

import torch
from torch import nn

from src.data.label_map import CLASSES
from src.models.architectures import LIGHT_CONFIG, MLPConfig, MLPClassifier

DEFAULT_HIDDEN_DIMS = LIGHT_CONFIG.hidden_dims


class LightMLP(MLPClassifier):
    """Canonical 46-feature light classifier with a compatibility signature."""

    def __init__(
        self,
        input_features: int,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        num_classes: int = len(CLASSES),
        dropout: float = LIGHT_CONFIG.dropout,
    ) -> None:
        config = MLPConfig(
            name=LIGHT_CONFIG.name,
            hidden_dims=tuple(hidden_dims),
            dropout=dropout,
        )
        super().__init__(input_features, num_classes, config)


def make_light_mlp(
    input_features: int,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    num_classes: int = len(CLASSES),
    dropout: float = LIGHT_CONFIG.dropout,
    seed: int | None = None,
) -> LightMLP:
    """Build the shared light architecture with optional seeded initialization.

    The server seeds the initial global model once. Clients immediately load
    those received weights, so their per-round seeds affect batch order and
    Dropout masks rather than creating different initial models.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return LightMLP(input_features, hidden_dims, num_classes, dropout)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters for the capacity side of the trade-off table."""
    params = (
        (parameter for parameter in model.parameters() if parameter.requires_grad)
        if trainable_only
        else model.parameters()
    )
    return sum(parameter.numel() for parameter in params)


def parameter_bytes(model: nn.Module) -> int:
    """Return exact tensor bytes, also one full-model client upload."""
    return sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

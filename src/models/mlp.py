"""Light MLP classifier for the 8-class CICIoT2023 task.

This is the "light" architecture used by both the centralized-light and
federated-light lanes (CLAUDE.md's three-lane comparison). Keeping it in one
place matters for the federated lane specifically: FedAvg averages parameter
tensors elementwise, so every client must instantiate a structurally
identical model or aggregation is meaningless.

Deliberately small — the point of the comparison is whether a model this size,
trained federated, is good enough to justify skipping a heavy centralized one.
Parameter counts here feed the efficiency/trade-off analysis; any latency or
power figure derived from them is *projected*, not measured (see CLAUDE.md).
"""

from __future__ import annotations

import torch
from torch import nn

from src.data.label_map import CLASSES

DEFAULT_HIDDEN_DIMS = (64, 32)


class LightMLP(nn.Module):
    """Fully-connected classifier over the 46 CICIoT2023 flow statistics.

    Args:
        input_features: number of numeric flow features (46 for this dataset).
        hidden_dims: widths of the hidden layers.
        num_classes: output classes (8 — see src/data/label_map.CLASSES).
        dropout: applied after each hidden activation; 0.0 disables it.
    """

    def __init__(
        self,
        input_features: int,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        num_classes: int = len(CLASSES),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_features
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_light_mlp(
    input_features: int,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    num_classes: int = len(CLASSES),
    dropout: float = 0.0,
    seed: int | None = None,
) -> LightMLP:
    """Build a LightMLP with optionally seeded initialization.

    Every federated run is seeded (CLAUDE.md convention). The server seeds the
    initial global model once; clients receive those weights over the wire and
    never re-initialize, so client-side seeding only affects batch shuffling.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return LightMLP(input_features, hidden_dims, num_classes, dropout)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count — input to the model-size side of the trade-off table."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_bytes(model: nn.Module) -> int:
    """Size of the parameter tensors in bytes.

    This is also the per-client upload volume of one FedAvg round, since a
    client returns exactly one full set of parameters per round.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())

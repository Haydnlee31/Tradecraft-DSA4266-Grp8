"""Single seeding entry point shared by the centralized and federated lanes."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU, plus all GPUs when present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def client_seed(seed: int, partition_id: int, server_round: int) -> int:
    """Per-client, per-round seed.

    Depends only on (seed, partition_id, server_round), so every client gets its own
    stream and the stream doesn't change with which simulation worker runs the client
    or in what order.
    """
    return seed * 100_003 + partition_id * 1_000 + server_round

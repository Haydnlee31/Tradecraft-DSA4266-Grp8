"""Do not label an all-failed federated round as a completed experiment."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
from flwr.app import ArrayRecord

from src.federated.custom_strategy import CustomFedAvg


class StrategyTests(unittest.TestCase):
    def test_missing_client_reply_fails_closed(self):
        strategy = CustomFedAvg(fraction_evaluate=0, min_available_nodes=1)
        strategy.configure_train = MagicMock(return_value=[MagicMock()])
        grid = MagicMock()
        grid.send_and_receive.return_value = []
        with tempfile.TemporaryDirectory() as directory:
            strategy.set_save_path(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "Client training failed"):
                strategy.start(grid, ArrayRecord({"weight": torch.ones(1)}), num_rounds=1)


if __name__ == "__main__":
    unittest.main()

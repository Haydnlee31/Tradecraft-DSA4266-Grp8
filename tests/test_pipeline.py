"""Exercise actual tensor training and Flower aggregation on tiny data."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from src.data.label_map import CLASSES
from src.federated import task
from src.federated.client_app import train_partition
from src.federated.partition import build_partition
from src.models.dataset import build_dataloaders
from src.models.architectures import MLPClassifier, LIGHT_CONFIG
from src.models.losses import build_criterion
from src.models.train import Trainer


class PipelineTests(unittest.TestCase):
    def test_training_aggregation_and_shared_scaler(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            splits = Path(directory) / "splits"
            splits.mkdir()
            rng = np.random.default_rng(123)
            for split, rows in (("train", 129), ("val", 32), ("test", 32)):
                data = {f"x{i}": rng.normal(size=rows) for i in range(46)}
                data["class"] = [CLASSES[i % 8] for i in range(rows)]
                pl.DataFrame(data).write_parquet(splits / f"{split}.parquet")
            data = build_dataloaders(batch_size=32, splits_dir=splits)
            model = MLPClassifier(46, 8, LIGHT_CONFIG)
            trainer = Trainer(
                model,
                build_criterion("sqrt_weighted_ce", data["class_counts"]),
                device="cpu",
            )
            trainer.fit(data["loaders"]["train"], data["loaders"]["val"], epochs=1)
            mapping = build_partition(
                splits / "train.parquet", Path(directory) / "parts", 0.5, 3, 0, 2
            )
            config = {
                "splits-dir": str(splits),
                "partitioner": "dirichlet",
                "partition-file": str(mapping),
                "seed": 0,
                "batch-size": 16,
                "local-epochs": 1,
                "loss": "sqrt_weighted_ce",
                "class-weights": "global",
            }
            task.configure(config)
            torch.testing.assert_close(
                task.load_server_val().dataset.tensors[0],
                data["loaders"]["val"].dataset.tensors[0],
            )
            replies = []
            for client in range(3):
                state, metrics = train_partition(
                    model.state_dict(), client, 3, config, 1, 0.001
                )
                replies.append(
                    RecordDict(
                        {"arrays": ArrayRecord(state), "metrics": MetricRecord(metrics)}
                    )
                )
            aggregated = aggregate_arrayrecords(
                replies, "num-examples"
            ).to_torch_state_dict()
            model.load_state_dict(aggregated)
            logits, labels = trainer.predict(data["loaders"]["test"])
            self.assertEqual(logits.shape, (32, 8))
            self.assertTrue(np.isfinite(logits).all())
            self.assertEqual(len(labels), 32)


if __name__ == "__main__":
    unittest.main()

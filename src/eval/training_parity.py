"""Compare the real central and client MLP loops on a fixed training subset.

No test/validation scores are used. Global class weights come from train only.
Reset and persistent-Adam cases are paired separately: conflating them would
mistake an intentional optimizer policy for an implementation discrepancy.
"""

import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import torch
from torch.utils.data import TensorDataset

from src.baselines.run import sha256
from src.data.label_map import CLASSES
from src.federated import task
from src.models.architectures import MLPClassifier, LIGHT_CONFIG
from src.models.dataset import class_counts, load_or_fit_scaler, to_arrays
from src.models.losses import build_criterion
from src.models.research import make_loader
from src.models.train import Trainer
from src.utils.seed import set_seed


def compare(dataset, counts, batch_size=512):
    torch.set_num_threads(1)
    set_seed(0)
    initial = MLPClassifier(dataset.tensors[0].shape[1], len(CLASSES), LIGHT_CONFIG)
    checks = []
    for mode in ("one_pass", "two_pass_reset", "two_pass_persistent"):
        central, federated = copy.deepcopy(initial), copy.deepcopy(initial)
        passes = 1 if mode == "one_pass" else 2
        a = Trainer(central, build_criterion("sqrt_weighted_ce", counts), device="cpu")
        if mode == "two_pass_persistent":
            # Both DataLoaders use private, equal shuffle RNGs; global RNG is
            # reset before each complete path so dropout masks also agree.
            set_seed(123)
            dl = make_loader(dataset, batch_size, 7)
            for _ in range(passes):
                a._run_epoch(dl, True)
            set_seed(123)
            with patch.object(task, "train_class_counts", return_value=counts):
                task.train(federated, make_loader(dataset, batch_size, 7), passes,
                           .001, "cpu", "sqrt_weighted_ce", "global", 1e-5)
        else:
            for epoch in range(passes):
                a = Trainer(central, build_criterion("sqrt_weighted_ce", counts), device="cpu")
                set_seed(123 + epoch)
                loss, _ = a._run_epoch(make_loader(dataset, batch_size, 7 + epoch), True)
                set_seed(123 + epoch)
                with patch.object(task, "train_class_counts", return_value=counts):
                    other_loss = task.train(federated, make_loader(dataset, batch_size, 7 + epoch),
                                             1, .001, "cpu", "sqrt_weighted_ce", "global", 1e-5)
                if abs(loss - other_loss) > 1e-6:
                    raise AssertionError("training loss differs")
        largest = 0.
        for key, value in central.state_dict().items():
            other = federated.state_dict()[key]
            torch.testing.assert_close(value, other, atol=1e-6, rtol=1e-5)
            if value.is_floating_point():
                largest = max(largest, float((value-other).abs().max()))
            elif not torch.equal(value, other):
                raise AssertionError(f"integer buffer {key} differs")
        # Include last-step gradients, not just the updated parameters/buffers.
        for left, right in zip(central.parameters(), federated.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)
        checks.append({"case": mode, "passed": True, "max_absolute_state_difference": largest})
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", type=Path, default=Path("data/splits"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scaler, features = load_or_fit_scaler(args.splits)
    frame = pl.read_parquet(args.splits / "train.parquet")
    counts = class_counts(frame)
    # Stratified fixed rows exercise every class instead of taking the first
    # 4096 rows of a potentially label-sorted sampled parquet.
    indices = []
    indexed = frame.with_row_index("_row")
    for name in CLASSES:
        indices.extend(indexed.filter(pl.col("class") == name)["_row"].head(512).to_list())
    selected = frame[indices]
    data = TensorDataset(*map(torch.from_numpy, to_arrays(selected, scaler, features)))
    result = {"checks": compare(data, counts), "rows": len(indices), "train_row_indices": indices,
              "train_sha256": sha256(args.splits / "train.parquet"),
              "device": "cpu", "batch_size": 512, "atol": 1e-6, "rtol": 1e-5,
              "scope": "matched local loops, not equivalence of 20-client FedAvg and centralized Adam"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()

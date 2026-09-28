"""Validation-only diagnostic of aggregated BatchNorm running statistics.

This deliberately uses pooled training features to recalibrate running moments
without changing learned weights. It is an oracle diagnostic, NOT a deployable
federated fix: a real server would not possess all clients' raw training rows.
No checkpoint is overwritten, no test data is read, no model is selected here.
"""

import argparse
import json
from pathlib import Path

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.eval.metrics import compute_metrics
from src.federated.task import predict
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.dataset import load_or_fit_scaler, to_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/full_run"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    scaler, features = load_or_fit_scaler(args.splits)
    datasets = {}
    for split in ("train", "val"):
        x, y = to_arrays(pl.read_parquet(args.splits / f"{split}.parquet"), scaler, features)
        datasets[split] = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    val = DataLoader(datasets["val"], batch_size=2048)
    results = {}
    for path in sorted((args.reports_dir / "checkpoints").glob("*seed0.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = MLPConfig("diagnostic", tuple(saved["config"]["hidden_dims"]), saved["config"]["dropout"])
        model = MLPClassifier(len(features), saved["num_classes"], config)
        model.load_state_dict(saved["model_state_dict"])
        _, truth, prediction = predict(model, val, "cpu")
        before = compute_metrics(truth, prediction)
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.reset_running_stats()
                module.momentum = None  # Cumulative, equal-batch running moments.
            if isinstance(module, nn.Dropout):
                module.eval()  # Recalibrate the deterministic inference network.
        loader = DataLoader(datasets["train"], batch_size=512, shuffle=True,
                            generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            for x, _ in loader:
                model(x)
        _, truth, prediction = predict(model, val, "cpu")
        after = compute_metrics(truth, prediction)
        results[path.stem] = {"before": before, "after_pooled_train_recalibration": after,
                              "validation_macro_f1_change": after["macro_f1"]-before["macro_f1"]}
        print(path.stem, before["macro_f1"], "->", after["macro_f1"], flush=True)
    (args.reports_dir / "normalization_audit.json").write_text(json.dumps({
        "scope": __doc__, "seed": 0, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()

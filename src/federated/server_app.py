"""Flower ServerApp — FedAvg aggregation plus centralized evaluation.

Port of the notebook's `fed_avg` + `run_federated_experiment`: Flower's FedAvg
strategy does the weighted parameter average (weighting by each client's
"num-examples", exactly as the notebook did), and this module supplies the
initial global model, the per-round client config, and a centralized scoring
pass over the held-out validation split.

The centralized pass is what produces the project's headline numbers. Clients
evaluate on their own non-IID shards, which is informative about site-level
behaviour but is not comparable across lanes — the val split is identical for
the centralized-heavy, centralized-light, and federated-light lanes, so only it
supports the trade-off comparison in CLAUDE.md.

`prepare_server` holds the setup shared by both execution backends (Flower's
Ray runtime via `app`, and the in-process driver in local_backend.py), so the
two cannot drift in how they seed, scale, or score.

Run configuration arrives either through the Context's run_config (under
`flwr run`) or through the TRADECRAFT_FED_CONFIG environment variable (under
run_simulation.py, whose backend has no run_config channel).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch import nn

from src.eval.metrics import format_report, format_summary_line, summarize
from src.federated.partition import SPLITS_DIR, SplitData, load_split
from src.models.mlp import count_parameters, make_light_mlp, parameter_bytes

app = ServerApp()

CONFIG_ENV_VAR = "TRADECRAFT_FED_CONFIG"

DEFAULTS: dict[str, Any] = {
    "num-rounds": 5,
    "local-epochs": 1,
    "batch-size": 64,
    "lr": 0.05,
    "momentum": 0.9,
    "alpha": 0.5,
    "seed": 7,
    "max-rows": 50_000,
    "eval-max-rows": 50_000,
    "hidden-dims": [64, 32],
    "splits-dir": str(SPLITS_DIR),
    "fraction-train": 1.0,
    "fraction-evaluate": 1.0,
    "history-path": "",
}


def settings_from_env() -> dict[str, Any]:
    """Defaults overlaid with TRADECRAFT_FED_CONFIG, if set."""
    settings = dict(DEFAULTS)
    raw = os.environ.get(CONFIG_ENV_VAR)
    if raw:
        settings.update(json.loads(raw))
    return settings


def _settings(context: Context) -> dict[str, Any]:
    """Merge run configuration: defaults < environment < run_config."""
    settings = settings_from_env()
    settings.update(dict(context.run_config))
    return settings


def hidden_dims_of(value: Any) -> tuple[int, ...]:
    # Arrives as a list via JSON, or as "64,32" via `flwr run --run-config`,
    # which only carries scalars.
    if isinstance(value, str):
        return tuple(int(part) for part in value.split(",") if part.strip())
    return tuple(int(width) for width in value)


@dataclass
class ServerSetup:
    """Everything both backends need to drive and score a federated run."""

    model: nn.Module
    val: SplitData
    input_features: int
    hidden_dims: tuple[int, ...]
    client_config: dict[str, Any]
    evaluate_fn: Callable[[int, Any], dict[str, float] | None]
    history: list[dict[str, float]] = field(default_factory=list)


def prepare_server(settings: dict[str, Any], verbose: bool = True) -> ServerSetup:
    """Seed, load the val split, build the initial model and the scoring hook."""
    seed = int(settings["seed"])
    splits_dir = Path(str(settings["splits-dir"]))
    hidden_dims = hidden_dims_of(settings["hidden-dims"])
    max_rows = int(settings["max-rows"])
    eval_max_rows = int(settings["eval-max-rows"]) or None

    torch.manual_seed(seed)

    # The standardizer must be the one fitted on the *training* rows the
    # clients use, or the server would score the global model on differently
    # scaled inputs than it was trained on.
    train_reference, standardizer = load_split(
        "train", max_rows=max_rows or None, seed=seed, splits_dir=splits_dir
    )
    val, _ = load_split(
        "val",
        max_rows=eval_max_rows,
        seed=seed,
        standardizer=standardizer,
        splits_dir=splits_dir,
    )
    input_features = train_reference.x.shape[1]

    model = make_light_mlp(input_features=input_features, hidden_dims=hidden_dims, seed=seed)
    if verbose:
        print(
            f"[server] light MLP: {input_features} features -> {hidden_dims} -> 8 classes | "
            f"{count_parameters(model)} params | "
            f"{parameter_bytes(model) / 1024:.1f} KiB uploaded per client per round"
        )
        print(f"[server] centralized val split: {len(val)} rows | {val.class_counts()}")

    history: list[dict[str, float]] = []

    def evaluate_state(server_round: int, state: dict[str, torch.Tensor]) -> dict[str, float]:
        evaluation_model = make_light_mlp(
            input_features=input_features, hidden_dims=hidden_dims
        )
        evaluation_model.load_state_dict(state)
        evaluation_model.eval()
        with torch.no_grad():
            predictions = evaluation_model(val.x).argmax(dim=1).numpy()
        metrics = summarize(val.y.numpy(), predictions)
        print(f"[server] round {server_round:>3}  centralized {format_summary_line(metrics)}")
        return metrics

    def evaluate_fn(server_round: int, arrays: Any) -> dict[str, float] | None:
        # Accepts a Flower ArrayRecord (Ray backend) or a plain state dict
        # (local backend), so both paths score identically.
        state = arrays.to_torch_state_dict() if isinstance(arrays, ArrayRecord) else arrays
        metrics = evaluate_state(server_round, state)
        history.append({"round": server_round, **metrics})
        return metrics

    client_config = {
        "seed": seed,
        "alpha": float(settings["alpha"]),
        "max-rows": max_rows,
        "splits-dir": str(splits_dir),
        "hidden-dims": list(hidden_dims),
        "batch-size": int(settings["batch-size"]),
        "local-epochs": int(settings["local-epochs"]),
        "lr": float(settings["lr"]),
        "momentum": float(settings["momentum"]),
        "server-round": 0,  # overwritten per round
    }

    return ServerSetup(
        model=model,
        val=val,
        input_features=input_features,
        hidden_dims=hidden_dims,
        client_config=client_config,
        evaluate_fn=evaluate_fn,
        history=history,
    )


def finalize(setup: ServerSetup, final_state: dict[str, torch.Tensor] | None, history_path: str) -> None:
    """Print the final per-class report and optionally persist the history."""
    if final_state is not None:
        setup.model.load_state_dict(final_state)
        setup.model.eval()
        with torch.no_grad():
            predictions = setup.model(setup.val.x).argmax(dim=1).numpy()
        print("\n[server] final global model on centralized val split:")
        print(format_report(setup.val.y.numpy(), predictions))

    if history_path:
        target = Path(history_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(setup.history, indent=2), encoding="utf-8")
        print(f"[server] per-round history -> {target}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Flower ServerApp entrypoint (Ray simulation runtime / `flwr run`)."""
    settings = _settings(context)
    setup = prepare_server(settings)

    train_config = ConfigRecord(setup.client_config)

    strategy = FedAvg(
        fraction_train=float(settings["fraction-train"]),
        fraction_evaluate=float(settings["fraction-evaluate"]),
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=1,
        weighted_by_key="num-examples",
    )

    def flwr_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        metrics = setup.evaluate_fn(server_round, arrays)
        return MetricRecord(metrics) if metrics is not None else None

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(setup.model.state_dict()),
        num_rounds=int(settings["num-rounds"]),
        train_config=train_config,
        evaluate_config=train_config,
        evaluate_fn=flwr_evaluate,
    )

    final_state = result.arrays.to_torch_state_dict() if result.arrays is not None else None
    finalize(setup, final_state, str(settings["history-path"]))

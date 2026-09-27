"""fed_ciciot: Flower ServerApp for CICIoT2023 federated learning simulation."""

import json
from datetime import datetime
from logging import INFO
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from src.data.label_map import CLASSES
from src.eval.metrics import compute_metrics
from src.federated import task
from src.federated.custom_strategy import STRATEGIES
from src.federated.task import load_model, load_server_val, load_test, test
from src.models.architectures import LIGHT_CONFIG
from src.utils.seed import set_seed

# Create ServerApp
app = ServerApp()


def build_strategy(rc):
    """The strategy named by run_config["strategy"], with its settings made explicit.

    Returns (strategy, settings dict for run_info.json).
    """
    name = str(rc["strategy"])
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name} (expected one of {sorted(STRATEGIES)})")
    kwargs = {
        "fraction_train": float(rc["fraction-train"]),
        "fraction_evaluate": float(rc["fraction-evaluate"]),
    }
    if name == "fedadagrad":
        kwargs.update(
            eta=float(rc["fedadagrad-eta"]),  # Server-side learning rate
            tau=float(rc["fedadagrad-tau"]),  # Adaptivity / numerical-stability term
            # Flower only records eta_l; clients actually train with "learning-rate"
            # (halved every 5 rounds), so record that value rather than a dummy
            eta_l=float(rc["learning-rate"]),
        )
    log(INFO, "Strategy %s with %s", name, kwargs)
    strategy = STRATEGIES[name](
        **kwargs, patience=int(rc["patience"]), use_wandb=bool(rc["use-wandb"])
    )
    return strategy, {"name": name, **kwargs}


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    task.configure(context.run_config)
    rc = context.run_config

    # Read run config
    num_rounds: int = rc["num-server-rounds"]
    lr: float = rc["learning-rate"]
    seed: int = int(rc["seed"])

    # Seed before building the model: fixes the initial weights and Flower's
    # per-round client sampling
    set_seed(seed)

    # Load global model
    global_model = load_model()
    arrays = ArrayRecord(global_model.state_dict())

    # FedAvg or FedAdagrad, per run_config["strategy"]
    strategy, strategy_settings = build_strategy(rc)

    # outputs/federated/<date>/<time>_seed{N}/ at the repo root. The seed is in the
    # name so runs with different seeds started in the same second don't collide.
    current_time = datetime.now()
    out_base = Path(rc.get("output-dir") or task.ROOT / "outputs" / "federated")
    save_path = out_base / current_time.strftime("%Y-%m-%d") / (
        current_time.strftime("%H-%M-%S") + f"_seed{seed}"
    )
    save_path.mkdir(parents=True, exist_ok=False)

    # Log what this run used, per the repo's reproducibility convention
    num_clients = len(list(grid.get_node_ids()))
    run_info = {
        "seed": seed,
        "run_config": dict(rc),
        "strategy": strategy_settings,
        "partition": {
            "partitioner": task.PARTITIONER,
            "num_clients": num_clients,
            "dirichlet_alpha": task.DIRICHLET_ALPHA if task.PARTITIONER == "dirichlet" else None,
            "partition_file": (
                task.partition_path(num_clients).name if task.PARTITIONER == "dirichlet" else None
            ),
            "train_file": task.TRAIN_PATH.name,
            "val_file": task.VAL_PATH.name,
            "splits_dir": str(task.SPLITS_DIR),
        },
    }
    (save_path / "run_info.json").write_text(json.dumps(run_info, indent=2))

    # Where checkpoints and history.json are saved
    strategy.set_output(save_path, dict(rc))

    # Run the strategy for up to `num_rounds` (fewer if early stopping triggers)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if rc["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, save_path / "final_model.pt")

    test_metrics = final_test_evaluate(save_path)
    if test_metrics is not None:
        write_report(context, num_clients, strategy, result, global_model, test_metrics, save_path)


def write_report(context, num_clients, strategy, result, model, test_metrics, save_path) -> None:
    """Write reports/federated_light_{strategy}_{loss}_{a<alpha>|iid}_n{clients}_seed{N}.json.

    Same fields as the centralized JSONs (variant, config, loss, num_parameters,
    history, test_metrics, args) so one script can tabulate all three lanes, plus
    FL-specific ones. `rounds` is the number actually run (after early stopping);
    with fraction-train = 1 and 1 local epoch, each round is ~one pass over train,
    so it is the counterpart of the centralized history's epoch count. The run
    folder's history.json has the full per-round record.
    """
    rc = context.run_config
    history = []
    for rnd in range(1, strategy.rounds_run + 1):
        train_m = dict(result.train_metrics_clientapp.get(rnd, {}))
        val_m = dict(result.evaluate_metrics_serverapp.get(rnd, {}))
        history.append(
            {
                "round": rnd,
                "train_loss": train_m.get("train_loss"),
                "val_loss": val_m.get("val_loss"),
                "val_macro_f1": val_m.get("val_macro_f1"),
                "val_accuracy": val_m.get("val_accuracy"),
                # Measured (simulation host), not an edge-device figure
                "training_time_sim_host_s": train_m.get("training_time_sim_host_s"),
            }
        )

    report = {
        "variant": "light",
        "lane": "federated",
        "config": {"hidden_dims": LIGHT_CONFIG.hidden_dims, "dropout": LIGHT_CONFIG.dropout},
        "loss": rc["loss"],
        "num_parameters": model.num_parameters(),
        "history": history,
        "test_metrics": test_metrics,
        "args": dict(rc),
        # FL-specific
        "num_clients": num_clients,
        "partitioner": task.PARTITIONER,
        "alpha": task.DIRICHLET_ALPHA if task.PARTITIONER == "dirichlet" else None,
        "strategy": rc["strategy"],
        "rounds": strategy.rounds_run,
        "num_server_rounds": int(rc["num-server-rounds"]),
        "early_stopped": strategy.early_stopped,
        "best_round": strategy.best_round,
        "local_epochs": int(rc["local-epochs"]),
        "fraction_train": float(rc["fraction-train"]),
        "class_weights": rc["class-weights"],
        "timing_note": "training_time_sim_host_s: measured (simulation host); not an edge-device figure",
        "run_dir": str(save_path),
    }

    reports_dir = Path(rc.get("reports-dir") or task.ROOT / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Strategy and partition setting in the name, so sweeps don't overwrite each other
    part = f"a{task.DIRICHLET_ALPHA}" if task.PARTITIONER == "dirichlet" else "iid"
    out_path = reports_dir / (
        f"federated_light_{rc['strategy']}_{rc['loss']}_{part}_n{num_clients}_seed{int(rc['seed'])}.json"
    )
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved report to {out_path} ({strategy.rounds_run} rounds run)")


def final_test_evaluate(save_path: Path) -> dict | None:
    """Score the val-selected best_model.pt on the test split, once, after training.

    This is the only place test.parquet is read; training and model selection only
    ever see val. Writes the full compute_metrics output (per-class recall, confusion
    matrix, classification report), matching the centralized result files.
    """
    best_path = save_path / "best_model.pt"
    if not best_path.exists():
        print("\nNo best_model.pt (no server-side evaluation ran); skipping test evaluation.")
        return None

    model = load_model()
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, y_true, y_pred = test(model, load_test(), device)
    test_metrics = compute_metrics(y_true, y_pred)
    best_info = json.loads((save_path / "best_model.json").read_text())

    out_path = save_path / "test_metrics.json"
    out_path.write_text(json.dumps({"best_model": best_info, "test_metrics": test_metrics}, indent=2))

    print(f"\nTest evaluation of best model (round {best_info['round']}) -> {out_path}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}  macro-F1: {test_metrics['macro_f1']:.4f}")
    for c, r in test_metrics["per_class_recall"].items():
        print(f"  recall[{c}]: {r:.4f}")
    return test_metrics


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the global model on the full val split -- the numbers that count.

    Returns a flat MetricRecord (values must be numbers, not dicts): val_loss,
    val_accuracy, val_macro_f1 and one val_recall_<Class> per class in CLASSES.
    """

    # Load the model and initialize it with the received weights
    model = load_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Full val split (cached after the first round)
    val_loss, y_true, y_pred = test(model, load_server_val(), device)
    m = compute_metrics(y_true, y_pred)

    metrics = {"val_loss": val_loss, "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"]}
    for c in CLASSES:
        metrics[f"val_recall_{c}"] = m["per_class_recall"][c]
    return MetricRecord(metrics)

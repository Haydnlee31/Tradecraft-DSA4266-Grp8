"""Flower ``ServerApp`` for the simulated federated-light lane.

The server initializes the exact same light model used by centralized-light,
runs sample-count-weighted FedAvg, evaluates every global model on the shared
validation split, and evaluates the validation-selected checkpoint on test
once at the end. Its JSON report intentionally follows the centralized report
schema so the decision layer can compare all three lanes without adapters.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from src.data.label_map import CLASSES
from src.eval.metrics import compute_metrics
from src.federated import task
from src.federated.custom_strategy import FEDERATED_DIR, CustomFedAvg
from src.models.architectures import LIGHT_CONFIG
from src.utils.seed import set_seed

app = ServerApp()


def _parameter_bytes(model: torch.nn.Module) -> int:
    """Exact bytes in model tensors; also one full client upload per round."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Configure and run the federated experiment."""
    task.configure(context.run_config)
    run_config = context.run_config
    num_rounds = int(run_config["num-server-rounds"])
    learning_rate = float(run_config["learning-rate"])
    seed = int(run_config["seed"])

    # Seed before model construction so initial weights and client sampling are
    # reproducible. Clients use a separate (seed, client, round) derivation.
    set_seed(seed)
    global_model = task.load_model()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = CustomFedAvg(
        fraction_train=float(run_config["fraction-train"]),
        fraction_evaluate=float(run_config["fraction-evaluate"]),
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=1,
        weighted_by_key="num-examples",
        patience=int(run_config["patience"]),
        lr_decay_every=int(run_config.get("lr-decay-every", 0)),
        lr_decay_factor=float(run_config.get("lr-decay-factor", 0.5)),
    )

    # Microseconds and seed keep repeated/swept runs from colliding.
    run_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S-%f") + f"_seed{seed}"
    output_base = Path(run_config.get("output-dir") or FEDERATED_DIR)
    save_path = output_base / "outputs" / run_name
    strategy.set_save_path(save_path)

    node_ids = list(grid.get_node_ids())
    num_clients = len(node_ids)
    if num_clients < 1:
        raise RuntimeError("Flower reported no available simulated clients")
    if task.PARTITIONER == "dirichlet":
        task.validate_partition_file(num_clients)

    # This provenance record is separate from the final metrics report so a
    # failed/interrupted run still leaves enough detail to reproduce it.
    run_info = {
        "seed": seed,
        "run_config": dict(run_config),
        "model": {
            "architecture": "MLPClassifier",
            "hidden_dims": list(LIGHT_CONFIG.hidden_dims),
            "dropout": LIGHT_CONFIG.dropout,
            "num_parameters": global_model.num_parameters(),
            "parameter_bytes": _parameter_bytes(global_model),
        },
        "partition": {
            "partitioner": task.PARTITIONER,
            "num_clients": num_clients,
            "dirichlet_alpha": (
                task.DIRICHLET_ALPHA if task.PARTITIONER == "dirichlet" else None
            ),
            "partition_file": (
                str(task.partition_path(num_clients))
                if task.PARTITIONER == "dirichlet"
                else None
            ),
            "train_file": str(task.TRAIN_PATH),
            "val_file": str(task.VAL_PATH),
        },
        # Wall-clock client timing is a simulation measurement only. Nothing
        # in this record is a measured Jetson/edge-hardware result.
        "edge_hardware_measurements": False,
    }
    (save_path / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": learning_rate}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if bool(run_config["save-model"]) and result.arrays is not None:
        torch.save(result.arrays.to_torch_state_dict(), save_path / "final_model.pt")

    test_metrics = final_test_evaluate(save_path)
    if test_metrics is not None:
        write_report(
            context=context,
            strategy=strategy,
            result=result,
            model=global_model,
            test_metrics=test_metrics,
            save_path=save_path,
            num_clients=num_clients,
        )


def write_report(
    context: Context,
    strategy: CustomFedAvg,
    result,
    model,
    test_metrics: dict,
    save_path: Path,
    num_clients: int,
) -> Path:
    """Write the same core result schema as centralized training.

    ``rounds`` is the number actually executed after early stopping. With all
    clients selected and one local epoch, one round is approximately one pass
    over the pooled training rows, which is the closest counterpart to a
    centralized epoch.
    """
    run_config = context.run_config
    history = []
    for server_round in range(0, strategy.rounds_run + 1):
        train_metrics = dict(result.train_metrics_clientapp.get(server_round, {}))
        val_metrics = dict(result.evaluate_metrics_serverapp.get(server_round, {}))
        if not train_metrics and not val_metrics:
            continue
        history.append(
            {
                "round": server_round,
                "train_loss": train_metrics.get("train_loss"),
                "val_loss": val_metrics.get("val_loss"),
                "val_macro_f1": val_metrics.get("val_macro_f1"),
                "val_accuracy": val_metrics.get("val_accuracy"),
                "val_per_class_recall": {
                    class_name: val_metrics.get(f"val_recall/{class_name}")
                    for class_name in CLASSES
                },
            }
        )

    report = {
        "variant": "light",
        "lane": "federated",
        "config": {
            "hidden_dims": list(LIGHT_CONFIG.hidden_dims),
            "dropout": LIGHT_CONFIG.dropout,
        },
        "loss": run_config["loss"],
        "num_parameters": model.num_parameters(),
        "parameter_bytes": _parameter_bytes(model),
        "history": history,
        "test_metrics": test_metrics,
        "args": dict(run_config),
        "num_clients": num_clients,
        "partitioner": task.PARTITIONER,
        "alpha": (task.DIRICHLET_ALPHA if task.PARTITIONER == "dirichlet" else None),
        "strategy": "FedAvg",
        "rounds": strategy.rounds_run,
        "num_server_rounds": int(run_config["num-server-rounds"]),
        "early_stopped": strategy.early_stopped,
        "best_round": strategy.best_round,
        "local_epochs": int(run_config["local-epochs"]),
        "fraction_train": float(run_config["fraction-train"]),
        "class_weights": run_config["class-weights"],
        "run_dir": str(save_path),
        "edge_hardware_measurements": False,
    }

    reports_dir = Path(run_config.get("reports-dir") or task.ROOT / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    partition_tag = (
        f"a{task.DIRICHLET_ALPHA}" if task.PARTITIONER == "dirichlet" else "iid"
    )
    out_path = reports_dir / (
        f"federated_light_{run_config['loss']}_{partition_tag}_"
        f"n{num_clients}_seed{int(run_config['seed'])}.json"
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {out_path} ({strategy.rounds_run} rounds run)")
    return out_path


def final_test_evaluate(save_path: Path) -> dict | None:
    """Evaluate the validation-selected checkpoint on test exactly once."""
    best_path = save_path / "best_model.pt"
    if not best_path.exists():
        print("No best_model.pt was produced; skipping final test evaluation.")
        return None

    model = task.load_model()
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, y_true, y_pred = task.predict(model, task.load_test(), device)
    test_metrics = compute_metrics(y_true, y_pred)
    best_info = json.loads((save_path / "best_model.json").read_text(encoding="utf-8"))

    out_path = save_path / "test_metrics.json"
    out_path.write_text(
        json.dumps({"best_model": best_info, "test_metrics": test_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Test evaluation of best model (round {best_info['round']}) -> {out_path}")
    print(
        f"Test accuracy: {test_metrics['accuracy']:.4f}  "
        f"macro-F1: {test_metrics['macro_f1']:.4f}"
    )
    for class_name, recall in test_metrics["per_class_recall"].items():
        print(f"  recall[{class_name}]: {recall:.4f}")
    return test_metrics


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate one global model on the full shared validation split."""
    model = task.load_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_loss, y_true, y_pred = task.predict(model, task.load_server_val(), device)
    metrics = compute_metrics(y_true, y_pred)

    flat = {
        "val_loss": float(val_loss),
        "val_accuracy": float(metrics["accuracy"]),
        "val_macro_f1": float(metrics["macro_f1"]),
    }
    for class_name, recall in metrics["per_class_recall"].items():
        flat[f"val_recall/{class_name}"] = float(recall)

    print(
        f"[server] round {server_round:>3}  val_macro_f1={flat['val_macro_f1']:.4f}  "
        f"recall[Brute Force]={flat['val_recall/Brute Force']:.3f}  "
        f"recall[Web-based]={flat['val_recall/Web-based']:.3f}"
    )
    return MetricRecord(flat)

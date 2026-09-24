"""fed_ciciot: Flower ServerApp for CICIoT2023 federated learning simulation."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdagrad
from src.federated.custom_strategy import FEDERATED_DIR, CustomFedAdagrad

from src.federated import task
from src.eval.metrics import compute_metrics
from src.federated.task import load_model, load_server_val, load_test, predict, test
from src.utils.seed import set_seed

from datetime import datetime

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    task.configure(context.run_config)

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    seed: int = int(context.run_config["seed"])

    # Seed before building the model: fixes the initial weights and Flower's
    # per-round client sampling
    set_seed(seed)

    # Load global model
    global_model = load_model()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg/FedAdagrad/CustomFedAdagrad strategy
    strategy = CustomFedAdagrad(
        fraction_evaluate=fraction_evaluate,
        patience=int(context.run_config["patience"]),
    )
    
    # Get the current date and time
    current_time = datetime.now()
    # Seed in the folder name, so runs with different seeds started in the same
    # second don't collide
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S") + f"_seed{seed}"
    # Save path is based on the current directory
    out_base = Path(context.run_config.get("output-dir") or FEDERATED_DIR)
    save_path = out_base / "outputs" / run_dir
    save_path.mkdir(parents=True, exist_ok=False)

    # Log what this run used, per the repo's reproducibility convention
    run_info = {
        "seed": seed,
        "run_config": dict(context.run_config),
        # No partition file yet: clients take every num_partitions-th row of these
        # splits (task._load_slice). Record the Dirichlet partition file here once
        # Step 4 lands.
        "partition": {
            "scheme": "row-modulo",
            "train_file": task.TRAIN_PATH.name,
            "val_file": task.VAL_PATH.name,
            "splits_dir": str(task.SPLITS_DIR),
        },
    }
    (save_path / "run_info.json").write_text(json.dumps(run_info, indent=2))

    # Set the path where results and model checkpoints will be saved
    strategy.set_save_path(save_path)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, save_path / "final_model.pt")

    final_test_evaluate(save_path)


def final_test_evaluate(save_path: Path) -> None:
    """Score the val-selected best_model.pt on the test split, once, after training.

    This is the only place test.parquet is read; training and model selection only
    ever see val.
    """
    best_path = save_path / "best_model.pt"
    if not best_path.exists():
        print("\nNo best_model.pt (no server-side evaluation ran); skipping test evaluation.")
        return

    model = load_model()
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, y_true, y_pred = predict(model, load_test(), device)
    test_metrics = compute_metrics(y_true, y_pred)
    best_info = json.loads((save_path / "best_model.json").read_text())

    out_path = save_path / "test_metrics.json"
    out_path.write_text(json.dumps({"best_model": best_info, "test_metrics": test_metrics}, indent=2))

    print(f"\nTest evaluation of best model (round {best_info['round']}) -> {out_path}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}  macro-F1: {test_metrics['macro_f1']:.4f}")
    for c, r in test_metrics["per_class_recall"].items():
        print(f"  recall[{c}]: {r:.4f}")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = load_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Full val split (cached after the first round)
    val_dataloader = load_server_val()

    # Evaluate the global model on the val set
    val_loss, val_acc, val_f1 = test(model, val_dataloader, device)

    # Return the evaluation metrics; the strategy picks the best round by val_macro_f1
    return MetricRecord({"val_macro_f1": val_f1, "val_accuracy": val_acc, "val_loss": val_loss})

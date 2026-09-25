"""Portable in-process smoke runner for the federated-light lane.

The report-producing experiment uses Flower's normal runtime through
``./run_fed.sh``. This module preserves the older branch's useful no-Ray path
for machines where Flower workers cannot launch. It uses the same client
helpers and Flower aggregation arithmetic but runs clients sequentially, so
its wall time is not a distributed-system or edge-device measurement.

Example::

    python -m src.federated.run_simulation --rounds 2 --clients 3 --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flwr.app import ArrayRecord

from src.eval.metrics import compute_metrics
from src.federated import task
from src.federated.local_backend import run_local_federation
from src.federated.partition import PARTITIONS_DIR, build_partition, partition_filename
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument(
        "--partitioner", choices=("dirichlet", "iid"), default="dirichlet"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--loss",
        choices=("ce", "weighted_ce", "sqrt_weighted_ce", "focal"),
        default="sqrt_weighted_ce",
    )
    parser.add_argument(
        "--class-weights", choices=("global", "local"), default="global"
    )
    parser.add_argument("--splits-dir", type=Path, default=task.SPLITS_DIR)
    parser.add_argument("--partitions-dir", type=Path, default=PARTITIONS_DIR)
    parser.add_argument("--history", type=Path)
    parser.add_argument(
        "--client-evaluate",
        action="store_true",
        help="Also compute site-level client validation metrics each round.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rounds < 1 or args.clients < 1:
        raise SystemExit("--rounds and --clients must both be >= 1")

    partition_file = ""
    if args.partitioner == "dirichlet":
        target = args.partitions_dir / partition_filename(
            args.alpha, args.clients, args.seed
        )
        if not target.exists():
            target = build_partition(
                train_path=args.splits_dir / "train.parquet",
                out_dir=args.partitions_dir,
                alpha=args.alpha,
                num_partitions=args.clients,
                seed=args.seed,
            )
        partition_file = str(target)

    run_config = {
        "seed": args.seed,
        "partitioner": args.partitioner,
        "dirichlet-alpha": args.alpha,
        "partition-file": partition_file,
        "splits-dir": str(args.splits_dir),
        "batch-size": args.batch_size,
        "local-epochs": args.local_epochs,
        "learning-rate": args.lr,
        "loss": args.loss,
        "class-weights": args.class_weights,
    }
    task.configure(run_config)
    set_seed(args.seed)
    model = task.load_model()

    print("=" * 78)
    print("Tradecraft — federated-light local smoke runner (simulated FedAvg)")
    print("=" * 78)
    print(json.dumps(run_config, indent=2))

    def evaluate(server_round: int, state) -> dict[str, float]:
        evaluation_model = task.load_model()
        evaluation_model.load_state_dict(state)
        loss, y_true, y_pred = task.predict(
            evaluation_model, task.load_server_val(), "cpu"
        )
        metrics = compute_metrics(y_true, y_pred)
        flat = {
            "val_loss": float(loss),
            "val_accuracy": float(metrics["accuracy"]),
            "val_macro_f1": float(metrics["macro_f1"]),
        }
        for class_name, recall in metrics["per_class_recall"].items():
            flat[f"val_recall/{class_name}"] = float(recall)
        print(
            f"[server] round {server_round:>3}  val_macro_f1={flat['val_macro_f1']:.4f}"
        )
        return flat

    _, history = run_local_federation(
        initial_state=ArrayRecord(model.state_dict()).to_torch_state_dict(),
        num_clients=args.clients,
        num_rounds=args.rounds,
        run_config=run_config,
        evaluate_fn=evaluate,
        fraction_evaluate=1.0 if args.client_evaluate else 0.0,
    )
    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        args.history.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"History -> {args.history}")


if __name__ == "__main__":
    main()

"""Entrypoint for the federated-light lane: simulated FedAvg over N clients.

Runs Flower's simulation engine in-process — multiple virtual clients on one
machine, non-IID by attack-category mix (CLAUDE.md's locked scope: federated
learning here is simulated, not physically distributed, and the edge numbers
derived from it are projected rather than measured).

Usage:
    python -m src.federated.run_simulation --rounds 5 --clients 3 --alpha 0.5

Every run prints its full configuration and the per-client class mix, so a
result can be traced back to the exact partition that produced it. Pass
--history to persist the per-round metrics as JSON for the write-up.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.federated.partition import SPLITS_DIR, describe_partitions, dirichlet_partition, load_split
from src.federated.server_app import CONFIG_ENV_VAR

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=5, help="Federated rounds.")
    parser.add_argument("--clients", type=int, default=3, help="Simulated clients (supernodes).")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration. Low (0.1) = strongly non-IID, high (100) ~ IID.",
    )
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="Training rows sampled from train.parquet before partitioning (0 = all 935k).",
    )
    parser.add_argument(
        "--eval-max-rows",
        type=int,
        default=50_000,
        help="Rows sampled from val.parquet for centralized scoring (0 = all).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="64,32",
        help="Comma-separated hidden layer widths for the light MLP.",
    )
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Optional path to write per-round metrics as JSON.",
    )
    parser.add_argument(
        "--num-cpus",
        type=float,
        default=1.0,
        help="CPU share reserved per simulated client by the Ray backend.",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "ray"),
        default="local",
        help=(
            "local (default): sequential in-process clients, Flower's FedAvg maths. "
            "ray: Flower's full simulation runtime — needs Ray, which cannot start "
            "under Windows Smart App Control (WinError 4551); use WSL2 for that."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.clients < 1:
        raise SystemExit("--clients must be >= 1")

    # Reproduce the partition the clients will build, purely so the run log
    # records which class mix each client got (CLAUDE.md: every training run
    # logs the rows it used).
    train, _ = load_split(
        "train",
        max_rows=args.max_rows or None,
        seed=args.seed,
        splits_dir=args.splits_dir,
    )
    partitions = dirichlet_partition(train.y, args.clients, args.alpha, args.seed)

    print("=" * 78)
    print("Tradecraft — federated-light lane (simulated FedAvg via Flower)")
    print("=" * 78)
    print(
        f"rounds={args.rounds}  clients={args.clients}  alpha={args.alpha}  "
        f"local_epochs={args.local_epochs}  batch_size={args.batch_size}  "
        f"lr={args.lr}  momentum={args.momentum}  seed={args.seed}"
    )
    print(f"train rows sampled: {len(train)}  |  full-split class mix: {train.class_counts()}")
    print(describe_partitions(train.y, partitions))
    print("-" * 78)

    run_config = {
        "num-rounds": args.rounds,
        "local-epochs": args.local_epochs,
        "batch-size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "alpha": args.alpha,
        "seed": args.seed,
        "max-rows": args.max_rows,
        "eval-max-rows": args.eval_max_rows,
        "hidden-dims": args.hidden_dims,
        "splits-dir": str(args.splits_dir),
        "fraction-train": 1.0,
        "fraction-evaluate": 1.0,
        "history-path": str(args.history) if args.history else "",
    }

    # The simulation backend offers no run_config channel (that exists only
    # under `flwr run`), so the ServerApp picks the settings up from the
    # environment. Set before import so any subprocess inherits it; the
    # ClientApps do not read this — they receive everything in the message's
    # ConfigRecord, which crosses the Ray worker boundary properly.
    os.environ[CONFIG_ENV_VAR] = json.dumps(run_config)

    if args.backend == "local":
        run_local(run_config, args)
    else:
        run_ray(run_config, args)


def run_local(run_config: dict, args: argparse.Namespace) -> None:
    """Sequential in-process backend — works where Ray cannot start."""
    from src.federated.local_backend import run_local_federation
    from src.federated.server_app import finalize, prepare_server

    print(f"backend: local (in-process, sequential)\n{'-' * 78}")
    setup = prepare_server(run_config)
    final_state, _ = run_local_federation(
        initial_state=setup.model.state_dict(),
        num_clients=args.clients,
        num_rounds=args.rounds,
        config=setup.client_config,
        evaluate_fn=setup.evaluate_fn,
        fraction_evaluate=float(run_config["fraction-evaluate"]),
    )
    finalize(setup, final_state, str(run_config["history-path"]))


def run_ray(run_config: dict, args: argparse.Namespace) -> None:
    """Flower's full simulation runtime (Ray-backed)."""
    from flwr.simulation import run_simulation

    # Imported here, after argument validation, so that a bad CLI invocation
    # fails before the simulation backend and Ray are initialized.
    from src.federated.client_app import app as client_app
    from src.federated.server_app import app as server_app

    print(f"backend: ray (Flower simulation runtime)\n{'-' * 78}")
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.clients,
        backend_config={"client_resources": {"num_cpus": args.num_cpus, "num_gpus": 0.0}},
    )


if __name__ == "__main__":
    main()

"""Exercise real Flower/Ray using tiny synthetic data, not study data.

This checks worker startup, message transport, client training, server
checkpointing and report production. Synthetic scores are NOT research results.
Run with ``python -m src.federated.runtime_smoke`` from the repository root.
The default Python API uses the installed environment (deprecated by Flower,
but working in the pinned version). ``--backend cli`` also tests app packaging
and the CLI's separate runtime dependency installation.
"""

import json
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from src.data.label_map import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("api", "cli"), default="api")
    parser.add_argument("--strategy", choices=("fedavg", "fedadagrad"), default="fedavg")
    parser.add_argument("--client-evaluate", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/runtime_smoke"))
    args = parser.parse_args()
    if importlib.util.find_spec("ray") is None:
        raise SystemExit("Ray workers are optional. Install them with: python -m pip install -r requirements-simulation.txt")
    # A unique directory prevents an old successful report from masking a
    # failed worker run. Synthetic results must stay out of the study reports.
    output = (args.output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tradecraft-smoke-") as directory:
        splits = Path(directory)
        rng = np.random.default_rng(123)
        for split, rows in (("train", 256), ("val", 64), ("test", 64)):
            data = {f"x{i}": rng.normal(size=rows) for i in range(46)}
            data["class"] = [CLASSES[i % len(CLASSES)] for i in range(rows)]
            pl.DataFrame(data).write_parquet(splits / f"{split}.parquet")
        command = [
            str(Path(sys.executable).with_name("flwr.exe" if os.name == "nt" else "flwr")), "run", ".", "local", "--stream",
            "--run-config",
            f"splits-dir='{splits}' output-dir='{output}' reports-dir='{output}' "
            f"partitioner='iid' num-server-rounds=1 batch-size=32 strategy='{args.strategy}' "
            f"fraction-evaluate={1.0 if args.client_evaluate else 0.0}",
            "--federation-config",
            "num-supernodes=2 init-args-num-cpus=1 client-resources-num-cpus=1",
        ]
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            # Flower launches companion executables by name, so invoking just
            # .venv/bin/python is insufficient unless PATH includes the venv.
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        }
        if args.backend == "cli":
            subprocess.run(command, check=True, env=environment, timeout=240)
        else:
            # The Python API uses the already-installed environment. Unlike
            # the 1.38 CLI it does not create/download a second environment.
            os.environ.update(environment)
            from flwr.simulation import run_simulation
            from flwr.serverapp import ServerApp
            from flwr.clientapp import ClientApp
            from src.federated.server_app import main as server_main
            from src.federated.client_app import train, evaluate

            config = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
            config.update({"splits-dir": str(splits), "output-dir": str(output),
                           "reports-dir": str(output), "partitioner": "iid",
                           "strategy": args.strategy,
                           "fraction-evaluate": 1.0 if args.client_evaluate else 0.0,
                           "num-server-rounds": 1, "batch-size": 32})
            server = ServerApp()
            client = ClientApp()

            @server.main()
            def run_server(grid, context):
                server_main(grid, replace(context, run_config=dict(config)))

            @client.train()
            def run_train(message, context):
                return train(message, replace(context, run_config=dict(config)))

            @client.evaluate()
            def run_evaluate(message, context):
                return evaluate(message, replace(context, run_config=dict(config)))

            run_simulation(server_app=server, client_app=client, num_supernodes=2,
                           backend_config={"init_args": {"num_cpus": 1},
                                           "client_resources": {"num_cpus": 1, "num_gpus": 0}})
        report = output / f"federated_light_{args.strategy}_sqrt_weighted_ce_iid_n2_seed0.json"
        result = json.loads(report.read_text())
        assert result["rounds"] == 1 and result["num_clients"] == 2
        assert "validation_metrics" in result and "test_metrics" in result
        assert any(h.get("train_loss") is not None for h in result["history"]), "no client updates aggregated"
        assert result["strategy"].lower() == args.strategy
        if args.client_evaluate:
            assert "client_macro_f1_median" in result["history"][-1]["client_eval"]
        print("PASS: Flower/Ray transport, two clients, checkpoint and report.")


if __name__ == "__main__":
    main()

"""Retrain only the frozen compressed Shared Adam result, without a new search.

This is an explicitly requested reproducibility check AFTER the study closed.
It does not amend the original no-more-tuning/confirmation execution plans.
Training starts from seeded initialization in a brand-new directory. Historical
checkpoints are comparison targets only, never initialization or resume inputs.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.data import shared_optimizer_confirmation as cohort
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import ROOT
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_block_compression as block
from src.eval import shared_optimizer_block_replication as replication
from src.eval import shared_optimizer_confirmation as confirmation
from src.eval import shared_optimizer_confirmation_prepare as prep
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.federated import block_shared_adam as codec
from src.federated.shared_adam import require
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_json

EVIDENCE = ROOT / "reports/federated_shared_optimizer/confirmation-results.json"
DATA = ROOT / "outputs/shared-optimizer-development-data-v1"
CONFIRMATION = ROOT / "outputs/shared-optimizer-confirmation-v1/data"


def published_reference():
    """Anchor comparisons to the tracked, audited public results, not local edits."""
    evidence = json.loads(EVIDENCE.read_text())
    summary, audit = evidence["summary"], evidence["audit"]
    import hashlib
    digest = hashlib.sha256(json.dumps(summary, indent=2).encode()).hexdigest()
    require(audit["status"] == summary["status"] == "complete" and
            audit["summary_sha256"] == digest, "Published summary/audit disagree")
    require(summary["contract"] == confirmation.contract(), "Frozen sources changed")
    return summary


def selected_entries(lock):
    entries = [e for e in lock["models"] if e["arm"] == "block128"]
    require([e["seed"] for e in entries] == [7, 17, 27],
            "All three seeds must be reproduced, with none selected or dropped")
    require(all(e["state_key"] == "visible" for e in entries),
            "Evaluate delivered client weights, not the hidden server target")
    return entries


def preflight(data_root, confirmation_root, project_root=ROOT):
    """Fail before spending training time if inputs, runtime or evidence differ."""
    _, lock, base, alarm, performance, parent = confirmation.read_contract(require_runtime=True)
    plan = replication.read_contract(require_runtime=True)[0]
    summary = published_reference()
    data = rep.FitData(data_root, base, alarm["data_manifest_sha256"])
    # Preserve exact source-row membership and preprocessing as well as arrays.
    for name in ("fit_row_ids.npy", "confirmation_row_ids.npy", "scaler.json"):
        require(sha256(Path(data_root) / name) == data.manifest["files"][name],
                f"Prepared provenance differs: {name}")
    x, y, calibration = replication.development(data_root, alarm, performance, parent["summary"]["split"])
    receipt_path = Path(confirmation_root) / "manifest.json"
    require(sha256(receipt_path) == summary["data_receipt_sha256"] and
            json.loads(receipt_path.read_text()) == summary["data"], "Confirmation receipt differs")
    # These are the ALREADY evaluated confirmation inputs, not a new holdout.
    cx, cy = cohort.load_reconstructed(confirmation_root, summary["data"])
    entries = selected_entries(lock)
    for entry in entries:
        checkpoint = prep.within_root(project_root, entry["checkpoint"])
        require(sha256(checkpoint) == entry["checkpoint_sha256"], "Reference checkpoint differs")
    return plan, base, alarm, parent, summary, data, x, y, calibration, cx, cy, entries


def require_fresh_output(output):
    output = Path(output)
    require(not output.exists(), "Output already exists; preserve it and choose a NEW directory")
    output.mkdir(parents=True, exist_ok=False)
    return output


def run(data_root, confirmation_root, output, project_root=ROOT):
    require(not Path(output).exists(), "Output already exists; no implicit resume or overwrite")
    started = time.perf_counter()
    plan, base, alarm, parent, published, data, x, y, calibration, cx, cy, entries = preflight(
        data_root, confirmation_root, project_root)
    output = require_fresh_output(output)
    identity = {"purpose": "from-scratch reproducibility, no selection",
                "runner_sha256": sha256(__file__),
                "published_evidence_sha256": sha256(EVIDENCE),
                "frozen_contract": confirmation.contract()}
    atomic_json(identity, output / "contract.json")
    atomic_json({"status": "running", "seeds": [7, 17, 27]}, output / "status.json")
    criterion = build_official_criterion("sqrt_weighted_ce", data.manifest["counts"]["fit"], "fixed_train_mean")
    originals = prep.endpoint_records(parent)
    results = []
    for entry in entries:
        seed = entry["seed"]
        recipe = {**base, "seed": seed}
        path = output / f"seed{seed}-block128"
        require(not path.exists(), "Fresh training directory required")
        print(f"START seed={seed}: fresh initialization, no resume", flush=True)
        require(state_digest(training.initial_model(recipe)) == entry["initial_model_sha256"],
                "Seeded initialization differs")
        # This unchanged trainer can resume in historical workflows. Here both
        # the output root and per-seed directory are new, so no checkpoint exists.
        state, seconds = block.train_candidate(
            data, recipe, plan["settings"], plan["learning_rate"], path,
            {**identity, "arm": "block128", "seed": seed}, criterion, plan["recovery_replay_steps"])
        old_path = prep.within_root(project_root, entry["checkpoint"])
        old = torch.load(old_path, map_location="cpu", weights_only=True)["state"]
        # Compare tensor values/state, not .pt bytes: timing/metadata differ.
        require(block.nested_equal(codec.snapshot(state), old),
                f"Seed {seed}: full training state differs; do not claim exact reproduction")
        block.check_work(state, recipe)
        require(state["work"] == entry["training_work"] and
                state["sample_chain_sha256"] == entry["sample_chain_sha256"] and
                state_digest(state["visible"]) == entry["model_sha256"], "Training endpoint differs")
        dev_logits = rep.infer(state["visible"], x, alarm["inference_batch_size"])
        np.save(path / "development_logits.npy", dev_logits, allow_pickle=False)
        require(sha256(path / "development_logits.npy") == entry["development_logits_sha256"],
                "Development logits differ")
        evaluated = alarms.evaluate_scores(dev_logits, y, calibration, plan["false_alert_targets_basis_points"])
        require(all(originals[seed, "block128"][k] == v for k, v in evaluated.items()),
                "Recomputed development thresholds/metrics differ")
        require([p["calibration"] for p in evaluated["operating_points"]] == entry["calibrations"],
                "Development-only threshold reproduction differs")
        logits = rep.infer(state["visible"], cx, alarm["inference_batch_size"])
        np.save(path / "confirmation_logits.npy", logits, allow_pickle=False)
        expected = next(r for r in published["results"] if (r["seed"], r["arm"]) == (seed, "block128"))
        require(sha256(path / "confirmation_logits.npy") == expected["logits_sha256"],
                "Confirmation logits differ")
        scores = confirmation.fixed_scores(logits, cy, entry["calibrations"])
        require(all(expected[k] == v for k, v in scores.items()), "Confirmation metrics differ")
        confirmation.independent_metrics(logits, cy, entry["calibrations"], scores)
        primary = next(p["confirmation_metrics"] for p in scores["operating_points"]
                       if p["calibration"]["target_basis_points"] == 500)
        result = {"seed": seed, "from_scratch": True, "full_training_state_exact": True,
                  "development_logits_exact": True, "development_thresholds_exact": True,
                  "confirmation_logits_exact": True, "confirmation_metrics_exact": True,
                  "work": state["work"], "model_sha256": state_digest(state["visible"]),
                  "checkpoint_sha256": sha256(path / "last.pt"),
                  "sample_chain_sha256": state["sample_chain_sha256"],
                  "training_seconds": seconds, "primary": primary, **scores}
        atomic_json(result, path / "result.json")
        results.append(result)
        print(f'PASS seed={seed}: macro-F1={primary["macro_f1"]:.12f}; '
              'entire training state, thresholds and predictions exact', flush=True)
    require(identity["frozen_contract"] == confirmation.contract() and
            identity["runner_sha256"] == sha256(__file__) and
            identity["published_evidence_sha256"] == sha256(EVIDENCE), "Sources changed during replay")
    values = [r["primary"]["macro_f1"] for r in results]
    total = sum(r["work"]["backward_examples"] for r in results)
    require(total == 3 * plan["backward_examples_per_run"] == 64_800_000, "Training budget differs")
    summary = {"status": "complete", "identity": identity, "results": results,
               "mean_macro_f1": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)),
               "new_training_runs": 3, "backward_examples": total,
               "confirmation_rows_per_seed": len(cy), "development_rows_per_seed": len(y),
               "confirmation_thresholds_refitted": False, "development_thresholds_reproduced": True,
               "original_validation_opened": False, "original_test_opened": False,
               "scaler_refitted": False, "cloud_used": False,
               "elapsed_seconds": time.perf_counter() - started,
               "runtime": {**replication.runtime(), "platform": platform.platform(),
                           "machine": platform.machine(), "device": "cpu", "threads": 2,
                           "git_commit": subprocess.check_output(
                               ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}}
    atomic_json(summary, output / "summary.json")
    atomic_json({"status": "complete", "all_three_seeds_exact": True}, output / "status.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--confirmation-data", type=Path, default=CONFIRMATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.action == "check":
        require(args.output is None, "Check does not create an output")
        preflight(args.data, args.confirmation_data)
        print("PASS: frozen source/runtime, fit/development/confirmation inputs and three reference checkpoints")
        print("No training performed. Ready for a fresh three-seed reproduction.")
    else:
        require(args.output is not None, "Run requires a new --output directory")
        result = run(args.data, args.confirmation_data, args.output)
        print(json.dumps({k: result[k] for k in ("status", "mean_macro_f1", "sample_sd", "backward_examples")}, indent=2))


if __name__ == "__main__":
    main()

"""Cross-platform protocol replication, NOT a promise of bitwise reproduction.

The frozen trainer, rows, partition, seeds and budgets are unchanged. Numerical
differences are reported rather than hidden. Each new model's thresholds are
calibrated on the original development calibration subset only, then saved BEFORE
confirmation inference. No threshold, model or seed is selected on confirmation.
The historical strict runner remains a separate, unchanged entry point.
"""
import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.eval import shared_optimizer_reproduce as strict
from src.eval.shared_optimizer_reproduce import (
    ROOT, DATA, CONFIRMATION, EVIDENCE, alarms, block, replication,
    confirmation, prep, training, rep, cohort, codec, state_digest,
    require, sha256, atomic_json, build_official_criterion,
)


def compatible_runtime(reference, actual):
    """Allow CPU/CUDA wheel suffixes, not arbitrary releases or nightly builds.

    A CUDA-enabled wheel still executes this experiment on CPU. Accepting its
    packaging suffix does not imply that its numerical results match macOS.
    Python and NumPy versions remain exact; the public torch release remains exact.
    """
    require(actual["python"] == reference["python"] and
            actual["numpy"] == reference["numpy"],
            "Portable mode still requires the recorded Python and NumPy versions")
    pattern = re.escape(reference["torch"]) + r"(?:\+(?:cpu|cu[0-9]+))?"
    require(re.fullmatch(pattern, actual["torch"]) is not None,
            "Portable mode requires the recorded PyTorch release (optional +cpu/+cuNNN)")


def preflight(data_root, confirmation_root, project_root=ROOT):
    """Skip ONLY historical runtime equality, retaining frozen integrity checks."""
    _, lock, base, alarm, performance, parent = confirmation.read_contract(require_runtime=False)
    compatible_runtime(lock["runtime"], replication.runtime())
    plan = replication.read_contract(require_runtime=False)[0]
    summary = strict.published_reference()
    data = rep.FitData(data_root, base, alarm["data_manifest_sha256"])
    for name in ("fit_row_ids.npy", "confirmation_row_ids.npy", "scaler.json"):
        require(sha256(Path(data_root) / name) == data.manifest["files"][name],
                f"Prepared provenance differs: {name}")
    x, y, calibration = replication.development(
        data_root, alarm, performance, parent["summary"]["split"])
    receipt = Path(confirmation_root) / "manifest.json"
    require(sha256(receipt) == summary["data_receipt_sha256"] and
            json.loads(receipt.read_text(encoding="utf-8")) == summary["data"],
            "Confirmation receipt differs")
    cx, cy = cohort.load_reconstructed(confirmation_root, summary["data"])
    entries = strict.selected_entries(lock)
    for entry in entries:
        require(sha256(prep.within_root(project_root, entry["checkpoint"])) ==
                entry["checkpoint_sha256"], "Reference checkpoint differs")
    return plan, base, alarm, parent, summary, data, x, y, calibration, cx, cy, entries


def primary(scores):
    return next(p["confirmation_metrics"] for p in scores["operating_points"]
                if p["calibration"]["target_basis_points"] == 500)


def metric_difference(current, reference):
    """Signed differences are descriptive, not a post-hoc equivalence test."""
    require(set(current["per_class"]) == set(reference["per_class"]), "Class set differs")
    return {
        "macro_f1": current["macro_f1"] - reference["macro_f1"],
        "benign_false_alert_rate": current["benign_false_alert_rate"] - reference["benign_false_alert_rate"],
        "per_class": {name: {k: scores[k] - reference["per_class"][name][k]
                             for k in ("precision", "recall", "f1", "support")}
                      for name, scores in current["per_class"].items()},
    }


def verify_work(state, recipe, entry):
    # Floating-point drift is allowed; different examples or communication work
    # are NOT a platform effect and must still stop the experiment.
    block.check_work(state, recipe)
    require(state["work"] == entry["training_work"], "Training budget differs")
    require(state["sample_chain_sha256"] == entry["sample_chain_sha256"],
            "Sampled training rows differ")


def runtime_record():
    return {**replication.runtime(), "platform": platform.platform(),
            "machine": platform.machine(), "device": "cpu", "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "torch_cuda_build": torch.version.cuda, "torch_build_config": torch.__config__.show(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


def run(data_root, confirmation_root, output, project_root=ROOT):
    require(not Path(output).exists(), "Output already exists; no implicit resume or overwrite")
    started = time.perf_counter()
    plan, base, alarm, parent, published, data, x, y, calibration, cx, cy, entries = preflight(
        data_root, confirmation_root, project_root)
    output = strict.require_fresh_output(output)
    identity = {"mode": "portable", "purpose": "cross-platform protocol replication, no selection",
                "runner_sha256": sha256(__file__), "strict_helper_sha256": sha256(strict.__file__),
                "published_evidence_sha256": sha256(EVIDENCE),
                "frozen_contract": confirmation.contract(), "runtime": runtime_record()}
    atomic_json(identity, output / "contract.json")
    atomic_json({"status": "running", "mode": "portable", "seeds": [7, 17, 27]}, output / "status.json")
    criterion = build_official_criterion("sqrt_weighted_ce", data.manifest["counts"]["fit"], "fixed_train_mean")
    originals = prep.endpoint_records(parent)
    results = []
    for entry in entries:
        seed = entry["seed"]
        recipe = {**base, "seed": seed}
        path = output / f"seed{seed}-block128"
        require(not path.exists(), "Fresh training directory required")
        print(f"START portable seed={seed}: fresh CPU training; numerical equality is not assumed", flush=True)
        initial_hash = state_digest(training.initial_model(recipe))
        state, seconds = block.train_candidate(
            data, recipe, plan["settings"], plan["learning_rate"], path,
            {**identity, "arm": "block128", "seed": seed}, criterion, plan["recovery_replay_steps"])
        verify_work(state, recipe, entry)
        old = torch.load(prep.within_root(project_root, entry["checkpoint"]),
                         map_location="cpu", weights_only=True)["state"]
        dev_logits = rep.infer(state["visible"], x, alarm["inference_batch_size"])
        np.save(path / "development_logits.npy", dev_logits, allow_pickle=False)
        evaluated = alarms.evaluate_scores(dev_logits, y, calibration, plan["false_alert_targets_basis_points"])
        replication.independent_points(dev_logits, y, calibration, evaluated["operating_points"])
        calibrations = [p["calibration"] for p in evaluated["operating_points"]]
        # Freeze the new runtime's development thresholds before confirmation.
        # Never substitute historical thresholds to obtain a preferred score.
        atomic_json(evaluated, path / "development.json")
        atomic_json(calibrations, path / "calibrations.json")
        calibration_hash = sha256(path / "calibrations.json")
        logits = rep.infer(state["visible"], cx, alarm["inference_batch_size"])
        np.save(path / "confirmation_logits.npy", logits, allow_pickle=False)
        scores = confirmation.fixed_scores(logits, cy, calibrations)
        confirmation.independent_metrics(logits, cy, calibrations, scores)
        require(sha256(path / "calibrations.json") == calibration_hash and
                json.loads((path / "calibrations.json").read_text()) == calibrations,
                "Development thresholds changed during confirmation")
        expected = next(r for r in published["results"] if (r["seed"], r["arm"]) == (seed, "block128"))
        exact = {
            "initial_model": initial_hash == entry["initial_model_sha256"],
            "full_training_state": block.nested_equal(codec.snapshot(state), old),
            "delivered_model": state_digest(state["visible"]) == entry["model_sha256"],
            "development_logits": sha256(path / "development_logits.npy") == entry["development_logits_sha256"],
            "development_metrics": all(originals[seed, "block128"][k] == v for k, v in evaluated.items()),
            "development_thresholds": calibrations == entry["calibrations"],
            "confirmation_logits": sha256(path / "confirmation_logits.npy") == expected["logits_sha256"],
            "confirmation_metrics": all(expected[k] == v for k, v in scores.items()),
        }
        result = {"seed": seed, "from_scratch": True, "exact_comparisons": exact,
                  "work": state["work"], "sample_chain_sha256": state["sample_chain_sha256"],
                  "initial_model_sha256": initial_hash, "model_sha256": state_digest(state["visible"]),
                  "checkpoint_sha256": sha256(path / "last.pt"),
                  "development_logits_sha256": sha256(path / "development_logits.npy"),
                  "confirmation_logits_sha256": sha256(path / "confirmation_logits.npy"),
                  "calibrations_sha256": calibration_hash, "training_seconds": seconds,
                  "primary": primary(scores), "reference_primary": primary(expected),
                  "primary_difference_from_reference": metric_difference(primary(scores), primary(expected)),
                  **scores}
        atomic_json(result, path / "result.json")
        results.append(result)
        print(f'COMPLETE seed={seed}: macro-F1={result["primary"]["macro_f1"]:.12f}; '
              f'difference={result["primary_difference_from_reference"]["macro_f1"]:+.12f}; '
              f'all exact checks={all(exact.values())}', flush=True)
    require(identity["frozen_contract"] == confirmation.contract() and
            identity["runner_sha256"] == sha256(__file__) and
            identity["strict_helper_sha256"] == sha256(strict.__file__) and
            identity["published_evidence_sha256"] == sha256(EVIDENCE), "Sources changed during replication")
    total = sum(r["work"]["backward_examples"] for r in results)
    require([r["seed"] for r in results] == [7, 17, 27] and
            total == 3 * plan["backward_examples_per_run"], "Seeds or training budget differ")
    values = [r["primary"]["macro_f1"] for r in results]
    reference_mean = float(np.mean([r["reference_primary"]["macro_f1"] for r in results]))
    all_exact = all(all(r["exact_comparisons"].values()) for r in results)
    summary = {"status": "complete", "mode": "portable", "identity": identity,
               "results": results, "mean_macro_f1": float(np.mean(values)),
               "sample_sd": float(np.std(values, ddof=1)), "reference_mean_macro_f1": reference_mean,
               "mean_macro_f1_difference_from_reference": float(np.mean(values)) - reference_mean,
               "all_recorded_comparisons_exact": all_exact,
               "interpretation": "Protocol completed; completion does not establish numerical equivalence or generalization.",
               "new_training_runs": 3, "backward_examples": total,
               "confirmation_rows_per_seed": len(cy), "development_rows_per_seed": len(y),
               "threshold_policy": "same development-only calibration procedure; freeze before confirmation",
               "confirmation_thresholds_refitted": False, "confirmation_model_selection": False,
               "original_validation_opened": False, "original_test_opened": False,
               "scaler_refitted": False, "cloud_used": False,
               "elapsed_seconds": time.perf_counter() - started, "runtime": identity["runtime"]}
    atomic_json(summary, output / "summary.json")
    atomic_json({"status": "complete", "mode": "portable",
                 "all_recorded_comparisons_exact": all_exact}, output / "status.json")
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
    require((args.action == "run") == (args.output is not None), "Only run requires --output")
    if args.action == "check":
        preflight(args.data, args.confirmation_data)
        print("PASS: compatible runtime, frozen sources, inputs and reference checkpoints; no training performed")
        print("PORTABLE: numerical differences will be reported, not accepted as exact reproduction")
    else:
        result = run(args.data, args.confirmation_data, args.output)
        print(json.dumps({k: result[k] for k in ("status", "mode", "mean_macro_f1", "sample_sd",
            "mean_macro_f1_difference_from_reference", "all_recorded_comparisons_exact")}, indent=2))


if __name__ == "__main__":
    main()

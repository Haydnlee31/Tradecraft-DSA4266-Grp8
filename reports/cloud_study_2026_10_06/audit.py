"""Audit trusted user-supplied cloud archives without extracting their contents.

Only JSON and weights_only PyTorch checkpoints are read; scaler pickle contents
are hashed, never executed. Archives remain private and are not copied into Git.
Run from the repository root. Output directory must be new.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from src.data.label_map import CLASSES


STAGES = {"pilot": 4, "reference": 2, "tuning": 2, "loss": 1,
          "controls": 2, "confirmation": 6, "partitions": 2}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric_check(m):
    c = np.asarray(m["confusion_matrix"], dtype=np.float64)
    require(c.shape == (8, 8) and np.isfinite(c).all() and (c >= 0).all()
            and (c == np.floor(c)).all(), "Invalid confusion counts")
    support, predicted, true = c.sum(1), c.sum(0), c.diagonal()
    recall = np.divide(true, support, out=np.zeros(8), where=support > 0)
    f1 = np.divide(2 * true, support + predicted, out=np.zeros(8), where=support + predicted > 0)
    require(np.isclose(f1.mean(), m["macro_f1"], atol=1e-12, rtol=0), "F1 mismatch")
    require(np.isclose(true.sum()/c.sum(), m["accuracy"], atol=1e-12, rtol=0), "Accuracy mismatch")
    for i, name in enumerate(CLASSES):
        require(np.isclose(recall[i], m["per_class_recall"][name], atol=1e-12, rtol=0), "Recall mismatch")
    require(np.isclose(1-recall[CLASSES.index("Benign")], m["benign_false_alert_rate"],
                       atol=1e-12, rtol=0), "False alert mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh output directory")
    frozen = json.loads(Path("reports/pre_cloud_2026_10_05/checks/summary.json").read_text())
    inventory = {item["path"]: item["sha256"] for item in frozen["transfer_inventory"]}
    summary = {"archives": {}, "runs": {}, "cuda_reexecuted_locally": False,
               "claim": "Artifact audit, not independent re-inference or deployment approval"}
    for stage, expected_count in STAGES.items():
        path = args.archive_dir / f"tradecraft-cloud-{stage}-backup.tgz"
        summary["archives"][stage] = {"filename": path.name, "bytes": path.stat().st_size,
                                         "sha256": digest(path.read_bytes())}
        with tarfile.open(path) as archive:
            names = archive.getnames()
            require(len(names) == len(set(names)), "Duplicate archive members")
            def raw(name):
                member = archive.getmember(name)
                require(member.isfile() and member.size <= 100_000_000, "Unexpected archive member")
                return archive.extractfile(member).read()
            root = f"outputs/cloud-{stage}/"
            results = [n for n in names if n.startswith(root) and n.endswith("/result.json")]
            require(len(results) == expected_count, f"Missing/extra {stage} runs")
            require(raw(root + "git-commit.txt").decode().strip().startswith("3ef7202"), "Wrong revision")
            require(raw(root + "cublas-workspace.txt").decode().strip() == ":4096:8", "Missing CUDA setting")
            for result_path in sorted(results):
                base = result_path.rsplit("/", 1)[0] + "/"
                def j(name):
                    return json.loads(raw(base + name))
                r, e, h = j("result.json"), j("environment.json"), j("history.json")
                require(e["source_sha256"] == frozen["current_source_sha256"], "Source drift")
                require(all(v == inventory[f"data/splits/{k}.parquet"]
                            for k, v in e["split_sha256"].items()), "Data drift")
                require(e["classes"] == CLASSES, "Class order drift")
                require(e["settings"]["resolved_device"] == "cuda", "Not a CUDA run")
                require(j("status.json") == {"status": "complete", "step": r["steps"]}, "Incomplete run")
                require(len(h) == r["steps"] == e["settings"]["epochs"] and h == r["history"], "History mismatch")
                require([x["step"] for x in h] == list(range(1, len(h)+1)), "Invalid steps")
                require(r["test_metrics"] is None, "Unexpected test evaluation")
                for entry in h:
                    metric_check(entry["validation_metrics"])
                metric_check(r["validation_metrics"])
                selected = max(h, key=lambda x: x["validation_metrics"]["macro_f1"])
                require(selected["step"] == r["best_step"] and
                        selected["validation_metrics"] == r["validation_metrics"], "Wrong best checkpoint")
                require(digest(raw(base + "best.pt")) == r["checkpoint_sha256"], "Checkpoint hash mismatch")
                require(digest(raw(base + "scaler.joblib")) == e["scaler_sha256"], "Scaler hash mismatch")
                if e["partition_sha256"]:
                    require(digest(raw(base + "assignments.npz")) == e["partition_sha256"], "Partition hash mismatch")
                best = torch.load(io.BytesIO(raw(base + "best.pt")), map_location="cpu", weights_only=True)
                last = torch.load(io.BytesIO(raw(base + "last.pt")), map_location="cpu", weights_only=True)
                require(last["step"] == len(h) and best["best_step"] == r["best_step"], "Wrong checkpoint step")
                for key, value in best["model_state_dict"].items():
                    torch.testing.assert_close(value, last["best_state"][key], rtol=0, atol=0)
                summary["runs"][base.rstrip("/")] = {
                    "archive_stage": stage, "manifest": e, "best_step": r["best_step"],
                    "metrics": r["validation_metrics"], "step_seconds_sum": r["step_seconds_sum"],
                    "timing_scope": r["timing_scope"], "test_metrics": None,
                    "curve": [{"step": x["step"], "macro_f1": x["validation_metrics"]["macro_f1"],
                               "recall": x["validation_metrics"]["per_class_recall"]} for x in h],
                    "checkpoint_sha256": r["checkpoint_sha256"], "checks_passed": True}
                print(f"PASS {base}", flush=True)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified {len(summary['runs'])} runs across {len(STAGES)} archives")


if __name__ == "__main__":
    main()

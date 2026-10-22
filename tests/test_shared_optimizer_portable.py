"""Portable means honest numerical reporting, not relaxed data/work integrity."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.eval import shared_optimizer_portable as portable
from src.eval import shared_optimizer_reproduce as strict
from src.eval import shared_optimizer_block_replication as replication
from src.models.research import atomic_save
from tests import test_shared_optimizer_compression as fixtures


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_runtime_policy_accepts_wheel_suffix_not_different_release(self):
        ref = {"python": "3.13.9", "torch": "2.14.0", "numpy": "2.5.3"}
        for version in ("2.14.0", "2.14.0+cpu", "2.14.0+cu130", "2.14.0+cu128"):
            portable.compatible_runtime(ref, {**ref, "torch": version})
        for version in ("2.13.0+cu130", "2.14.1", "2.14.0rc1", "2.14.0.dev1", "2.14.0+custom"):
            with self.assertRaises(ValueError):
                portable.compatible_runtime(ref, {**ref, "torch": version})
        for key in ("python", "numpy"):
            with self.assertRaises(ValueError):
                portable.compatible_runtime(ref, {**ref, key: "wrong"})

    def test_strict_guard_remains_strict_for_cuda_suffix(self):
        actual = {"python": "3.13.9", "torch": "2.14.0+cu130", "numpy": "2.5.3"}
        with patch.object(replication, "runtime", return_value=actual):
            with self.assertRaisesRegex(ValueError, "Reference runtime differs"):
                strict.preflight(strict.DATA, strict.CONFIRMATION)

    def test_frozen_source_failure_is_not_suppressed(self):
        with patch.object(portable.confirmation, "read_contract", side_effect=ValueError("Frozen sources changed")) as read:
            with self.assertRaisesRegex(ValueError, "Frozen sources"):
                portable.preflight(Path("missing"), Path("missing"))
            read.assert_called_once_with(require_runtime=False)

    def test_provenance_tampering_still_fails(self):
        base, data, _ = fixtures.SharedOptimizerCompressionTests().fixture()
        data.manifest["files"] = {"fit_row_ids.npy": "expected"}
        runtime = replication.runtime()
        with patch.object(portable.confirmation, "read_contract", return_value=(
                {}, {"runtime": runtime}, base, {"data_manifest_sha256": "digest"}, {}, {})), \
             patch.object(portable.replication, "read_contract", return_value=({},)), \
             patch.object(strict, "published_reference", return_value={}), \
             patch.object(portable.rep, "FitData", return_value=data), \
             patch.object(portable, "sha256", return_value="corrupted"):
            with self.assertRaisesRegex(ValueError, "Prepared provenance differs"):
                portable.preflight(Path("missing"), Path("missing"))

    def test_work_and_sample_drift_still_fail(self):
        base, data, criterion = fixtures.SharedOptimizerCompressionTests().fixture()
        state = portable.codec.make_state(base, 4, True, .003, 128)
        for _ in range(6):
            portable.codec.advance(state, data, base, .003, criterion)
        entry = {"training_work": copy.deepcopy(state["work"]),
                 "sample_chain_sha256": state["sample_chain_sha256"]}
        portable.verify_work(state, base, entry)
        with self.assertRaisesRegex(ValueError, "Sampled training rows"):
            portable.verify_work(state, base, {**entry, "sample_chain_sha256": "wrong"})
        bad = copy.deepcopy(entry)
        bad["training_work"]["backward_examples"] += 1
        with self.assertRaisesRegex(ValueError, "Training budget"):
            portable.verify_work(state, base, bad)
        state["work"]["estimated_payload_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "Work/payload"):
            portable.verify_work(state, base, entry)

    def test_output_preserved_before_preflight(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(portable, "preflight") as preflight:
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                portable.run(Path("missing"), Path("missing"), Path(tmp))
            preflight.assert_not_called()

    def test_nonexact_toy_replication_reports_drift_without_refitting_confirmation(self):
        """Exercise the unchanged trainer on all seeds, with distinct reference logits.

        This simulates numerical drift, NOT a claim of actual Linux execution.
        Confirmation labels must never reach the development calibrator.
        """
        base, data, criterion = fixtures.SharedOptimizerCompressionTests().fixture()
        plan = copy.deepcopy(replication.read_contract()[0])
        plan.update(backward_examples_per_run=48, recovery_replay_steps=2)
        x = np.random.default_rng(9).normal(size=(800, 39)).astype(np.float32)
        y = np.arange(800, dtype=np.int64) % 8
        calibration = np.arange(800) < 400
        cx = x[:160].copy()
        cy = y[:160].copy()
        reference_logits = np.zeros((800, 8), dtype=np.float32)
        evaluated = portable.alarms.evaluate_scores(reference_logits, y, calibration,
                                                    plan["false_alert_targets_basis_points"])
        calibrations = [p["calibration"] for p in evaluated["operating_points"]]
        expected_scores = portable.confirmation.fixed_scores(reference_logits[:160], cy, calibrations)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            published = {"results": []}
            originals = {}
            for seed in (7, 17, 27):
                recipe = {**base, "seed": seed}
                state = portable.codec.make_state(recipe, 4, True, .003, 128)
                for _ in range(6):
                    portable.codec.advance(state, data, recipe, .003, criterion)
                old = portable.codec.snapshot(state)
                # Deliberately perturb a comparison target. The new trainer must
                # still start from its seed, never load this reference to train.
                next(iter(old["visible"].values())).add_(1.)
                path = root / f"reference{seed}.pt"
                atomic_save({"state": old}, path)
                entries.append({"seed": seed, "checkpoint": path.name,
                    "checkpoint_sha256": portable.sha256(path), "training_work": copy.deepcopy(state["work"]),
                    "sample_chain_sha256": state["sample_chain_sha256"], "initial_model_sha256": "different",
                    "model_sha256": "different", "development_logits_sha256": "different",
                    "calibrations": copy.deepcopy(calibrations)})
                published["results"].append({"seed": seed, "arm": "block128", "logits_sha256": "different", **expected_scores})
                originals[seed, "block128"] = copy.deepcopy(evaluated)
            packed = (plan, base, {"inference_batch_size": 128}, {}, published,
                      data, x, y, calibration, cx, cy, entries)
            original_calibrator = portable.alarms.evaluate_scores
            original_fixed = portable.confirmation.fixed_scores
            seen = []

            def development_only(logits, labels, mask, targets):
                self.assertIs(labels, y)
                self.assertIs(mask, calibration)
                seen.append(len(labels))
                return original_calibrator(logits, labels, mask, targets)

            def fixed_after_saved(logits, labels, thresholds):
                self.assertIs(labels, cy)
                seed = (7, 17, 27)[len(seen) - 1]
                saved = root / "run" / f"seed{seed}-block128" / "calibrations.json"
                self.assertEqual(json.loads(saved.read_text()), thresholds)
                return original_fixed(logits, labels, thresholds)

            with patch.object(portable, "preflight", return_value=packed), \
                 patch.object(portable.prep, "endpoint_records", return_value=originals), \
                 patch.object(portable.alarms, "evaluate_scores", side_effect=development_only), \
                 patch.object(portable.confirmation, "fixed_scores", side_effect=fixed_after_saved), \
                 redirect_stdout(io.StringIO()):
                result = portable.run(root, root, root / "run", project_root=root)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["backward_examples"], 144)
            self.assertFalse(result["all_recorded_comparisons_exact"])
            self.assertFalse(result["confirmation_thresholds_refitted"])
            self.assertEqual(seen, [800, 800, 800])
            self.assertEqual([r["seed"] for r in result["results"]], [7, 17, 27])
            for r in result["results"]:
                self.assertFalse(r["exact_comparisons"]["full_training_state"])
                self.assertFalse(r["exact_comparisons"]["confirmation_logits"])
                self.assertEqual(len(r["primary_difference_from_reference"]["per_class"]), 8)
                self.assertAlmostEqual(r["primary_difference_from_reference"]["macro_f1"],
                                       r["primary"]["macro_f1"] - r["reference_primary"]["macro_f1"])


if __name__ == "__main__":
    unittest.main()

"""Small tests for reproduction safeguards; no real training/data access."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval import shared_optimizer_reproduce as replay


class ReproduceTests(unittest.TestCase):
    def test_distributed_input_hashes_match_public_study_evidence(self):
        config = json.loads((replay.ROOT / "reports/federated_shared_optimizer/reproduction-inputs-v1.json").read_text())
        _, lock, _, alarm, performance, _ = replay.confirmation.read_contract()
        files = config["files"]
        prefix = "outputs/shared-optimizer-development-data-v1/"
        self.assertEqual(files[prefix + "manifest.json"]["sha256"], alarm["data_manifest_sha256"])
        for name, expected in performance["summary"]["data"]["files"].items():
            self.assertEqual(files[prefix + name]["sha256"], expected)
        summary = replay.published_reference()
        prefix = "outputs/shared-optimizer-confirmation-v1/data/"
        self.assertEqual(files[prefix + "manifest.json"]["sha256"], summary["data_receipt_sha256"])
        for name, expected in summary["data"]["files"].items():
            self.assertEqual(files[prefix + name]["sha256"], expected)
        for entry in replay.selected_entries(lock):
            self.assertEqual(files[entry["checkpoint"]]["sha256"], entry["checkpoint_sha256"])

    def test_frozen_public_summary_is_self_consistent(self):
        # Static evidence is portable; this does not promise bitwise training
        # equivalence on the Linux/Windows CI runners.
        summary = replay.published_reference()
        self.assertEqual(summary["status"], "complete")
        rows = [r for r in summary["results"] if r["arm"] == "block128"]
        values = [next(p["confirmation_metrics"]["macro_f1"]
                       for p in r["operating_points"]
                       if p["calibration"]["target_basis_points"] == 500) for r in rows]
        self.assertEqual(len(values), 3)
        self.assertAlmostEqual(sum(values) / 3, 0.6325367323327683, places=14)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            replay.require_fresh_output(root)
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                replay.require_fresh_output(root)
            # Refuse before touching any training or confirmation inputs.
            with patch.object(replay, "preflight") as check:
                with self.assertRaisesRegex(ValueError, "Output already exists"):
                    replay.run("missing-data", "missing-confirmation", root)
                check.assert_not_called()

    def test_all_seeds_and_delivered_model_are_required(self):
        entries = [{"arm": "block128", "seed": s, "state_key": "visible"}
                   for s in (7, 17, 27)]
        self.assertEqual(replay.selected_entries({"models": entries}), entries)
        for invalid in (entries[:2], entries + [entries[0]],
                        [{**e, "state_key": "model"} for e in entries]):
            with self.assertRaises(ValueError):
                replay.selected_entries({"models": invalid})


if __name__ == "__main__":
    unittest.main()

"""Reject damaged experiment evidence without needing real project data."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import official39_numeric_retrieval as run


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ident = {'synthetic_fixture': True}
        y = np.arange(409918, dtype=np.int64) % 8
        pred = np.zeros(len(y), np.uint8)
        population = [100000]*8
        score = run.prior.old.prior.prior.scored(y, pred, population)
        names = ['tree', 'knn', 'ple', 'self_context_control', 'retrieval']
        files = {n+'-calibration.npy': run.save_array(self.root, n+'-calibration.npy', pred) for n in names}
        models = []
        for arm in names[-2:]:
            name = arm+'.pt'
            (self.root/name).write_bytes(b'synthetic-checksum-fixture-not-a-model')
            files[name] = run.digest(self.root/name)
            models.append(dict(arm=arm, file=name, sha256=files[name],
                               examples_processed=16219560, optimizer_steps=31680))
        recipe = dict(identity=self.ident, fit_population_counts=population)
        self.result = dict(status='complete', recipe=recipe, models=models,
            scores={n: copy.deepcopy(score) for n in names}, files=files,
            calibration_supports=np.bincount(y, minlength=8).tolist(),
            continuation_checks={'tree': False, 'self_context_control': False},
            continue_to_confirmation=False, test_opened=False, test_evaluated=False,
            audit_scored=False, independent_confirmation=False, deployment_authorized=False,
            examples_processed=32439120, optimizer_steps=63360)
        run.frozen.save(self.root/'frozen-recipe.json', recipe)
        run.frozen.save(self.root/'models-frozen.json', models)
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        run.frozen.save(self.root/'receipt.json', self.result)

    def verify(self):
        with patch.object(run, 'identity', return_value=self.ident):
            return run.verify(self.root)

    def test_valid_fixture(self):
        self.assertEqual(self.verify(), self.result)

    def test_changed_prediction_rejected(self):
        path = self.root/'retrieval-calibration.npy'
        pred = np.load(path, allow_pickle=False)
        pred[0] = 1
        np.save(path, pred, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, 'Artifact changed'):
            self.verify()

    def test_rehashed_prediction_still_rejected(self):
        path = self.root/'retrieval-calibration.npy'
        pred = np.load(path, allow_pickle=False)
        pred[0] = 1
        np.save(path, pred, allow_pickle=False)
        self.result['files'][path.name] = run.digest(path)
        self.save()
        with self.assertRaisesRegex(ValueError, 'Predictions changed'):
            self.verify()

    def test_false_promotion_rejected(self):
        self.result['continue_to_confirmation'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'Continuation decision changed'):
            self.verify()

    def test_reopened_test_rejected(self):
        self.result['test_opened'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'Invalid receipt'):
            self.verify()

    def test_incomplete_budget_rejected(self):
        self.result['examples_processed'] -= 1
        self.save()
        with self.assertRaisesRegex(ValueError, 'Invalid receipt'):
            self.verify()

    def test_only_pinned_receipt_upgrade_is_accepted(self):
        identity = {'plan_sha256': 'unchanged', 'sources': {run.RUNNER: 'current', 'core': 'same'}}
        legacy = copy.deepcopy(identity)
        legacy['sources'][run.RUNNER] = run.PRE_VERIFIER_FIX_RUNNER
        with patch.object(run, 'identity', return_value=identity):
            self.assertTrue(run.compatible_identity(identity))
            self.assertTrue(run.compatible_identity(legacy))
            legacy['sources']['core'] = 'different'
            self.assertFalse(run.compatible_identity(legacy))
            legacy['sources']['core'] = 'same'
            legacy['sources'][run.RUNNER] = 'unreviewed'
            self.assertFalse(run.compatible_identity(legacy))


if __name__ == '__main__':
    unittest.main()

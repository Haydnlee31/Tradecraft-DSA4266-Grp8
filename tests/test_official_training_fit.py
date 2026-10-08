"""Training-only fixtures: no validation/test files even exist in the data root."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.official_training_fit import (
    balanced_rows, load_training_panel, panel_batches, fit_mlp, fit_tree,
    diagnose, validate_plan,
)


PLAN = Path(__file__).parents[1]/'reports/full_data_extension/training-fit-plan.json'


class TrainingFitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(3)
        self.x = rng.normal(size=(96, 39)).astype('float32')
        self.y = np.arange(96, dtype=np.int64) % len(CLASSES)
        self.features = [f'f{i}' for i in range(39)]
        self.scaler = {'features': self.features, 'fit_split': 'train', 'n_samples_seen': 96}
        (self.root/'scaler.json').write_text(json.dumps(self.scaler))
        np.save(self.root/'train_x.npy', self.x)
        np.save(self.root/'train_y.npy', self.y)
        self.manifest = {
            'status': 'complete', 'classes': CLASSES, 'features': self.features,
            'test_opened': False, 'train_rows': 96,
            'train_class_counts': dict.fromkeys(CLASSES, 12),
            'scaler_sha256': sha256(self.root/'scaler.json'),
            'files': {f'train_{k}.npy': sha256(self.root/f'train_{k}.npy') for k in ('x', 'y')},
        }
        self.write_manifest()
        self.plan = json.loads(PLAN.read_text())
        self.plan['packed_manifest_sha256'] = sha256(self.root/'manifest.json')
        self.plan['expected_training_rows'] = 96
        self.plan['panel']['per_class'] = 8
        self.plan['mlp'].update(epochs=2, report_epochs=[1, 2], batch_size=16, seeds=[7])
        self.plan['hist_gradient_boosting'].update(max_iter=2, min_samples_leaf=2, max_leaf_nodes=7)

    def write_manifest(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))

    def test_balanced_sampling_repeatable_bounded_and_without_replacement(self):
        rows = balanced_rows(self.y, 8, 7)
        np.testing.assert_array_equal(rows, balanced_rows(self.y, 8, 7))
        self.assertEqual(len(rows), len(np.unique(rows)))
        np.testing.assert_array_equal(np.bincount(self.y[rows]), np.full(8, 8))
        np.testing.assert_array_equal(balanced_rows(self.y, 12, 7), np.arange(96))
        for labels, cap in ((self.y, 13), (self.y, 0), (self.y, 1270),
                            (self.y.astype(float), 2), (self.y-1, 2)):
            with self.assertRaises(ValueError):
                balanced_rows(labels, cap, 7)

    def test_loader_never_opens_other_splits(self):
        original_load = np.load
        def training_only(path, **kwargs):
            self.assertIn(Path(path).name, ('train_x.npy', 'train_y.npy'))
            return original_load(path, **kwargs)
        with patch('src.eval.official_training_fit.np.load', side_effect=training_only):
            x, y, rows, _ = load_training_panel(self.root, 8, 7)
        np.testing.assert_array_equal(x, self.x[rows])
        np.testing.assert_array_equal(y, self.y[rows])

    def test_loader_rejects_tampered_arrays_and_scaler(self):
        altered = self.x.copy()
        altered[0, 0] += 1
        np.save(self.root/'train_x.npy', altered)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_training_panel(self.root, 8, 7)
        np.save(self.root/'train_x.npy', self.x)
        self.scaler['fit_split'] = 'val'
        (self.root/'scaler.json').write_text(json.dumps(self.scaler))
        self.manifest['scaler_sha256'] = sha256(self.root/'scaler.json')
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'training cohort'):
            load_training_panel(self.root, 8, 7)

    def test_batch_coverage_and_merged_singleton(self):
        x, y = self.x[:17], np.arange(17)
        batches = list(panel_batches(x, y, 8, 9))
        self.assertEqual([len(b) for _, b in batches], [8, 9])
        np.testing.assert_array_equal(np.sort(np.concatenate([b for _, b in batches])), y)

    def test_neural_replay_uniform_weights_and_exact_budget(self):
        x, y, _, _ = load_training_panel(self.root, 8, 7)
        original = x.copy()
        a, b = (fit_mlp(x, y, self.plan['mlp'], 7) for _ in range(2))
        a.pop('elapsed_seconds')
        b.pop('elapsed_seconds')
        self.assertEqual(a, b)
        self.assertEqual(a['examples_processed'], 128)
        self.assertEqual(a['optimizer_steps'], 8)
        self.assertEqual(a['num_parameters'], 5096)
        np.testing.assert_allclose(a['class_weights'], [1.]*8)
        self.assertEqual(len(set(a['class_weights'])), 1)
        np.testing.assert_array_equal(x, original)
        self.assertEqual(a['observations'][-1]['epoch'], 2)

    def test_tree_replay_without_early_stopping(self):
        with threadpool_limits(limits=2):
            a, b = (fit_tree(self.x, self.y, self.plan['hist_gradient_boosting']) for _ in range(2))
        a.pop('elapsed_seconds')
        b.pop('elapsed_seconds')
        self.assertEqual(a, b)
        self.assertEqual(a['trees'], 16)
        bad = copy.deepcopy(self.plan['hist_gradient_boosting'])
        bad['early_stopping'] = True
        with self.assertRaises(ValueError):
            fit_tree(self.x, self.y, bad)

    def test_plan_rejects_scope_expansion(self):
        validate_plan(self.plan)
        for group, key, value in ((None, 'validation_opened', True), ('panel', 'replacement', True),
                                  ('mlp', 'epochs', 101), ('hist_gradient_boosting', 'early_stopping', True)):
            bad = copy.deepcopy(self.plan)
            (bad if group is None else bad[group])[key] = value
            with self.assertRaises(ValueError):
                validate_plan(bad)

    def test_complete_receipt_and_no_overwrite(self):
        plan_path, output = self.root/'plan.json', self.root/'result.json'
        plan_path.write_text(json.dumps(self.plan))
        r = diagnose(self.root, plan_path, output)
        self.assertEqual(r['status'], 'complete')
        self.assertFalse(r['validation_opened'])
        self.assertFalse(r['test_opened'])
        self.assertEqual(r['panel_counts'], dict.fromkeys(CLASSES, 8))
        self.assertEqual(len(r['mlp_fits']), 1)
        self.assertEqual(r['plan_sha256'], sha256(plan_path))
        with self.assertRaises(FileExistsError):
            diagnose(self.root, plan_path, output)


if __name__ == '__main__':
    unittest.main()

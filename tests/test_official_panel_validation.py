"""Frozen model replay, bounded validation and a sealed final-test boundary."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import official_training_fit as training
from src.eval.official_dropout_fit import comparable
from src.eval.official_width_fit import diagnose as width_diagnose
from src.eval import official_panel_validation as validation
from src.models.official_streaming import evaluate
from tests import test_official_width_fit as fixtures


class PanelValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WidthFitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        # Find the shared tiny packed fixture beneath the historical wrappers.
        self.training = self.fixture
        while hasattr(self.training, 'fixture'):
            self.training = self.training.fixture
        self.counts = dict(zip(CLASSES, (40, 16, 12, 9, 7, 5, 4, 3)))
        self.vx = np.random.default_rng(44).normal(size=(96, 39)).astype('float32')
        self.vy = np.repeat(np.arange(8, dtype=np.int64), list(self.counts.values()))
        np.save(self.root/'val_x.npy', self.vx)
        np.save(self.root/'val_y.npy', self.vy)
        self.training.manifest.update(val_rows=96)
        self.training.manifest['files'].update({f'val_{k}.npy': sha256(self.root/f'val_{k}.npy') for k in ('x', 'y')})
        # A bogus test entry must never be opened, loaded, or hashed.
        self.training.manifest['files']['test_x.npy'] = 'sealed-and-nonexistent'
        self.training.write_manifest()
        self.training.plan['packed_manifest_sha256'] = sha256(self.root/'manifest.json')

    def reference_files(self):
        duration_path, width_plan, _, _ = self.fixture.reference_files()
        ref_path = self.root/'width-result.json'
        ref = width_diagnose(self.root, duration_path, width_plan, ref_path)
        tree_path = self.root/'reference.json'
        plan = json.loads((Path(__file__).parents[1]/'reports/full_data_extension/panel-validation-plan.json').read_text())
        plan.update(reference_receipt_sha256=sha256(ref_path), reference_plan_sha256=ref['plan_sha256'],
                    previous_fitter_sha256=ref['source_sha256']['eval/official_training_fit.py'],
                    tree_reference_sha256=sha256(tree_path), panel_sha256=ref['panel_sha256'],
                    epochs=4, seeds=[7], validation_rows=96, validation_class_counts=self.counts,
                    evaluation_batch_size=17)
        plan_path = self.root/'validation-plan.json'
        plan_path.write_text(json.dumps(plan))
        return ref_path, tree_path, plan_path, ref, plan

    def test_returning_models_preserves_the_training_receipts(self):
        x, y, _, _ = training.load_training_panel(self.root, 8, 7)
        settings = self.training.plan['mlp']
        old = training.fit_mlp(x, y, settings, 7)
        new, model = training.fit_mlp(x, y, settings, 7, return_model=True)
        self.assertEqual(comparable(old), comparable(new))
        self.assertEqual(training.state_hash(model), old['final_state_sha256'])
        tree_settings = self.training.plan['hist_gradient_boosting']
        a = training.fit_tree(x, y, tree_settings)
        b, tree = training.fit_tree(x, y, tree_settings, return_model=True)
        self.assertEqual(comparable(a), comparable(b))
        self.assertEqual(len(tree.predict(x)), len(y))

    def test_bounded_batches_cover_every_row_without_shuffle(self):
        batches = list(validation.validation_batches(self.vx, self.vy, 17))
        self.assertEqual([len(y) for _, y in batches], [17, 17, 17, 17, 17, 11])
        np.testing.assert_array_equal(np.concatenate([x for x, _ in batches]), self.vx)
        np.testing.assert_array_equal(np.concatenate([y for _, y in batches]), self.vy)
        self.assertTrue(all(x.flags.writeable and y.flags.writeable for x, y in batches))
        for size in (0, 8193):
            with self.assertRaises(ValueError):
                list(validation.validation_batches(self.vx, self.vy, size))

    def test_loader_checks_only_validation_arrays_and_returns_memmaps(self):
        plan = {'validation_rows': 96, 'validation_class_counts': self.counts, 'evaluation_batch_size': 17}
        real_load, real_sha = np.load, sha256
        def restricted_load(path, **kwargs):
            self.assertIn(Path(path).name, ('val_x.npy', 'val_y.npy'))
            return real_load(path, **kwargs)
        def restricted_sha(path):
            self.assertIn(Path(path).name, ('val_x.npy', 'val_y.npy'))
            return real_sha(path)
        with patch('src.eval.official_panel_validation.np.load', side_effect=restricted_load), \
             patch('src.eval.official_panel_validation.sha256', side_effect=restricted_sha):
            x, y = validation.load_validation(self.root, self.training.manifest, plan)
        self.assertIsInstance(x, np.memmap)
        self.assertIsInstance(y, np.memmap)

    def test_loader_rejects_tampering_nonfinite_and_count_changes(self):
        plan = {'validation_rows': 96, 'validation_class_counts': self.counts, 'evaluation_batch_size': 17}
        bad = self.vx.copy()
        bad[0, 0] = np.nan
        np.save(self.root/'val_x.npy', bad)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validation.load_validation(self.root, self.training.manifest, plan)
        self.training.manifest['files']['val_x.npy'] = sha256(self.root/'val_x.npy')
        with self.assertRaisesRegex(ValueError, 'Invalid validation'):
            validation.load_validation(self.root, self.training.manifest, plan)
        np.save(self.root/'val_x.npy', self.vx)
        self.training.manifest['files']['val_x.npy'] = sha256(self.root/'val_x.npy')
        plan['validation_class_counts'] = dict.fromkeys(CLASSES, 12)
        with self.assertRaisesRegex(ValueError, 'class counts'):
            validation.load_validation(self.root, self.training.manifest, plan)

    def test_streamed_predictions_match_one_batch(self):
        _, model = training.fit_mlp(self.training.x, self.training.y, self.training.plan['mlp'], 7, return_model=True)
        _, a = evaluate(model, [(self.vx, self.vy)], 'cpu')
        _, b = evaluate(model, validation.validation_batches(self.vx, self.vy, 17), 'cpu')
        self.assertEqual(a, b)
        _, tree = training.fit_tree(self.training.x, self.training.y, self.training.plan['hist_gradient_boosting'], return_model=True)
        self.assertEqual(validation.evaluate_tree(tree, [(self.vx, self.vy)]),
                         validation.evaluate_tree(tree, validation.validation_batches(self.vx, self.vy, 17)))

    def test_complete_receipt_checkpoints_and_no_test_access(self):
        ref, tree, plan, _, _ = self.reference_files()
        output = self.root/'validation'
        real_sha, real_load = sha256, np.load
        def no_test_sha(path):
            self.assertFalse(Path(path).name.startswith('test_'))
            return real_sha(path)
        def no_test_load(path, **kwargs):
            self.assertFalse(Path(path).name.startswith('test_'))
            return real_load(path, **kwargs)
        with patch('src.eval.official_panel_validation.sha256', side_effect=no_test_sha), \
             patch('src.eval.official_panel_validation.np.load', side_effect=no_test_load):
            r = validation.diagnose(self.root, ref, tree, plan, output)
        self.assertEqual(r['status'], 'complete')
        self.assertTrue(r['all_training_bridges_exact'])
        self.assertTrue(r['validation_opened'])
        self.assertFalse(r['test_opened'])
        self.assertEqual(len(r['validation_results']), 5)
        for fit in r['neural_fits']:
            checkpoint = output/fit['checkpoint']
            self.assertEqual(sha256(checkpoint), fit['checkpoint_sha256'])
            saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
            self.assertEqual(saved['panel_sha256'], r['panel_sha256'])
        for row in r['validation_results']:
            self.assertEqual(sum(p['support'] for p in row['validation_metrics']['per_class'].values()), 96)
        with self.assertRaises(FileExistsError):
            validation.diagnose(self.root, ref, tree, plan, output)

    def test_bad_reference_hash_blocks_all_data_loading(self):
        ref, tree, plan_path, _, plan = self.reference_files()
        plan['reference_receipt_sha256'] = 'wrong'
        plan_path.write_text(json.dumps(plan))
        with patch('src.eval.official_panel_validation.load_training_panel') as loading:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validation.diagnose(self.root, ref, tree, plan_path, self.root/'bad')
        loading.assert_not_called()

    def test_neural_bridge_failure_leaves_validation_sealed(self):
        ref, tree, plan_path, reference, plan = self.reference_files()
        reference['wide_fits']['zero_dropout_fits'][-1]['final_state_sha256'] = 'wrong'
        ref.write_text(json.dumps(reference))
        plan['reference_receipt_sha256'] = sha256(ref)
        plan_path.write_text(json.dumps(plan))
        output = self.root/'failed'
        with patch('src.eval.official_panel_validation.load_validation') as loading:
            with self.assertRaisesRegex(ValueError, 'Neural training bridge'):
                validation.diagnose(self.root, ref, tree, plan_path, output)
        loading.assert_not_called()
        r = json.loads((output/'receipt.json').read_text())
        self.assertFalse(r['validation_opened'])
        self.assertFalse(r['all_training_bridges_exact'])
        self.assertEqual(r['status'], 'incomplete')

    def test_tree_bridge_failure_leaves_validation_sealed(self):
        ref, tree, plan_path, _, plan = self.reference_files()
        reference = json.loads(tree.read_text())
        reference['tree_fit']['training_prediction_sha256'] = 'wrong'
        tree.write_text(json.dumps(reference))
        plan['tree_reference_sha256'] = sha256(tree)
        plan_path.write_text(json.dumps(plan))
        output = self.root/'failed'
        with patch('src.eval.official_panel_validation.load_validation') as loading:
            with self.assertRaisesRegex(ValueError, 'Tree training bridge'):
                validation.diagnose(self.root, ref, tree, plan_path, output)
        loading.assert_not_called()
        self.assertFalse(json.loads((output/'receipt.json').read_text())['validation_opened'])

    def test_protocol_rejects_new_controls_or_missing_seeds(self):
        _, tree, _, reference, plan = self.reference_files()
        tree_reference = json.loads(tree.read_text())
        for key, value in (('prediction_rule', 'threshold'), ('test_opened', True),
                            ('epochs', 300), ('model_promoted', True), ('seeds', [7, 17])):
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError):
                validation.check_protocol(bad, reference, tree_reference)
        reference['wide_fits']['zero_dropout_fits'] = []
        with self.assertRaisesRegex(ValueError, 'seed coverage'):
            validation.check_protocol(plan, reference, tree_reference)


if __name__ == '__main__':
    unittest.main()

"""Synthetic numerical/bridge tests; no real train, validation or test rows."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.eval.federated_class_safeguard import (
    TrainOnlyData, PROTECTED, project_delta, class_gradients, training_evaluation,
    fedavg_round, inspect_step, moved_model, vector, gate, state_digest,
    verify_source, RUNNER_BRIDGE,
)
from src.eval.official_aggregation_probe import one_round
from src.models.architectures import MLPClassifier, MLPConfig
from src.utils.seed import set_seed


class SafeguardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(7)
        y = np.arange(80, dtype=np.int64) % 8
        x = rng.normal(size=(80, 39)).astype(np.float32)
        self.data = object.__new__(TrainOnlyData)
        self.data.arrays = {'train': (x, y)}
        self.data.manifest = {'features': [f'f{i}' for i in range(39)], 'train_rows': 80,
                              'train_class_counts': dict.fromkeys(CLASSES, 10)}
        # Deliberately unequal and label-skewed clients. Some own no rare rows.
        self.parts = [np.arange(7), np.arange(7, 35), np.arange(35, 80)]
        self.settings = {'seed': 7, 'lr': .001, 'weight_decay': 1e-5, 'batch_size': 8,
                         'loss': 'sqrt_weighted_ce', 'loss_reduction': 'batch_weight_sum'}
        set_seed(7)
        self.model = MLPClassifier(39, 8, MLPConfig('fixture', (8,), .2, 'layer'))

    def test_projection_inactive_is_exact_identity(self):
        d = np.array([-1., -2., 3.])
        p, info = project_delta(d, np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]))
        np.testing.assert_array_equal(d, p)
        self.assertFalse(info['changed'])

    def test_narrow_historical_source_bridge(self):
        checked = verify_source('models/official_streaming.py', RUNNER_BRIDGE[0])
        self.assertFalse(checked['exact_file'])
        self.assertIn('train_epoch', checked['imported_function_ast_sha256'])
        with self.assertRaisesRegex(ValueError, 'source changed'):
            verify_source('models/official_streaming.py', '0'*64)

    def test_projection_orthogonal_constraints_and_sign(self):
        d, _ = project_delta([2., 3., -4.], [[1., 0., 0.], [0., 1., 0.]])
        np.testing.assert_allclose(d, [0, 0, -4], atol=1e-12)

    def test_projection_correlated_and_duplicate_constraints(self):
        d, _ = project_delta([2., 1.], [[1., 1.], [2., 2.], [0., 1.]])
        np.testing.assert_allclose(d, [.5, -.5], atol=1e-12)

    def test_projection_opposing_constraints_can_stall(self):
        d, _ = project_delta([2.], [[1.], [-1.]])
        np.testing.assert_allclose(d, [0.], atol=1e-12)

    def test_projection_positive_rescaling_invariant_and_nearest(self):
        rng = np.random.default_rng(4266)
        for _ in range(20):
            g, d = rng.normal(size=(3, 9)), rng.normal(size=9)
            p, info = project_delta(d, g)
            q, _ = project_delta(d, g*np.array([1e-6, 1e3, 2.])[:, None])
            np.testing.assert_allclose(p, q, atol=1e-10)
            self.assertLessEqual(np.max(g @ p), 1e-8)
            self.assertLessEqual(np.linalg.norm(p-d), np.linalg.norm(d)+1e-8)
            self.assertLessEqual(info['kkt_residual'], 1e-8)

    def test_projection_rejects_nonfinite_and_invalid_shapes(self):
        for d, g in (([np.nan], [[1.]]), ([1.], [[np.inf]]), ([1., 2.], [[1.]]),
                     ([1.], [[1.]]*4), ([[1.]], [[1.]])):
            with self.subTest(d=d, g=g), self.assertRaises(ValueError):
                project_delta(d, g)

    def test_gradient_pooling_matches_central_gradient_and_preserves_rng(self):
        before, state = state_digest(self.model), torch.get_rng_state().clone()
        g, work = class_gradients(self.model, self.data, self.parts, 8)
        pooled, _ = class_gradients(self.model, self.data, [np.arange(80)], 8)
        np.testing.assert_allclose(g, pooled, atol=3e-7, rtol=2e-5)
        self.assertEqual(state_digest(self.model), before)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(work['gradient_examples'], 80)
        self.assertEqual(work['class_counts'], [10]*8)
        # Class0 loss also sends NONZERO gradients to absent output classes.
        self.assertTrue(np.all(np.abs(g[0, -8:]) > 0))

    def test_gradient_finite_difference(self):
        g, _ = class_gradients(self.model, self.data, self.parts)
        d = np.random.default_rng(17).normal(size=g.shape[1])
        d /= np.linalg.norm(d)
        plus = training_evaluation(moved_model(self.model, d*.001), self.data)
        minus = training_evaluation(moved_model(self.model, -d*.001), self.data)
        numerical = (np.array(plus['mean_class_ce'])-minus['mean_class_ce'])/.002
        np.testing.assert_allclose(numerical, g @ d, atol=3e-4, rtol=.02)

    def test_historical_flower_round_bridge(self):
        ordinary, _, work = fedavg_round(self.model, self.data, self.parts, self.settings, 3)
        # The old diagnostic's validation path is used only on synthetic copies.
        old_data = object.__new__(PackedData)
        old_data.manifest = self.data.manifest
        old_data.arrays = {s: self.data.arrays['train'] for s in ('train', 'val')}
        old = one_round(old_data, self.model, self.parts, self.settings, np.arange(80), 3)
        self.assertEqual(training_evaluation(ordinary, self.data)['metrics'], old['full_validation_parameter_average'])
        self.assertEqual([w['task_loss'] for w in work], [w['task_loss'] for w in old['clients']])
        self.assertEqual(sum(w['examples_processed'] for w in work), 80)

    def test_diagnostic_replays_exactly_and_never_calls_validation(self):
        a = inspect_step(self.model, self.data, self.parts, self.settings, 2)
        b = inspect_step(self.model, self.data, self.parts, self.settings, 2)
        self.assertEqual(a, b)
        self.assertEqual(a['gradient_work']['class_counts'], [10]*8)
        self.assertEqual(set(a['per_class']), set(CLASSES))
        self.assertEqual(a['hypothetical_extra_upload_bytes_per_client'], 3*self.model.parameter_bytes())

    def test_loader_succeeds_without_validation_files_and_refuses_val(self):
        with tempfile.TemporaryDirectory() as td, ExitStack() as mapped_files:
            root = Path(td)
            for key, a in zip(('x', 'y'), self.data.arrays['train']):
                np.save(root/f'train_{key}.npy', a)
            (root/'scaler.json').write_text('{}')
            m = dict(self.data.manifest, status='complete', classes=CLASSES,
                     scaler_sha256=sha256(root/'scaler.json'),
                     files={f'train_{k}.npy': sha256(root/f'train_{k}.npy') for k in ('x', 'y')})
            (root/'manifest.json').write_text(json.dumps(m))
            loaded = TrainOnlyData(root)
            # Windows cannot delete a file while its NumPy mapping is open.
            # Close mappings before TemporaryDirectory, including on failure.
            for array in loaded.arrays['train']:
                mapped_files.callback(array._mmap.close)
            self.assertEqual(sum(len(y) for _, y in loaded.batches('train', 8)), 80)
            with self.assertRaisesRegex(ValueError, 'Only training'):
                list(loaded.batches('val', 8))
            # Byte-level integrity is checked before mapping arrays.
            m['files']['train_x.npy'] = '0'*64
            (root/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'array changed'):
                TrainOnlyData(root)

    def test_gate_requires_actual_loss_safety_and_useful_nonzero_updates(self):
        plan = {'loss_tolerance': 1e-6, 'min_retained_norm_fraction': .1, 'min_useful_checkpoints': 2}
        c = {'retained_norm_fraction': .5, 'per_class': {name: {
            'fedavg_predicted_ce_change': .1, 'fedavg_actual_ce_change': .2,
            'projected_actual_ce_change': -.01} for name in PROTECTED}}
        self.assertEqual(gate([c, c, c], plan)['decision'], 'eligible_for_separately_planned_pilot')
        unsafe = copy.deepcopy(c)
        unsafe['per_class']['Benign']['projected_actual_ce_change'] = .01
        self.assertEqual(gate([c, c, unsafe], plan)['decision'], 'stop_no_full_training')
        stalled = dict(c, retained_norm_fraction=0.)
        self.assertEqual(gate([c, c, stalled], plan)['decision'], 'stop_no_full_training')
        no_harm = copy.deepcopy(c)
        for v in no_harm['per_class'].values():
            v['fedavg_actual_ce_change'] = -.1
        self.assertEqual(gate([no_harm]*3, plan)['decision'], 'stop_no_full_training')


if __name__ == '__main__':
    unittest.main()

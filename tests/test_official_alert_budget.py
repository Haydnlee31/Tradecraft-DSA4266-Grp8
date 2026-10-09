"""Empirical caps are tie-safe and exploratory, with frozen inference only."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import official_alert_budget as gate
from src.eval import official_panel_validation as validation
from src.models.official_streaming import evaluate
from tests import test_official_panel_validation as fixtures


class AlertBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PanelValidationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def reference_files(self):
        ref, tree, previous_plan, _, _ = self.fixture.reference_files()
        folder = self.root/'validation'
        reference = validation.diagnose(self.root, ref, tree, previous_plan, folder)
        plan = json.loads((Path(__file__).parents[1]/'reports/full_data_extension/alert-budget-plan.json').read_text())
        plan.update(reference_receipt_sha256=sha256(folder/'receipt.json'),
                    reference_plan_sha256=reference['plan_sha256'], seeds=[7], evaluation_batch_size=17)
        plan_path = self.root/'alert-plan.json'
        plan_path.write_text(json.dumps(plan))
        return folder, plan_path, reference, plan

    def test_threshold_ties_and_zero_budget_are_conservative(self):
        a = gate.threshold_for_budget(np.array([4., 3., 3., 1., -2.]), .4)
        self.assertEqual(a['allowed_benign_alerts'], 2)
        self.assertEqual(a['threshold'], 3.)
        self.assertEqual(a['observed_benign_alerts'], 1)
        b = gate.threshold_for_budget(np.array([4., 3., 3., 1., -2.]), 0.)
        self.assertEqual(b['threshold'], 4.)
        self.assertEqual(b['observed_benign_alerts'], 0)
        self.assertEqual(gate.threshold_for_budget(np.array([-4., -3., -1., -2.]), .5)['threshold'], 0.)
        self.assertEqual(gate.threshold_for_budget(np.arange(100), .01)['allowed_benign_alerts'], 1)

    def test_invalid_budgets_scores_and_batches_rejected(self):
        for values, budget in (([], .01), ([np.nan], .01), ([np.inf], .01), ([1.], -1.), ([1.], 1.)):
            with self.assertRaises(ValueError):
                gate.threshold_for_budget(values, budget)
        for threshold in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                gate.gated_metrics(np.array([1.]), np.array([1]), np.array([0]), threshold, 1)
        with self.assertRaises(ValueError):
            gate.gated_metrics(np.array([1.]), np.array([0]), np.array([0]), 0., 1)

    def test_higher_thresholds_only_suppress_attack_predictions(self):
        y = np.repeat(np.arange(8), 3)
        winners = np.tile(np.array([1, 4, 5]), 8)
        margins = np.tile(np.array([-1., 1., 3.]), 8)
        previous = gate.gated_metrics(margins, winners, y, 0., 7)
        for threshold in (1., 3.):
            current = gate.gated_metrics(margins, winners, y, threshold, 7)
            self.assertLessEqual(current['benign_false_alert_rate'], previous['benign_false_alert_rate'])
            for name in CLASSES[1:]:
                self.assertLessEqual(current['per_class'][name]['recall'], previous['per_class'][name]['recall'])
            previous = current

    def test_scoring_handles_argmax_ties_and_reproduces_reference(self):
        class IdentityLogits(torch.nn.Module):
            def forward(self, x):
                return x
        x = np.array([[2., 2., 1., 0., 0., 0., 0., 0.],
                      [0., 3., 3., 0., 0., 0., 0., 0.],
                      [0., 1., 2., 3., 4., 5., 6., 7.]], dtype=np.float32)
        y = np.array([0, 1, 7], dtype=np.int64)
        a, b = self.root/'scores.npy', self.root/'winners.npy'
        loss, metrics = gate.collect_scores(IdentityLogits(), x, y, 2, ['Web-based', 'Brute Force'], a, b)
        expected_loss, expected_metrics = evaluate(IdentityLogits(), validation.validation_batches(x, y, 2), 'cpu')
        self.assertEqual(loss, expected_loss)
        self.assertEqual(metrics, expected_metrics)
        self.assertEqual(gate.gated_metrics(np.load(a)[:, 0], np.load(b), y, 0., 2), metrics)

    def test_complete_inference_only_diagnostic_and_no_overwrite(self):
        folder, plan_path, _, _ = self.reference_files()
        output = self.root/'gate'
        real_load, real_sha = np.load, sha256
        def allowed_load(path, **kwargs):
            name = Path(path).name
            self.assertFalse(name.startswith(('train_', 'test_')))
            return real_load(path, **kwargs)
        def allowed_sha(path):
            self.assertFalse(Path(path).name.startswith(('train_', 'test_')))
            return real_sha(path)
        with patch('src.eval.official_alert_budget.np.load', side_effect=allowed_load), \
             patch('src.eval.official_alert_budget.sha256', side_effect=allowed_sha), \
             patch('src.eval.official_training_fit.fit_mlp') as fitting:
            r = gate.diagnose(self.root, folder, plan_path, output)
        fitting.assert_not_called()
        self.assertEqual(r['status'], 'complete')
        self.assertTrue(r['all_argmax_bridges_exact'])
        for key in ('test_opened', 'training_arrays_opened', 'model_trained', 'model_promoted', 'independent_assessment'):
            self.assertFalse(r[key])
        self.assertEqual(len(r['models']), 4)
        self.assertEqual(len(r['operating_points']), 12)
        for row in r['operating_points']:
            self.assertLessEqual(row['observed_benign_alerts'], row['allowed_benign_alerts'])
            self.assertEqual(sum(s['support'] for s in row['validation_metrics']['per_class'].values()), 96)
        with self.assertRaises(FileExistsError):
            gate.diagnose(self.root, folder, plan_path, output)

    def test_corrupt_checkpoint_blocks_validation_loading(self):
        folder, plan_path, reference, _ = self.reference_files()
        path = folder/reference['neural_fits'][-1]['checkpoint']
        path.write_bytes(b'not a valid checkpoint')
        with patch('src.eval.official_alert_budget.load_validation') as loading:
            with self.assertRaisesRegex(ValueError, 'Checkpoint checksum'):
                gate.diagnose(self.root, folder, plan_path, self.root/'bad')
        loading.assert_not_called()

    def test_reference_hash_failure_before_checkpoint_loading(self):
        folder, plan_path, _, plan = self.reference_files()
        plan['reference_receipt_sha256'] = 'wrong'
        plan_path.write_text(json.dumps(plan))
        with patch('src.eval.official_alert_budget.load_checkpoint') as loading:
            with self.assertRaisesRegex(ValueError, 'Reference checksum'):
                gate.diagnose(self.root, folder, plan_path, self.root/'bad')
        loading.assert_not_called()

    def test_failed_last_argmax_bridge_blocks_all_threshold_selection(self):
        folder, plan_path, reference, plan = self.reference_files()
        record = next(r for r in reference['validation_results'] if r['name']=='wide_no_dropout_seed7')
        record['validation_cross_entropy'] += 1.
        (folder/'receipt.json').write_text(json.dumps(reference))
        plan['reference_receipt_sha256'] = sha256(folder/'receipt.json')
        plan_path.write_text(json.dumps(plan))
        output = self.root/'failed'
        with patch('src.eval.official_alert_budget.threshold_for_budget') as thresholds:
            with self.assertRaisesRegex(ValueError, 'Raw argmax bridge failed'):
                gate.diagnose(self.root, folder, plan_path, output)
        thresholds.assert_not_called()
        r = json.loads((output/'receipt.json').read_text())
        self.assertEqual(r['status'], 'incomplete')
        self.assertFalse(r['all_argmax_bridges_exact'])
        self.assertEqual(r['operating_points'], [])

    def test_protocol_refuses_independent_assessment_claims_or_missing_models(self):
        _, _, reference, plan = self.reference_files()
        for key, value in (('assessment_split', 'test'), ('test_opened', True), ('model_trained', True),
                            ('model_promoted', True), ('benign_false_alert_budgets', [.1]), ('score', 'another_gate')):
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError):
                gate.check_protocol(bad, reference)
        reference['neural_fits'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            gate.check_protocol(plan, reference)


if __name__ == '__main__':
    unittest.main()

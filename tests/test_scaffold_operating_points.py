"""Synthetic safeguards for frozen training-only operating-point diagnosis."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.eval.federated_class_safeguard import state_digest
from src.eval.scaffold_operating_points import (
    BENIGN, ARMS, calibration_groups, conservative_threshold, classify,
    metrics, frozen_scores, describe_arm, primary_diagnostic, run,
)
from src.models.architectures import MLPClassifier, MLPConfig

ROOT = Path(__file__).parents[1]
PLAN = ROOT/'reports/federated_class_safeguards/operating-points-plan.json'


class OperatingPointTests(unittest.TestCase):
    def test_groups_are_feature_only_order_independent_and_normalize_zero(self):
        x = np.random.default_rng(7).normal(size=(200, 39)).astype(np.float32)
        x[0] = x[1] = 0
        x[1, 0] = -0.
        x[2] = x[3]
        first = calibration_groups(x, 13)
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2], first[3])
        np.testing.assert_array_equal(first, calibration_groups(x, 97))
        np.testing.assert_array_equal(first, calibration_groups(x[::-1])[::-1])
        self.assertTrue(first.any() and (~first).any())

    def test_group_nonfinite_empty_and_wrong_dtype_fail(self):
        for x in (np.full((2, 39), np.nan, np.float32), np.empty((0, 39), np.float32), np.zeros((2, 39))):
            with self.assertRaises(ValueError): calibration_groups(x)

    def test_conservative_order_statistic_and_ties(self):
        r = conservative_threshold(np.arange(100), 500)
        self.assertEqual(r['threshold'], 94.)
        self.assertEqual(r['false_alerts'], 5)
        tied = conservative_threshold(np.array([0]*90+[1]*9+[2]), 500)
        self.assertEqual(tied['threshold'], 1.)
        self.assertEqual(tied['false_alerts'], 1)
        self.assertEqual(tied['boundary_ties'], 9)
        self.assertEqual(conservative_threshold(np.ones(7), 100)['false_alerts'], 0)
        self.assertEqual(conservative_threshold(np.arange(19), 500)['allowed_false_alerts'], 0)

    def test_calibration_never_exceeds_integer_budget(self):
        rng = np.random.default_rng(91)
        for n in (1, 2, 19, 20, 101, 2000):
            for target in (0, 100, 500, 1000, 9999):
                r = conservative_threshold(rng.integers(-10, 10, n), target)
                self.assertLessEqual(r['false_alerts'], n*target//10000)

    def test_threshold_invalid_inputs_fail(self):
        for scores, target in (([], 500), ([np.nan], 500), ([1], -1), ([1], 10000), ([1], .05)):
            with self.assertRaises(ValueError): conservative_threshold(scores, target)

    def test_rule_strict_ties_and_attack_ranking_preserved(self):
        scores, attacks = np.array([-.5, 0, .5]), np.array([1, 2, 3])
        np.testing.assert_array_equal(classify(scores, attacks, 0), [BENIGN, BENIGN, 3])
        np.testing.assert_array_equal(classify(scores, attacks, -1), attacks)
        with self.assertRaises(ValueError): classify(scores, [0, 1, 2], 0)

    def fixture(self):
        labels = np.tile(np.arange(8), 40)
        mask = np.arange(len(labels)) < len(labels)//2
        scores = np.linspace(-2, 2, len(labels))
        attacks = np.maximum(labels, 1)
        pred = classify(scores, attacks, 0)
        return labels, mask, scores, attacks, pred

    def test_threshold_does_not_use_audit_labels_or_scores_or_calibration_attacks(self):
        labels, mask, scores, attacks, pred = self.fixture()
        first = describe_arm(labels, mask, scores, attacks, pred, [100, 500, 1000])
        changed_labels, changed_scores = labels.copy(), scores.copy()
        changed_labels[~mask] = np.roll(labels[~mask], 7)
        changed_scores[(~mask) | (labels != BENIGN)] += 1000
        other = describe_arm(changed_labels, mask, changed_scores, attacks, pred, [100, 500, 1000])
        for target in first['targets']:
            self.assertEqual(first['targets'][target]['calibration'], other['targets'][target]['calibration'])
        self.assertNotEqual(first['targets']['500']['audit'], other['targets']['500']['audit'])

    def test_ceiling_is_correct_category_not_binary_alarm(self):
        labels, mask, scores, attacks, pred = self.fixture()
        attacks[:] = CLASSES.index('DDoS')
        r = describe_arm(labels, mask, scores, attacks, pred, [500])
        self.assertEqual(r['attack_rank_recall_ceiling']['Web-based']['recall'], 0.)
        self.assertEqual(r['attack_rank_recall_ceiling']['DDoS']['recall'], 1.)
        self.assertGreater(r['targets']['500']['audit_attack_alarm_recall']['Web-based'], 0)
        self.assertEqual(r['targets']['500']['audit']['per_class']['Web-based']['recall'], 0)

    def test_missing_class_partition_fails_without_retry(self):
        labels, mask, scores, attacks, pred = self.fixture()
        labels[:] = BENIGN
        with self.assertRaises(ValueError): describe_arm(labels, mask, scores, attacks, pred, [500])

    def test_inference_only_preserves_weights_and_zero_bridge(self):
        torch.manual_seed(4)
        model = MLPClassifier(39, 8, MLPConfig('test', (8,), .5, 'layer'))
        x = np.random.default_rng(7).normal(size=(33, 39)).astype(np.float32)
        y = np.arange(33) % 8
        class Data:
            arrays = {'train': (x, y)}
            def batches(self, split, size):
                if split != 'train': raise AssertionError('Held-out data opened')
                for start in range(0, len(y), size): yield x[start:start+size], y[start:start+size]
        def hook(module, inputs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(module.training)
        handle = model.register_forward_pre_hook(hook)
        before = state_digest(model)
        with patch.object(torch.Tensor, 'backward', side_effect=AssertionError('No backward allowed')):
            scores, attacks, pred = frozen_scores(model, Data(), 8)
        handle.remove()
        self.assertEqual(before, state_digest(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        np.testing.assert_array_equal(classify(scores, attacks, 0), pred)
        self.assertEqual(sum(map(sum, metrics(y, pred)['confusion_matrix'])), 33)

    def test_primary_only_screen_does_not_rewrite_original_failure(self):
        plan = json.loads(PLAN.read_text())
        m = {'macro_f1': .5, 'benign_false_alert_rate': .05,
             'per_class': {c: {'recall': .1} for c in ('Web-based', 'Brute Force')}}
        arms = {a: {'targets': {'500': {'audit': copy.deepcopy(m)}}} for a in ARMS}
        arms['scaffold']['targets']['500']['audit']['macro_f1'] = .52
        r = primary_diagnostic(arms, plan)
        self.assertTrue(r['eligible_for_separately_planned_confirmation'])
        self.assertFalse(r['prior_training_screen_overturned'])
        self.assertFalse(r['new_training_launched'])
        arms['scaffold']['targets']['500']['audit']['benign_false_alert_rate'] = .06
        self.assertFalse(primary_diagnostic(arms, plan)['eligible_for_separately_planned_confirmation'])
        arms['scaffold']['targets']['500']['audit']['benign_false_alert_rate'] = .05
        arms['scaffold']['targets']['500']['audit']['per_class']['Web-based']['recall'] = 0
        self.assertFalse(primary_diagnostic(arms, plan)['eligible_for_separately_planned_confirmation'])

    def test_existing_output_is_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'preserve'):
            run(None, 'missing', 'missing', 'missing', ROOT)


if __name__ == '__main__':
    unittest.main()

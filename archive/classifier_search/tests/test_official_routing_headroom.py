"""Tiny synthetic routing fixtures; never read private data or cloud archives."""
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import official39_routing_headroom as runner


def fixture():
    y = np.tile(np.arange(runner.N, dtype=np.int64), 3)
    base, expert = y.astype(np.uint8), y.astype(np.uint8)
    # One corrected rare attack, one new benign false alert, two different
    # forms of missed attack, and a Web-based row missed by BOTH models.
    base[4], expert[4] = 0, 4
    expert[0] = 1
    base[5], expert[5] = 0, 1
    base[6], expert[6] = 0, 0
    base[12], expert[12] = 0, 0
    confidence = np.full(len(y), .8, dtype=np.float32)
    confidence[0], confidence[4] = .05, .1
    return y, base, expert, confidence


class RoutingHeadroomTests(unittest.TestCase):
    def test_frozen_scope_and_no_training_flags(self):
        plan = runner.frozen.read(runner.PLAN)
        runner.validate_plan(plan)
        for key, value in [('arm', 'number_masked'), ('seeds', [7]), ('device', 'cuda'),
                           ('budgets_percent', [50]), ('expert_lane', 'heavy')]:
            altered = copy.deepcopy(plan)
            altered[key] = value
            with self.assertRaisesRegex(ValueError, 'scope'):
                runner.validate_plan(altered)
        for flag in runner.FLAGS:
            altered = copy.deepcopy(plan)
            altered[flag] = True
            with self.assertRaisesRegex(ValueError, 'cannot train'):
                runner.validate_plan(altered)

    def test_parent_and_population_hashes_are_bound(self):
        for key in ('parent_plan_sha256', 'validation_panel_receipt_sha256'):
            plan = runner.frozen.read(runner.PLAN)
            plan[key] = '0'*64
            with self.assertRaises(ValueError):
                runner.validate_plan(plan)

    def test_all_nine_models_and_no_masked_arm_promotion(self):
        parent = runner.frozen.read(runner.frozen.PLAN)
        candidates = runner.select_candidates(parent)
        self.assertEqual(len(candidates), 9)
        self.assertEqual({c['arm'] for c in candidates}, {'full39'})
        for altered in (candidates[:-1], candidates+[candidates[0]]):
            with self.assertRaisesRegex(ValueError, 'all nine'):
                runner.select_candidates(dict(candidates=altered))

    def test_oracle_only_corrects_expert_rescuable_rows(self):
        y, base, expert, confidence = fixture()
        r = runner.pair_analysis(y, base, expert, confidence, 7, [10])
        expected = base.copy()
        expected[4] = 4
        self.assertEqual(r['oracle']['metrics'], runner.metrics(y, expected)['metrics'])
        self.assertFalse(r['oracle']['implementable'])
        self.assertFalse(r['oracle']['macro_f1_upper_bound'])
        self.assertEqual(r['oracle']['corrected_rows'], 1)
        self.assertEqual(r['oracle']['required_escalation_rate'], 1/len(y))
        self.assertEqual(r['oracle']['metrics']['per_class']['Web-based']['recall'], 2/3)
        self.assertEqual(r['oracle']['metrics']['benign_false_alert_rate'], 0)

    def test_joint_counts_conserve_population_and_marginals(self):
        y, base, expert, _ = fixture()
        joint = runner.joint_counts(y, base, expert)
        self.assertEqual(int(joint.sum()), len(y))
        self.assertEqual(joint.sum(2).tolist(), runner.metrics(y, base)['metrics']['confusion_matrix'])
        self.assertEqual(joint.sum(1).tolist(), runner.metrics(y, expert)['metrics']['confusion_matrix'])
        # Last-class rows exercise key 511, which would overflow uint8.
        self.assertEqual(int(joint[7, 7, 7]), 3)

    def test_benign_misses_distinguish_correct_category_and_wrong_alert(self):
        r = runner.pair_analysis(*fixture(), 7, [10])
        self.assertEqual(r['per_class']['Web-based']['base_benign_misses'],
                         dict(total=2, corrected_category=1, flagged_wrong_attack=0, still_benign=1))
        self.assertEqual(r['per_class']['Brute Force']['base_benign_misses'],
                         dict(total=1, corrected_category=0, flagged_wrong_attack=1, still_benign=0))
        for c in runner.CLASSES:
            item = r['per_class'][c]
            self.assertEqual(sum(item[k] for k in ('both_correct', 'base_only_correct',
                                                  'expert_only_correct', 'neither_correct')), item['support'])

    def test_budget_control_counts_harm_and_keeps_all_rows(self):
        r = runner.pair_analysis(*fixture(), 7, [1, 5, 10])
        ten = next(c for c in r['controls'] if c['rule'] == 'lowest_base_confidence' and c['budget_percent'] == 10)
        self.assertEqual(ten['escalated_rows'], 2)
        self.assertEqual(ten['corrected_rows'], 1)
        self.assertEqual(ten['damaged_rows'], 1)
        self.assertEqual(ten['metrics']['benign_false_alert_rate'], 1/3)
        self.assertEqual(sum(m['support'] for m in ten['metrics']['per_class'].values()), len(fixture()[0]))
        for control in r['controls']:
            self.assertEqual(sum(c['escalated'] for c in control['per_class_routing'].values()),
                             control['escalated_rows'])

    def test_controls_are_label_blind_reproducible_and_inputs_unchanged(self):
        values = fixture()
        copies = [a.copy() for a in values]
        r = runner.pair_analysis(*values, 17, [10])
        other = runner.pair_analysis(np.roll(values[0], 1), *values[1:], 17, [10])
        self.assertEqual([c['selected_rows_sha256'] for c in r['controls']],
                         [c['selected_rows_sha256'] for c in other['controls']])
        self.assertEqual(r, runner.pair_analysis(*values, 17, [10]))
        for actual, original in zip(values, copies):
            np.testing.assert_array_equal(actual, original)

    def test_confidence_ties_use_original_row_order(self):
        y, base, expert, confidence = fixture()
        confidence[:] = .5
        r = runner.pair_analysis(y, base, expert, confidence, 7, [10])
        take = np.array([0, 1], dtype='<i8')
        self.assertEqual(r['controls'][0]['selected_rows_sha256'], hashlib.sha256(take.tobytes()).hexdigest())

    def test_zero_and_full_budgets_recover_base_and_expert(self):
        r = runner.pair_analysis(*fixture(), 7, [0, 100])
        for control in r['controls']:
            endpoint = 'base' if control['budget_percent'] == 0 else 'always_expert'
            self.assertEqual(control['metrics'], r[endpoint]['metrics'])

    def test_invalid_labels_confidences_and_empty_class_fail(self):
        for broken in (-1, 8, .2):
            y, b, e, conf = fixture()
            y = y.astype(float) if broken == .2 else y
            y[0] = broken
            with self.assertRaisesRegex(ValueError, 'indices'):
                runner.pair_analysis(y, b, e, conf, 7, [10])
        for broken in (np.nan, np.inf, -.1, 1.1):
            y, b, e, conf = fixture()
            conf[0] = broken
            with self.assertRaisesRegex(ValueError, 'confidence'):
                runner.pair_analysis(y, b, e, conf, 7, [10])
        y, b, e, conf = fixture()
        y[y == 7] = 0
        with self.assertRaisesRegex(ValueError, 'Every class'):
            runner.pair_analysis(y, b, e, conf, 7, [10])

    def test_wrong_lengths_and_budgets_fail(self):
        y, b, e, conf = fixture()
        with self.assertRaisesRegex(ValueError, 'Unaligned'):
            runner.pair_analysis(y, b[:-1], e, conf, 7, [10])
        for budgets in ([], [-1], [101], [1.5], [10, 10]):
            with self.assertRaisesRegex(ValueError, 'budgets'):
                runner.pair_analysis(y, b, e, conf, 7, budgets)

    def test_validation_loader_opens_only_two_named_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'val_x.npy', np.zeros((8, 39), np.float32))
            np.save(root/'val_y.npy', np.arange(8, dtype=np.int64))
            manifest = dict(val_rows=8, files={p.name: runner.digest(p) for p in root.iterdir()})
            # An extra manifest entry cannot make this loader touch the test.
            manifest['files']['test_x.npy'] = 'not_a_real_file'
            load = np.load
            opened = []
            def guarded(path, **kwargs):
                self.assertIn(Path(path).name, ('val_x.npy', 'val_y.npy'))
                opened.append(Path(path).name)
                return load(path, **kwargs)
            with patch.object(runner.np, 'load', side_effect=guarded):
                with runner.validation_only(root, manifest) as (x, y):
                    self.assertEqual(x.shape, (8, 39))
                    self.assertEqual(len(y), 8)
            self.assertEqual(opened, ['val_x.npy', 'val_y.npy'])
            self.assertTrue(x._mmap.closed)

    def test_validation_hash_mismatch_precedes_mmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'val_x.npy', np.zeros((8, 39), np.float32))
            with patch.object(runner.np, 'load', side_effect=AssertionError('Must not load')):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    with runner.validation_only(root, dict(files={'val_x.npy': 'wrong'})):
                        pass

    def test_scan_matches_frozen_evaluator_without_changing_model(self):
        runner.frozen.configure('cpu')
        rng = np.random.default_rng(9)
        x = rng.normal(size=(513, 39)).astype(np.float32)
        y = np.arange(len(x), dtype=np.int64) % runner.N
        model = runner.frozen.MLPClassifier(39, 8, replace(runner.frozen.config_for_variant('light'), normalization='layer'))
        features = runner.frozen.read(runner.frozen.PLAN)['data_identity']['features']
        before = runner.frozen.state_digest(model)
        predicted, confidence = runner.scan(model, x, y, features)
        reference = runner.frozen.score(model, runner.frozen.batches(x, y, features, 'full39'),
                                        'cpu', np.ones(len(y), bool))
        self.assertEqual(runner.metrics(y, predicted)['metrics'], reference['all']['metrics'])
        self.assertEqual(before, runner.frozen.state_digest(model))
        self.assertTrue(np.all((confidence >= 1/runner.N) & (confidence <= 1)))
        with patch.object(model, 'forward', return_value=torch.full((513, 8), float('nan'))):
            with self.assertRaisesRegex(ValueError, 'logits'):
                runner.scan(model, x, y, features)

    def test_summary_requires_paired_seeds_and_retains_class_recall(self):
        pairs = []
        for seed in (7, 17, 27):
            for lane in ('iid', 'dirichlet'):
                r = runner.pair_analysis(*fixture(), seed, [10])
                pairs.append(dict(r, seed=seed, base_lane=lane))
        summary = runner.summarize(pairs)
        self.assertEqual(set(summary['iid']['oracle']['per_class_recall']), set(runner.CLASSES))
        self.assertEqual(summary['iid']['base']['macro_f1']['sample_sd'], 0)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            runner.summarize(pairs[:-1])

    def test_existing_output_is_preserved_before_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner.frozen, 'read', side_effect=AssertionError('Must not read')):
                with self.assertRaisesRegex(ValueError, 'preserve'):
                    runner.run('absent', 'absent', 'absent', tmp)

    def test_published_summary_is_aggregate_only_and_preserves_existing_files(self):
        pairs = []
        for seed in (7, 17, 27):
            for lane in ('iid', 'dirichlet'):
                pairs.append(dict(runner.pair_analysis(*fixture(), seed, [10]),
                                  seed=seed, base_lane=lane, expert_lane='light'))
        r = dict(status='complete', scope='synthetic', identity={}, pairs=pairs,
                 summary=runner.summarize(pairs),
                 model_checks=[dict(same_runtime_reference_exact=True, weights_unchanged=True)]*9,
                 **dict.fromkeys(runner.FLAGS, False))
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'receipt.json', Path(tmp)/'summary.json'
            runner.frozen.save(source, r)
            runner.publish_summary(r, source, output)
            result = runner.frozen.read(output)
            self.assertEqual(result['source_receipt_sha256'], runner.digest(source))
            self.assertEqual(result['summary'], r['summary'])
            self.assertNotIn('joint_counts', result['pairs'][0])
            before = output.read_bytes()
            with self.assertRaisesRegex(ValueError, 'preserve'):
                runner.publish_summary(r, source, output)
            self.assertEqual(before, output.read_bytes())
            r['test_opened'] = True
            runner.frozen.save(source, r)
            with self.assertRaisesRegex(ValueError, 'validation-only'):
                runner.publish_summary(r, source, Path(tmp)/'bad.json')


if __name__ == '__main__':
    unittest.main()

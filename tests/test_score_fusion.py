"""No real flow data needed: fitting, leakage guards and automatic inference."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from scripts import official39_score_fusion as runner
from src.eval import score_fusion as f
from src.eval.routing_gate import partitions


def fixture():
    rng = np.random.default_rng(7)
    y = np.tile(np.arange(8, dtype=np.int64), 40)
    p = rng.uniform(.01, .1, (len(y), 8))
    p[np.arange(len(y)), y] += .3
    p /= p.sum(1, keepdims=True)
    e = rng.uniform(.01, .1, (len(y), 8))
    # Expert is deliberately shifted; a head must learn a mapping instead of
    # merely passing through its top label. Classes remain canonical.
    e[np.arange(len(y)), (y+1) % 8] += .4
    e /= e.sum(1, keepdims=True)
    return p.astype(np.float32), e.astype(np.float32), y


class ScoreFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, cls.e, cls.y = fixture()
        cls.z = f.inputs(cls.p, cls.e)
        cls.counts = np.full(8, 40, dtype=np.int64)
        cls.head = f.fit(cls.z, cls.y, cls.counts)

    def test_score_shape_order_and_no_label_input(self):
        self.assertEqual(self.z.shape, (320, 16))
        self.assertTrue(np.array_equal(self.z[:, :8], f.inputs(self.p)))
        self.assertTrue(np.array_equal(self.z[:, 8:], f.inputs(self.e)))
        self.assertEqual(list(inspect.signature(f.inputs).parameters), ['probabilities'])

    def test_zero_scores_clipped_and_all_ties_deterministic(self):
        p = np.eye(8, dtype=np.float32)
        self.assertTrue(np.isfinite(f.inputs(p)).all())
        self.assertEqual(f.inputs(p).min(), np.log(1e-6))
        uniform = np.full((9, 8), .125)
        self.assertTrue(np.all(f.mean_prediction(uniform, uniform) == 0))

    def test_malformed_probability_inputs_rejected(self):
        bad = [self.p[:, :7], self.p*2, self.p*np.nan, self.p*0,
               np.full((3, 8), -.125), self.p[:0], self.p[:, :, None]]
        for p in bad:
            with self.subTest(shape=p.shape), self.assertRaises(ValueError):
                f.inputs(p)
        for args in [(), (self.p, self.e[:3]), (self.p, self.e, self.p)]:
            with self.assertRaises(ValueError):
                f.inputs(*args)

    def test_capped_sample_weights_reconstruct_sqrt_population_mass(self):
        y = np.repeat(np.arange(8), np.arange(1, 9))
        counts = np.arange(1, 9, dtype=np.int64)**2
        w = f.fit_weights(y, counts)
        mass = np.bincount(y, weights=w)
        np.testing.assert_allclose(mass/mass.sum(), np.sqrt(counts)/np.sqrt(counts).sum())
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertGreater(w.min(), 0.)

    def test_weights_reject_missing_classes_or_nonfit_counts(self):
        for y, n in [(self.y, np.ones(8, dtype=int)), (self.y, self.counts.astype(float)),
                     (np.zeros(10, dtype=int), self.counts), (self.y, self.counts[:7])]:
            with self.assertRaises(ValueError):
                f.fit_weights(y, n)

    def test_fit_json_prediction_exact_and_reproducible(self):
        h = json.loads(json.dumps(self.head, allow_nan=False))
        self.assertEqual(h, f.fit(self.z, self.y, self.counts))
        self.assertTrue(h['sklearn_prediction_bridge_exact'])
        self.assertEqual(h['learned_parameters'], 136)
        self.assertTrue(np.array_equal(f.predict(h, self.z), self.y))
        self.assertTrue(np.array_equal(f.predict(h, self.z), np.concatenate([
            f.predict(h, self.z[:77]), f.predict(h, self.z[77:])])) )

    def test_single_model_head_is_real_control(self):
        z = f.inputs(self.e)
        h = f.fit(z, self.y, self.counts)
        self.assertEqual(h['learned_parameters'], 72)
        self.assertTrue(np.array_equal(f.predict(h, z), self.y))
        self.assertFalse(np.array_equal(self.e.argmax(1), self.y))

    def test_constant_scores_have_finite_parameters(self):
        z = f.inputs(np.full((len(self.y), 8), .125))
        h = f.fit(z, self.y, self.counts)
        self.assertTrue(np.all(np.array(h['scale']) == 1.))
        self.assertEqual(f.predict(h, z).shape, self.y.shape)

    def test_nonconvergence_stops_without_retry(self):
        with patch.object(f.LogisticRegression, 'fit', side_effect=ConvergenceWarning('no convergence')) as mock:
            with self.assertRaises(ConvergenceWarning):
                f.fit(self.z, self.y, self.counts)
            self.assertEqual(mock.call_count, 1)

    def test_invalid_head_and_nonfinite_input_rejected(self):
        for field, value in [('classes', list(reversed(f.CLASSES))), ('epsilon', .01),
                             ('coef', [[0]]), ('scale', [0]*16), ('mean', [float('nan')]*16), ('inputs', 39)]:
            h = copy.deepcopy(self.head)
            h[field] = value
            with self.assertRaises(ValueError):
                f.predict(h, self.z)
        for z in (self.z[:, :-1], self.z*np.nan):
            with self.assertRaises(ValueError):
                f.predict(self.head, z)

    def test_grouped_split_unchanged_and_fit_only(self):
        keys = np.arange(30, dtype=np.uint64)
        split, rows = partitions(keys), np.arange(30, dtype=np.int64)
        fit = np.flatnonzero(split == 0)
        f.check_split(rows, split, fit, keys)
        for chosen in [np.array([0, 6]), np.array([0, 0]), np.array([-1, 2]), np.array([31])]:
            with self.assertRaises(ValueError):
                f.check_split(rows, split, chosen, keys)
        bad = split.copy()
        bad[20] = 2
        with self.assertRaisesRegex(ValueError, 'groups'):
            f.check_split(rows, bad, fit[fit != 20], keys)

    def test_repair_and_harm_are_not_review_referrals(self):
        y, base, expert = np.array([0, 1, 2, 3]), np.array([0, 0, 0, 3]), np.array([1, 1, 1, 3])
        c = runner.change_counts(y, base, expert, np.array([1, 1, 2, 3]))
        self.assertEqual((c['corrected'], c['damaged'], c['third_label_correct']), (2, 1, 1))
        self.assertEqual(c['third_label'], 1)

    def test_plan_scope_rejects_test_cloud_review_or_new_lanes(self):
        p = runner.frozen.read(runner.PLAN)
        runner.validate_plan(p)
        for field in runner.FALSE_FLAGS:
            with self.assertRaises(ValueError):
                runner.validate_plan(dict(p, **{field: True}))
        with self.assertRaises(ValueError):
            runner.validate_plan(dict(p, base_lanes=['light']))

    def test_replay_excludes_timing_not_coefficients_or_metrics(self):
        a = dict(seconds=1, elapsed_seconds=2, heads_frozen_sha256='a', latency=[],
                 head=dict(coef=[1.], seconds=2), metrics=dict(macro_f1=.4))
        b = dict(a, seconds=90, elapsed_seconds=200, heads_frozen_sha256='b')
        self.assertEqual(runner.without_timing(a), runner.without_timing(b))
        b['head'] = dict(coef=[2.])
        self.assertNotEqual(runner.without_timing(a), runner.without_timing(b))

    def test_run_preserves_existing_output_before_opening_data(self):
        with self.assertRaisesRegex(ValueError, 'Preserve existing'):
            runner.run('absent', 'absent', 'absent', 'absent', 'absent', Path(__file__).parent)

    def test_fit_cannot_see_changed_audit_labels_and_audit_cannot_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bp, ep = root/'seed7-iid-probabilities.npy', root/'seed7-specialist-probabilities.npy'
            np.save(bp, self.p)
            np.save(ep, self.e)
            np.save(root/'seed7-light-predictions.npy', self.p.argmax(1).astype(np.uint8))
            fit_rows, audit_rows = np.arange(160), np.arange(160, len(self.y))
            heads = {}
            for mode in ('base', 'specialist', 'fusion'):
                heads[mode] = runner.fit_head(root, mode, bp, ep, fit_rows, self.y, self.counts, mode)
            changed = self.y.copy()
            changed[audit_rows] = (changed[audit_rows]+3) % 8
            repeated = runner.fit_head(root, 'repeat', bp, ep, fit_rows, changed, self.counts, 'fusion')
            self.assertEqual(heads['fusion']['sha256'], repeated['sha256'])
            y = self.y[audit_rows]
            old_base = runner.headroom.metrics(y, self.p[audit_rows].argmax(1))
            old_expert = runner.headroom.metrics(y, self.e[audit_rows].argmax(1))
            historical = dict(pairs=[dict(seed=7, lane='iid', base=old_base,
                always=dict(light=old_base, balanced_panel=old_expert))])
            pair = dict(seed=7, lane='iid', heads=heads)
            before = {s['file']: (root/s['file']).read_bytes() for s in heads.values()}
            with patch.object(f, 'fit', side_effect=AssertionError('Audit cannot fit')):
                result = runner.evaluate_pair(pair, root, root, audit_rows, self.y, historical)
            self.assertTrue(result['historical_controls_exact'])
            self.assertEqual(set(result['methods']), set(runner.METHODS))
            self.assertEqual(before, {s['file']: (root/s['file']).read_bytes() for s in heads.values()})


class PublishedFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = runner.ROOT/'reports/risk_aware_routing/fusion-results.json'
        if not path.exists():
            raise unittest.SkipTest('Published experiment not generated yet')
        cls.record = runner.frozen.read(path)

    def test_publication_audit_needs_no_private_flow_data(self):
        with patch.object(np, 'load', side_effect=AssertionError('No private arrays')):
            self.assertEqual(runner.audit_record(self.record)['fitted_heads'], 15)
        self.assertTrue(self.record['non_timing_replay_exact'])

    def test_changed_metrics_scope_and_missing_pairs_fail(self):
        for flag in runner.FALSE_FLAGS:
            with self.assertRaises(ValueError):
                runner.audit_record(dict(self.record, **{flag: True}))
        bad = copy.deepcopy(self.record)
        bad['pairs'][0]['methods']['fusion']['metrics']['macro_f1'] = .999
        with self.assertRaises(ValueError):
            runner.audit_record(bad)
        bad = copy.deepcopy(self.record)
        bad['pairs'].pop()
        with self.assertRaises(ValueError):
            runner.audit_record(bad)

    def test_tampered_corrections_fit_budget_and_timing_fail(self):
        for key in ('correction', 'fit', 'timing', 'screen'):
            bad = copy.deepcopy(self.record)
            if key == 'correction':
                bad['pairs'][0]['methods']['fusion']['changes']['corrected'] += 1
            elif key == 'fit':
                bad['prepared_pairs'][0]['heads']['fusion']['fit_rows'] += 1
            elif key == 'timing':
                bad['latency'][0]['batches'][0]['methods']['fusion']['median_batch_ms'] += 1
            else:
                bad['summary']['continuation_screen_passed'] = not bad['summary']['continuation_screen_passed']
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.audit_record(bad)


if __name__ == '__main__':
    unittest.main()

"""Portable class-weight controls with no original flows or model uploads."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import official39_balanced_tree as r
from src.eval import balanced_tree as b

FEATURES = ['Number', 'Tot sum']+[f'f{i}' for i in range(37)]


class BalancedTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = np.repeat(np.arange(8), np.arange(1, 9)*10).astype(np.int64)
        cls.pop = np.bincount(cls.y)*np.arange(2, 10)
        cls.x = np.random.default_rng(19).normal(size=(len(cls.y), 39)).astype(np.float32)
        cls.x[:, 5] = cls.y
        cls.z = r.ablation.ablation.project(cls.x, FEATURES, r.ARM)
        cls.model, cls.spec = b.fit(cls.z, cls.y, cls.pop, 7)

    def test_equal_class_mass_and_mean_one_on_unequal_sample(self):
        w = b.weights_for(self.y, self.pop)
        self.assertAlmostEqual(w.mean(), 1.)
        np.testing.assert_allclose(np.bincount(self.y, weights=w), np.full(8, len(self.y)/8))
        # Independent scalar definition, not a second call to the helper.
        expected = np.array([len(self.y)/(8*int((self.y == c).sum())) for c in self.y])
        np.testing.assert_array_equal(w, expected)

    def test_full_population_prior_does_not_double_compensate(self):
        np.testing.assert_array_equal(b.weights_for(self.y, self.pop), b.weights_for(self.y, self.pop*100))
        self.assertFalse(np.allclose(b.weights_for(self.y, self.pop), b.weights_for(self.y, self.pop, 'sqrt_population')))

    def test_balanced_sample_reduces_to_uniform_weights(self):
        y = np.tile(np.arange(8), 5)
        np.testing.assert_array_equal(b.weights_for(y, np.arange(1, 9)*10), np.ones(len(y)))

    def test_invalid_scope_missing_classes_and_counts_fail(self):
        for y, pop, mode in [(self.y, self.pop, 'unknown'), (self.y[:10], self.pop, 'equal_class'),
                             (self.y, np.ones(8, np.int64), 'equal_class'),
                             (self.y.astype(float), self.pop, 'equal_class')]:
            with self.assertRaises(ValueError):
                b.weights_for(y, pop, mode)
        with self.assertRaises(ValueError):
            b.fit(self.z.astype(np.float32), self.y, self.pop, 7)

    def test_sqrt_path_reproduces_historical_implementation_exactly(self):
        old, old_spec = b.expert.fit(self.z, self.y, self.pop, 'features', 7)
        control, spec = b.fit(self.z, self.y, self.pop, 7, 'sqrt_population')
        self.assertTrue(all(spec[k] == v for k, v in old_spec.items()))
        self.assertEqual(b.expert.model_bytes(old), b.expert.model_bytes(control))

    def test_fixed_work_and_mask_invariance(self):
        self.assertEqual(self.spec['config'], b.expert.CONFIG)
        self.assertEqual(self.spec['trees'], 800)
        self.assertEqual(self.spec['iterations'], 100)
        self.assertFalse(self.model.do_early_stopping_)
        self.assertEqual(r.ablation.ablation.masked_splits(self.model, FEATURES, r.ARM), 0)
        altered = self.x.copy()
        altered[:, :2] = 12345
        with threadpool_limits(limits=2):
            a = r.ablation.predict_rows(self.model, self.x, np.arange(len(self.y)), FEATURES, r.ARM)
            c = r.ablation.predict_rows(self.model, altered, np.arange(len(self.y)), FEATURES, r.ARM)
        np.testing.assert_array_equal(a, c)

    def test_replay_and_owned_serialization(self):
        model, spec = b.fit(self.z, self.y, self.pop, 7)
        self.assertEqual(spec, self.spec)
        raw = b.expert.model_bytes(model)
        self.assertEqual(raw, b.expert.model_bytes(self.model))
        restored = b.expert.restore_owned(raw, spec['model_sha256'], 'features', 7)
        with threadpool_limits(limits=2):
            np.testing.assert_array_equal(b.expert.predict(restored, self.z), b.expert.predict(model, self.z))
        with patch.object(b.expert.joblib, 'load', side_effect=AssertionError('No untrusted unpickle')):
            with self.assertRaises(ValueError):
                b.expert.restore_owned(raw+b'wrong', spec['model_sha256'], 'features', 7)

    def test_fitting_does_not_mutate_sources_or_use_audit_labels(self):
        rows = np.arange(320)
        z = r.ablation.ablation.project(self.x[rows], FEATURES, r.ARM)
        _, old = b.expert.fit(z, self.y[rows], self.pop, 'features', 7)
        old.update(fit_inputs_sha256=r.ablation.hash_vector(z), original_fit_rows_sha256=r.ablation.hash_vector(rows.astype('<i8')))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y = self.x.copy(), self.y.copy()
            a = root/'a'
            c = root/'c'
            a.mkdir()
            c.mkdir()
            first = r.fit_one(a, x, y, rows, self.pop, FEATURES, 7, old)
            np.testing.assert_array_equal(x, self.x)
            np.testing.assert_array_equal(y, self.y)
            y[320:] = (y[320:]+1) % 8
            second = r.fit_one(c, x, y, rows, self.pop, FEATURES, 7, old)
            self.assertEqual(first['model_sha256'], second['model_sha256'])
            with self.assertRaisesRegex(ValueError, 'Preserve'):
                r.fit_one(a, x, y, rows, self.pop, FEATURES, 7, old)

    def test_scoring_cannot_refit(self):
        with patch.object(b, 'fit', side_effect=AssertionError('No fit during score')), threadpool_limits(limits=2):
            p = r.ablation.predict_rows(self.model, self.x, np.arange(20), FEATURES, r.ARM)
        self.assertEqual(p.shape, (20,))

    def test_pair_counter_has_unambiguous_names_and_valid_marginals(self):
        y = np.tile(np.arange(8), 4)
        a, c = y.copy(), y.copy()
        a[8:24] = (a[8:24]+1) % 8
        c[16:] = (c[16:]+1) % 8
        pair = r.paired(y, a, c)
        for item in pair.values():
            self.assertEqual(item, dict(support=4, both_correct=1, sqrt_only_correct=1,
                equal_only_correct=1, neither_correct=1, oracle_recall_upper_bound=.75))
        r.verify_pair(pair, r.prior.scored(y, a, self.pop), r.prior.scored(y, c, self.pop))

    def test_success_requires_both_rare_f1_and_no_extra_false_alerts(self):
        a = dict(macro_f1=.6, benign_false_alert_rate=.1,
                 per_class={c: dict(f1=.3, recall=.4) for c in (r.WEB_BASED, r.BRUTE_FORCE)})
        b = copy.deepcopy(a)
        b['macro_f1'] = .7
        for c in b['per_class']:
            b['per_class'][c].update(recall=.9, f1=.2)
        b['benign_false_alert_rate'] = .2
        checks = r.screen(dict(seed=7, audit=dict(sqrt_population=dict(metrics=a), equal_class=dict(metrics=b))))
        self.assertTrue(checks['macro_f1_higher'])
        self.assertFalse(any(checks['rare_f1_higher'].values()))
        self.assertFalse(checks['benign_false_alerts_not_higher'])

    def test_plan_and_output_guards(self):
        p = r.frozen.read(r.PLAN)
        r.validate_plan(p)
        for key in r.FLAGS:
            with self.assertRaises(ValueError):
                r.validate_plan(dict(p, **{key: True}))
        for key, value in [('weighting', 'sqrt_population'), ('seeds', [7]),
                           ('classifier', dict(b.expert.CONFIG, max_iter=200))]:
            with self.assertRaises(ValueError):
                r.validate_plan(dict(p, **{key: value}))
        with self.assertRaisesRegex(ValueError, 'Preserve'):
            r.run('none', 'none', 'none', 'none', Path(__file__).parent)
        with self.assertRaises(ValueError):
            r.summarize([])


class PublishedBalancedTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = r.ROOT/'reports/risk_aware_routing/balanced-tree-results.json'
        if not p.exists():
            raise unittest.SkipTest('Experiment not published yet')
        cls.record = r.frozen.read(p)

    def test_aggregate_only_audit_and_exact_replay(self):
        with patch.object(np, 'load', side_effect=AssertionError('No private arrays')), \
                patch.object(b.expert.joblib, 'load', side_effect=AssertionError('No private models')):
            self.assertEqual(r.audit_record(self.record)['trees_fitted'], 2400)
        self.assertTrue(self.record['non_timing_replay_exact'])

    def test_altered_evidence_fails(self):
        for key in r.FLAGS:
            with self.assertRaises(ValueError):
                r.audit_record(dict(self.record, **{key: True}))
        for key in ('weights', 'rows', 'budget', 'metric', 'prior', 'paired', 'screen', 'source'):
            bad = copy.deepcopy(self.record)
            if key == 'weights':
                bad['models'][0]['weighted_class_mass'][0] += 5
            elif key == 'rows':
                bad['models'][0]['original_fit_rows_sha256'] = 'bad'
            elif key == 'budget':
                bad['models'][0]['trees'] += 1
            elif key == 'metric':
                bad['cases'][0]['audit']['equal_class']['metrics']['macro_f1'] = 1.
            elif key == 'prior':
                bad['cases'][0]['fit']['equal_class']['prior_standardized']['macro_f1'] = 1.
            elif key == 'paired':
                bad['cases'][0]['audit_paired_correctness'][b.CLASSES[0]]['both_correct'] += 1
            elif key == 'screen':
                bad['summary']['continuation_screen_passed'] = not bad['summary']['continuation_screen_passed']
            else:
                bad['identity']['source_sha256'] = {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.audit_record(bad)


if __name__ == '__main__':
    unittest.main()

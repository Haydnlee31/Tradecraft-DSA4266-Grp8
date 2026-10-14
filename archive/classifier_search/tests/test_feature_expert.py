"""Small portable fixtures; no original flows, uploads or cloud needed."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import official39_feature_expert as runner
from src.eval import feature_expert as f
from src.eval.official_evidence import digest


def fixture(n=400):
    rng = np.random.default_rng(77)
    y = (np.arange(n) % 8).astype(np.int64)
    x = rng.normal(size=(n, 39)).astype(np.float32)
    x[:, 0] = y + rng.uniform(-.2, .2, n)
    p = np.full((n, 8), .1, np.float32)
    p[np.arange(n), y] = .3
    e = np.roll(p, 1, axis=1).copy()
    return x, p, e, y


class FeatureExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.p, cls.e, cls.y = fixture()
        cls.counts = np.bincount(cls.y)*2
        cls.z = f.inputs('features', x=cls.x)
        cls.model, cls.spec = f.fit(cls.z, cls.y, cls.counts, 'features', 7)

    def test_three_input_schemas_and_identical_score_transform(self):
        z = f.inputs('hybrid', x=self.x, base=self.p, specialist=self.e)
        self.assertEqual(z.shape, (len(self.y), 55))
        self.assertTrue(np.array_equal(z[:, :39], self.z))
        self.assertTrue(np.array_equal(z[:, 39:], f.inputs('scores', base=self.p, specialist=self.e)))
        self.assertTrue(np.array_equal(z[:, 39:], f.score_fusion.inputs(self.p, self.e)))

    def test_unused_inputs_cannot_leak_into_controls(self):
        for args in [dict(mode='features', x=self.x, base=self.p),
                     dict(mode='scores', x=self.x, base=self.p, specialist=self.e),
                     dict(mode='hybrid', x=self.x, base=self.p), dict(mode='unknown')]:
            with self.assertRaises(ValueError):
                f.inputs(**args)

    def test_nonfinite_misaligned_and_wrong_width_fail(self):
        for x in [self.x[:, :38], self.x*np.nan, self.x[:0]]:
            with self.assertRaises(ValueError):
                f.inputs('features', x=x)
        with self.assertRaises(ValueError):
            f.inputs('hybrid', x=self.x[:3], base=self.p, specialist=self.e)
        with self.assertRaises(ValueError):
            f.inputs('scores', base=self.p*0, specialist=self.e)

    def test_exact_eight_class_work_and_weight_objective(self):
        s = self.spec
        self.assertEqual(s['trees'], 800)
        self.assertEqual(s['iterations'], 100)
        self.assertEqual(s['nodes'], 2*s['leaves']-800)
        self.assertLessEqual(s['maximum_leaves_in_tree'], 15)
        self.assertFalse(self.model.do_early_stopping_)
        np.testing.assert_allclose(s['weighted_class_mass'], [50.]*8)

    def test_fit_deterministic_and_serialization_safe_scope(self):
        model, spec = f.fit(self.z, self.y, self.counts, 'features', 7)
        self.assertEqual(self.spec, spec)
        raw = f.model_bytes(model)
        self.assertEqual(raw, f.model_bytes(self.model))
        restored = f.restore_owned(raw, spec['model_sha256'], 'features', 7)
        with threadpool_limits(limits=2):
            self.assertTrue(np.array_equal(f.predict(restored, self.z), f.predict(model, self.z)))
        with patch.object(f.joblib, 'load', side_effect=AssertionError('Do not unpickle changed bytes')):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                f.restore_owned(raw+b'changed', spec['model_sha256'], 'features', 7)

    def test_wrong_seed_schema_or_settings_fail(self):
        for mode, seed in [('scores', 7), ('features', 17)]:
            with self.assertRaises(ValueError):
                f.validate_model(self.model, mode, seed)
        bad = copy.deepcopy(self.model)
        bad.set_params(early_stopping=True)
        with self.assertRaises(ValueError):
            f.validate_model(bad, 'features', 7)
        with self.assertRaises(ValueError):
            f.predict(self.model, self.z[:, :-1])

    def test_prediction_does_not_change_model_and_is_batch_independent(self):
        before = f.state_hash(self.model)
        with threadpool_limits(limits=2):
            p = f.predict(self.model, self.z)
            split = np.concatenate([f.predict(self.model, self.z[:123]), f.predict(self.model, self.z[123:])])
        self.assertTrue(np.array_equal(p, split))
        self.assertEqual(f.state_hash(self.model), before)
        self.assertGreater(np.mean(p == self.y), .9)

    def test_feature_only_never_opens_score_cache(self):
        with patch.object(np, 'load', side_effect=AssertionError('Features only')):
            z = runner.sample_inputs('features', self.x, Path('missing'), Path('missing'), np.arange(10))
        self.assertTrue(np.array_equal(z, self.z[:10]))

    def test_fit_uses_only_fit_rows_not_changed_audit_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = np.arange(320)
            first = runner.fit_one(root, 'first', 'features', 7, self.x, self.y, rows, self.counts, None, None)
            changed = self.y.copy()
            changed[320:] = (changed[320:]+1) % 8
            second = runner.fit_one(root, 'second', 'features', 7, self.x, changed, rows, self.counts, None, None)
            self.assertEqual(first['model_sha256'], second['model_sha256'])
            self.assertEqual(first['fit_inputs_sha256'], second['fit_inputs_sha256'])
            with self.assertRaisesRegex(ValueError, 'Preserve'):
                runner.fit_one(root, 'first', 'features', 7, self.x, self.y, rows, self.counts, None, None)

    def test_batched_scoring_cannot_fit(self):
        with patch.object(f, 'fit', side_effect=AssertionError('No refit')), threadpool_limits(limits=2):
            p = runner.predict_rows(self.model, 'features', self.x, np.arange(20), None, None)
        self.assertEqual(p.shape, (20,))

    def test_plan_rejects_cloud_test_review_and_changed_budget(self):
        p = runner.frozen.read(runner.PLAN)
        runner.validate_plan(p)
        for flag in runner.FALSE_FLAGS:
            with self.assertRaises(ValueError):
                runner.validate_plan(dict(p, **{flag: True}))
        bad = copy.deepcopy(p)
        bad['classifier']['max_iter'] = 200
        with self.assertRaises(ValueError):
            runner.validate_plan(bad)

    def test_summary_requires_all_paired_seeds(self):
        with self.assertRaises(ValueError):
            runner.summarize([])

    def test_success_screen_checks_rare_f1_and_alerts_not_recall_alone(self):
        ref = dict(macro_f1=.6, benign_false_alert_rate=.1,
            per_class={c: dict(f1=.2) for c in ['Web-based', 'Brute Force']})
        treatment = copy.deepcopy(ref)
        treatment.update(macro_f1=.7, benign_false_alert_rate=.2)
        treatment['per_class']['Web-based']['f1'] = .3
        checks = runner.meets_reference(treatment, ref)
        self.assertTrue(checks['macro_f1_higher'])
        self.assertFalse(checks['both_rare_f1_higher'])
        self.assertFalse(checks['benign_false_alerts_not_higher'])

    def test_no_overwrite_before_any_input_opened(self):
        with self.assertRaisesRegex(ValueError, 'Preserve existing'):
            runner.run('missing', 'missing', 'missing', 'missing', 'missing', 'missing', Path(__file__).parent)

    def test_replay_keeps_model_and_prediction_identity(self):
        a = dict(seconds=1, models_frozen_sha256='a', latency=[], model_sha256='X', prediction_sha256='Y')
        b = dict(a, seconds=2, models_frozen_sha256='b')
        self.assertEqual(runner.without_timing(a), runner.without_timing(b))
        self.assertNotEqual(runner.without_timing(a), runner.without_timing(dict(b, model_sha256='changed')))


class PublishedFeatureExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = runner.ROOT/'reports/risk_aware_routing/feature-expert-results.json'
        if not path.exists():
            raise unittest.SkipTest('Experiment not published yet')
        cls.record = runner.frozen.read(path)

    def test_aggregate_audit_without_private_models_or_data(self):
        with patch.object(np, 'load', side_effect=AssertionError('No private arrays')), \
                patch.object(f.joblib, 'load', side_effect=AssertionError('No model unpickling')):
            self.assertEqual(runner.audit_record(self.record)['unique_fits'], 15)
        self.assertTrue(self.record['non_timing_replay_exact'])

    def test_tampered_metrics_scope_counts_and_weights_fail(self):
        for field in runner.FALSE_FLAGS:
            with self.assertRaises(ValueError):
                runner.audit_record(dict(self.record, **{field: True}))
        for field in ['metric', 'counts', 'weight', 'budget', 'latency', 'screen']:
            bad = copy.deepcopy(self.record)
            if field == 'metric':
                bad['pairs'][0]['methods']['hybrid']['metrics']['macro_f1'] = 1.
            elif field == 'counts':
                bad['pairs'][0]['methods']['hybrid']['changes']['damaged'] += 1
            elif field == 'weight':
                bad['prepared_pairs'][0]['models']['hybrid']['weighted_class_mass'][0] += 100
            elif field == 'budget':
                bad['prepared_pairs'][0]['models']['hybrid']['trees'] += 1
            elif field == 'latency':
                bad['latency'][0]['batches'][0]['methods']['hybrid']['median_batch_ms'] += 1
            else:
                bad['summary']['hybrid_screen_passed'] = not bad['summary']['hybrid_screen_passed']
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.audit_record(bad)

    def test_shared_feature_model_and_historical_controls_match(self):
        for seed in [7, 17, 27]:
            p = [p for p in self.record['pairs'] if p['seed'] == seed]
            self.assertEqual(p[0]['methods']['features']['metrics'], p[1]['methods']['features']['metrics'])
        bad = copy.deepcopy(self.record)
        bad['pairs'].pop()
        with self.assertRaises(ValueError):
            runner.audit_record(bad)


if __name__ == '__main__':
    unittest.main()

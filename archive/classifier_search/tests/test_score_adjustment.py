"""Synthetic decision-rule and evidence checks; no dataset needed in CI."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import official39_score_adjustment as r
from src.eval import score_adjustment as s

FEATURES = ['Number', 'Tot sum']+[f'f{i}' for i in range(37)]


def metric(macro=.6, rare=.3, far=.1):
    return dict(macro_f1=macro, benign_false_alert_rate=far,
                per_class={c: dict(f1=rare) for c in s.RARE})


def calibration():
    return [dict(seed=seed, scores={v['id']: dict(metrics=metric()) for v in s.candidates()}) for seed in s.SEEDS]


class ScoreAdjustmentTests(unittest.TestCase):
    def test_zero_is_exact_argmax_with_zero_probabilities_and_ties(self):
        p = np.array([[.5, .5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1.]])
        for prior in (np.ones(8, dtype=int), np.arange(1, 9)):
            np.testing.assert_array_equal(s.predict(p, prior, 0.), [0, 7])

    def test_matches_direct_positive_probability_formula_without_mutation(self):
        p = np.random.default_rng(41).dirichlet(np.ones(8), size=500)
        pop = np.arange(1, 9)**3
        untouched, pop_before = p.copy(), pop.copy()
        for alpha in s.EXPONENTS:
            expected = (p*(pop/pop.sum())**alpha).argmax(axis=1)
            np.testing.assert_array_equal(s.predict(p, pop, alpha), expected)
            np.testing.assert_array_equal(s.predict(p, pop*10, alpha), expected)
        np.testing.assert_array_equal(p, untouched)
        np.testing.assert_array_equal(pop, pop_before)

    def test_invalid_inputs_and_unplanned_exponents_fail_closed(self):
        p, pop = np.full((2, 8), .125), np.ones(8, dtype=int)
        for bad in [p[:, :7], p*2, np.full((2, 8), np.nan), np.full((2, 8), -.125), np.empty((0, 8))]:
            with self.assertRaises(ValueError):
                s.predict(bad, pop, 0.)
        for badpop in [np.zeros(8, int), np.ones(7, int), pop.astype(float)]:
            with self.assertRaises(ValueError):
                s.predict(p, badpop, .5)
        for alpha in [.1, -1., 2., float('nan'), True]:
            with self.assertRaises(ValueError):
                s.predict(p, pop, alpha)

    def test_no_improvement_means_baseline_not_success(self):
        chosen = s.select(calibration())
        self.assertTrue(chosen['fallback_to_baseline'])
        self.assertEqual(chosen['eligible_alternatives'], 0)
        self.assertEqual(chosen['selected_rule']['id'], s.BASELINE)

    def test_every_seed_every_guard_and_both_rare_f1_required(self):
        candidates = s.candidates()
        for failure in ('macro', 'far', *s.RARE):
            cases = calibration()
            target = candidates[5]['id']
            for case in cases:
                case['scores'][target]['metrics'] = metric(.7, .4, .09)
            failed = cases[2]['scores'][target]['metrics']
            if failure == 'macro':
                failed['macro_f1'] = .6
            elif failure == 'far':
                failed['benign_false_alert_rate'] = .10001
            else:
                failed['per_class'][failure]['f1'] = .3
            self.assertTrue(s.select(cases)['fallback_to_baseline'])

    def test_one_shared_rule_and_declared_tie_breakers(self):
        cases = calibration()
        a, b, c = [v['id'] for v in s.candidates()[5:8]]
        for case in cases:
            case['scores'][a]['metrics'] = metric(.8, .4, .09)
            case['scores'][b]['metrics'] = metric(.7, .5, .1)
            case['scores'][c]['metrics'] = metric(.7, .5, .1)
        self.assertEqual(s.select(cases)['selected_rule']['id'], b)
        for case in cases:
            case['scores'][c]['metrics']['macro_f1'] = .71
        self.assertEqual(s.select(cases)['selected_rule']['id'], c)
        self.assertEqual(s.select(cases)['eligible_alternatives'], 3)

    def test_no_best_seed_or_missing_candidate(self):
        cases = calibration()
        with self.assertRaises(ValueError):
            s.select(cases[:2])
        del cases[0]['scores'][s.BASELINE]
        with self.assertRaises(ValueError):
            s.select(cases)

    def test_label_blind_screen_excludes_full_fit_and_audit_matches(self):
        x = np.random.default_rng(11).normal(size=(8, 39)).astype(np.float32)
        x[3], x[4], x[5] = x[0], x[1], x[2]
        x[3:6, :2] += 1000  # All three become matches only after the joint mask.
        before = x.copy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kept, p = r.calibration_panel(x, FEATURES, np.array([0, 1]), np.array([0]),
                np.array([3, 4, 5, 6]), np.array([2, 7]), root)
            np.testing.assert_array_equal(kept, [6])
            self.assertEqual(p['excluded_fit'], 2)
            self.assertEqual(p['excluded_audit'], 1)
            self.assertEqual(p['against_fit']['matches_only_to_unsampled_fit_rows'], 1)
            self.assertFalse(p['against_audit']['labels_used_for_membership'])
            with self.assertRaisesRegex(ValueError, 'Preserve'):
                r.calibration_panel(x, FEATURES, np.array([0, 1]), np.array([0]),
                    np.array([3, 4, 5, 6]), np.array([2, 7]), root)
        np.testing.assert_array_equal(x, before)

    def test_frozen_probability_path_bridges_labels_and_cannot_train(self):
        y = np.tile(np.arange(8), 10)
        x = np.random.default_rng(5).normal(size=(80, 39)).astype(np.float32)
        z = r.ablation.ablation.project(x, FEATURES, r.prior.ARM)
        model, spec = r.prior.balanced.fit(z, y, np.bincount(y)*2, 7)
        with patch.object(type(model), 'fit', side_effect=AssertionError('No retraining')), threadpool_limits(limits=2):
            p = r.probability_rows(model, x, np.arange(80), FEATURES)
            np.testing.assert_array_equal(s.predict(p, np.bincount(y), 0), model.predict(z))
            altered = x.copy()
            altered[:, :2] += 1000
            np.testing.assert_array_equal(p, r.probability_rows(model, altered, np.arange(80), FEATURES))
        self.assertEqual(r.prior.balanced.expert.state_hash(model), spec['model_sha256'])

    def test_freeze_excludes_audit_and_covers_all_selection_inputs(self):
        result = dict(identity={}, calibration_panel={}, calibration=calibration(), selection=s.select(calibration()),
                      audit='must not enter freeze')
        self.assertNotIn('audit', r.freeze_content(result))
        self.assertEqual(r.freeze_content(result)['selection'], result['selection'])

    def test_plan_and_output_guards(self):
        p = r.frozen.read(r.PLAN)
        r.validate_plan(p)
        for key in (*r.FALSE_FLAGS, *r.TRUE_FLAGS):
            with self.assertRaises(ValueError):
                r.validate_plan(dict(p, **{key: not p[key]}))
        for key, value in [('exponents', [0, .1]), ('seeds', [7]), ('audit', 'choose best audit')]:
            with self.assertRaises(ValueError):
                r.validate_plan(dict(p, **{key: value}))
        with self.assertRaisesRegex(ValueError, 'Preserve'):
            r.run('none', 'none', 'none', 'none', 'none', Path(__file__).parent)


class PublishedScoreAdjustmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = r.ROOT/'reports/risk_aware_routing/score-adjustment-results.json'
        if not path.exists():
            raise unittest.SkipTest('Experiment not published yet')
        cls.record = r.frozen.read(path)

    def test_aggregate_only_audit_with_no_private_data_or_models(self):
        with patch.object(np, 'load', side_effect=AssertionError('No private arrays')), \
                patch.object(r.prior.balanced.expert.joblib, 'load', side_effect=AssertionError('No private models')):
            self.assertEqual(r.audit_record(self.record)['new_models_fitted'], 0)
        self.assertTrue(self.record['non_timing_replay_exact'])

    def test_altered_evidence_fails(self):
        for change in ('prior', 'selection', 'scores', 'panel', 'audit', 'summary', 'source', 'training'):
            bad = copy.deepcopy(self.record)
            if change == 'prior':
                bad['fit_population_counts'][0] += 1
            elif change == 'selection':
                bad['selection']['eligible_alternatives'] += 1
            elif change == 'scores':
                bad['calibration'][0]['scores'][s.BASELINE]['metrics']['macro_f1'] = 1.
            elif change == 'panel':
                bad['calibration_panel']['excluded_audit'] += 1
            elif change == 'audit':
                bad['audit'][0]['scores']['posthoc_extra'] = bad['audit'][0]['scores']['selected']
            elif change == 'summary':
                bad['summary']['continuation_screen_passed'] = not bad['summary']['continuation_screen_passed']
            elif change == 'source':
                bad['identity']['source_sha256'] = {}
            else:
                bad['new_model_count'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                r.audit_record(bad)


if __name__ == '__main__':
    unittest.main()

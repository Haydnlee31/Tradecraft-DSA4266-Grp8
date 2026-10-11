"""Synthetic controls and aggregate-only checks, without original flows."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts import official39_matched_neural as r
from src.eval import matched_neural as n
from src.eval.score_fusion import fit_weights
from tests.test_feature_expert import fixture

FEATURES = ['Number', 'Tot sum']+[f'f{i}' for i in range(37)]
SMALL = dict(n.SETTINGS, epochs=2, batch_size=64, report_epochs=[1, 2])


class MatchedNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.x, _, _, cls.y = fixture(n=401)
        cls.x = n.inputs(cls.x, FEATURES)
        cls.pop = np.bincount(cls.y)*np.arange(1, 9)
        with threadpool_limits(limits=2):
            cls.model, cls.fit = n.fit(cls.x, cls.y, cls.pop, 7, SMALL)

    def test_fixed_denominator_not_batch_weight_sum(self):
        logits = torch.tensor([[.2]*8, [.5, .1, .2, .3, .4, .5, .6, .7]], requires_grad=True)
        y, w = torch.tensor([0, 1]), torch.tensor([1., 9.], dtype=torch.float64)
        expected = (torch.nn.functional.cross_entropy(logits, y, reduction='none')*w).sum()/2
        loss = n.weighted_loss(logits, y, w)
        self.assertEqual(loss.item(), expected.item())
        self.assertNotEqual(loss.item(), (expected*2/w.sum()).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_same_class_objective_and_exact_exposure(self):
        s = self.fit
        w = fit_weights(self.y, self.pop)
        self.assertEqual(s['weighted_class_mass'], np.bincount(self.y, weights=w).tolist())
        np.testing.assert_allclose(np.asarray(s['weighted_class_mass'])/sum(w), np.sqrt(self.pop)/np.sqrt(self.pop).sum())
        self.assertEqual(s['sample_weights_sha256'], n.vector_hash(w.astype('<f8')))
        self.assertEqual(s['examples_processed'], 802)
        self.assertEqual(s['optimizer_steps'], 14)
        self.assertEqual(s['parameters'], 5096)
        self.assertEqual(s['parameter_bytes'], 20384)
        self.assertEqual([o['epoch'] for o in s['observations']], [1, 2])

    def test_replay_matches_weights_curves_predictions_and_serialization(self):
        with threadpool_limits(limits=2):
            model, record = n.fit(self.x, self.y, self.pop, 7, SMALL)
        self.assertEqual(record, self.fit)
        self.assertEqual(n.serialize(model), n.serialize(self.model))

    def test_fitting_keeps_supplied_arrays_unchanged(self):
        x, y, pop = self.x.copy(), self.y.copy(), self.pop.copy()
        with threadpool_limits(limits=2):
            n.fit(x, y, pop, 17, SMALL)
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_array_equal(y, self.y)
        np.testing.assert_array_equal(pop, self.pop)

    def test_restore_is_restricted_and_preserves_rng(self):
        raw = n.serialize(self.model)
        rng = torch.get_rng_state().clone()
        model = n.restore(raw, hashlib.sha256(raw).hexdigest(), self.fit['final_state_sha256'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        np.testing.assert_array_equal(n.predict(model, self.x), n.predict(self.model, self.x))
        with patch.object(torch, 'load', side_effect=AssertionError('Reject before loading')):
            with self.assertRaises(ValueError):
                n.restore(raw+b'bad', hashlib.sha256(raw).hexdigest(), self.fit['final_state_sha256'])

    def test_masked_inference_and_source_purity(self):
        x, _, _, _ = fixture(n=401)
        before = x.copy()
        first = n.predict(self.model, n.inputs(x, FEATURES))
        np.testing.assert_array_equal(x, before)
        x[:, :2] = 999
        np.testing.assert_array_equal(first, n.predict(self.model, n.inputs(x, FEATURES)))

    def test_reweighting_removes_only_class_prior_difference(self):
        cm = np.eye(8)*8+1
        different_supports = cm*np.arange(1, 9)[:, None]
        self.assertEqual(n.prior_standardized(cm, self.pop), n.prior_standardized(different_supports, self.pop))
        out = n.prior_standardized(cm, self.pop)
        self.assertEqual(out['class_prior'], (self.pop/self.pop.sum()).tolist())
        for c in n.CLASSES:
            self.assertAlmostEqual(out['per_class'][c]['recall'], 9/16)

    def test_reweighting_matches_independent_weighted_confusion(self):
        y = np.repeat(np.arange(8), np.arange(8)+3)
        pred = (np.arange(len(y)) % 8).astype(np.int64)
        cm = n.confusion(y, pred)
        weights = (self.pop/np.bincount(y))[y]
        weighted = np.bincount(y*8+pred, weights=weights, minlength=64).reshape(8, 8)
        expected = []
        out = n.prior_standardized(cm, self.pop)
        for i, c in enumerate(n.CLASSES):
            p = weighted[i, i]/weighted[:, i].sum()
            recall = weighted[i, i]/weighted[i].sum()
            f1 = 2*p*recall/(p+recall) if p+recall else 0
            expected.append(f1)
            self.assertAlmostEqual(out['per_class'][c]['f1'], f1)
        self.assertAlmostEqual(out['macro_f1'], np.mean(expected))

    def test_pair_counts_and_marginals(self):
        y = np.tile(np.arange(8), 4)
        tree, mlp = y.copy(), y.copy()
        tree[8:24] = (tree[8:24]+1) % 8
        mlp[16:] = (mlp[16:]+1) % 8
        pairs = n.paired_correctness(y, tree, mlp)
        for v in pairs.values():
            self.assertEqual(v, dict(support=4, both_correct=1, tree_only_correct=1,
                                    neural_only_correct=1, neither_correct=1, oracle_recall_upper_bound=.75))
        r.verify_pair(pairs, r.scored(y, tree, self.pop), r.scored(y, mlp, self.pop))

    def test_invalid_inputs_and_missing_class_stop(self):
        for matrix in [np.zeros((8, 8)), np.ones((7, 8)), np.eye(8)*np.nan, -np.eye(8)]:
            with self.assertRaises(ValueError):
                n.prior_standardized(matrix, self.pop)
        with self.assertRaises(ValueError):
            n.fit(self.x, self.y, self.pop, 7, dict(SMALL, epochs=61))
        with self.assertRaises(ValueError):
            n.predict(self.model, self.x.astype(np.float64))

    def test_historical_tree_inputs_must_match_before_fitting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(n, 'fit', side_effect=AssertionError('Do not fit')):
            with self.assertRaisesRegex(ValueError, 'fitting inputs differ'):
                r.fit_one(Path(tmp), self.x, self.y, np.arange(len(self.y)), self.pop, FEATURES, 7,
                          dict(fit_inputs_sha256='bad', original_fit_rows_sha256='bad'))

    def test_plan_and_output_guards(self):
        p = r.frozen.read(r.PLAN)
        r.validate_plan(p)
        for key in r.FLAGS:
            with self.assertRaises(ValueError):
                r.validate_plan(dict(p, **{key: True}))
        for key, value in [('seeds', [7]), ('settings', dict(n.SETTINGS, epochs=100)), ('arm', 'full39')]:
            with self.assertRaises(ValueError):
                r.validate_plan(dict(p, **{key: value}))
        with self.assertRaisesRegex(ValueError, 'Preserve'):
            r.run('none', 'none', 'none', 'none', Path(__file__).parent)
        with self.assertRaises(ValueError):
            r.summarize([])


class PublishedMatchedNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = r.ROOT/'reports/risk_aware_routing/matched-neural-results.json'
        if not p.exists():
            raise unittest.SkipTest('Experiment not published yet')
        cls.record = r.frozen.read(p)

    def test_public_evidence_without_private_inputs(self):
        with patch.object(np, 'load', side_effect=AssertionError('No arrays')), \
                patch.object(torch, 'load', side_effect=AssertionError('No models')):
            self.assertEqual(r.audit_record(self.record)['optimizer_steps'], 95040)
        self.assertTrue(self.record['non_timing_replay_exact'])

    def test_tampering_rejected(self):
        for key in r.FLAGS:
            with self.assertRaises(ValueError):
                r.audit_record(dict(self.record, **{key: True}))
        for key in ('metric', 'prior', 'budget', 'rows', 'paired', 'source', 'summary'):
            bad = copy.deepcopy(self.record)
            if key == 'metric':
                bad['cases'][0]['audit']['neural']['metrics']['macro_f1'] = 1.
            elif key == 'prior':
                bad['cases'][0]['fit']['neural']['prior_standardized']['macro_f1'] = 1.
            elif key == 'budget':
                bad['models'][0]['optimizer_steps'] += 1
            elif key == 'rows':
                bad['models'][0]['original_fit_rows_sha256'] = 'bad'
            elif key == 'paired':
                bad['cases'][0]['audit_paired_correctness'][n.CLASSES[0]]['both_correct'] += 1
            elif key == 'source':
                bad['identity']['source_sha256'] = {}
            else:
                bad['summary']['audit']['neural']['metrics']['macro_f1']['mean'] = 1.
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.audit_record(bad)


if __name__ == '__main__':
    unittest.main()

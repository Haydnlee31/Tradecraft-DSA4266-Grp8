"""Portable synthetic checks for exact projection screening and fresh fits."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import official39_feature_ablation as runner
from src.eval import feature_ablation as a
from tests.test_feature_expert import fixture

FEATURES = ['Number', 'Tot sum']+[f'f{i}' for i in range(37)]


class FeatureAblationTests(unittest.TestCase):
    def test_projection_is_pure_and_keeps_unmasked_values(self):
        x, _, _, _ = fixture()
        before = x.copy()
        for arm, drop in a.ARMS.items():
            z = a.project(x, FEATURES, arm)
            self.assertEqual(z.dtype, np.float64)
            self.assertEqual(z.shape, x.shape)
            for i, name in enumerate(FEATURES):
                np.testing.assert_array_equal(z[:, i], np.zeros(len(x)) if name in drop else x[:, i])
        np.testing.assert_array_equal(x, before)

    def test_names_not_positions_define_masks(self):
        x, _, _, _ = fixture()
        names = FEATURES[::-1]
        z = a.project(x, names, 'number_total_masked')
        self.assertTrue(np.all(z[:, -2:] == 0))
        np.testing.assert_array_equal(z[:, :-2], x[:, :-2])

    def test_schema_and_nonfinite_inputs_fail_even_if_masked(self):
        x, _, _, _ = fixture()
        for arm, names in [('other', FEATURES), ('full39', FEATURES[:-1]),
                           ('full39', ['same']*39)]:
            with self.assertRaises(ValueError):
                a.project(x, names, arm)
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            a.project(x, FEATURES, 'number_masked')

    def test_exact_membership_matches_independent_tuple_reference(self):
        x = np.random.default_rng(19).normal(size=(40, 39)).astype(np.float32)
        population, sampled, audit = np.arange(16), np.arange(8), np.arange(16, 32)
        x[16] = x[0]                       # Already a full-vector match.
        x[17] = x[1]
        x[17, :2] += 5                     # Only matches after joint removal.
        x[18] = x[9]
        x[18, 0] += 5                      # Match to UNSAMPLED fit row.
        x[2, 6] = -0.
        x[19] = x[2]
        x[19, 6] = +0.                     # Canonical signed zero must match.
        x[20] = x[1]
        x[20, 5] = np.nextafter(x[1, 5], np.float32(np.inf))  # Near is not exact.
        key = lambda row: tuple(float(v) for v in row[2:])
        fit_keys = {key(x[i]) for i in population}
        sample_keys = {key(x[i]) for i in sampled}
        expected = [i for i in audit if key(x[i]) in fit_keys]
        records = []
        with tempfile.TemporaryDirectory() as tmp:
            for batch in (1, 7, 8192):
                retained, excluded, report = a.collision_panel(x, FEATURES, population, sampled,
                    audit, Path(tmp)/f'{batch}.sqlite', batch)
                np.testing.assert_array_equal(excluded, expected)
                np.testing.assert_array_equal(retained, [i for i in audit if i not in expected])
                self.assertEqual(report['matches_to_sampled_fit_rows'], sum(key(x[i]) in sample_keys for i in audit))
                self.assertEqual(report['matches_only_to_unsampled_fit_rows'], 1)
                self.assertFalse(report['labels_used_for_membership'])
                records.append(report)
        self.assertEqual(records[0], records[1])
        self.assertEqual(records[1], records[2])

    def test_nested_projection_union_and_labels_cannot_select_rows(self):
        # collision_panel deliberately has no y argument. Matching removes a
        # feature-equivalent row regardless of its actual class label.
        x = np.zeros((6, 39), dtype=np.float32)
        x[1:, 3] = np.arange(1, 6)
        x[3] = x[0]
        x[4] = x[1]
        x[4, 0] = 8
        x[5] = x[2]
        x[5, :2] = 9
        expected = {'full39': [3], 'number_masked': [3, 4], 'number_total_masked': [3, 4, 5]}
        for arm in a.ARMS:
            z = a.project(x, FEATURES, arm)
            matches = [i for i in [3, 4, 5] if any(np.array_equal(z[i], z[j]) for j in [0, 1, 2])]
            self.assertEqual(matches, expected[arm])
        with tempfile.TemporaryDirectory() as tmp:
            keep, drop, _ = a.collision_panel(x, FEATURES, np.arange(3), np.arange(3),
                                             np.arange(3, 6), Path(tmp)/'index.sqlite')
        self.assertEqual(len(keep), 0)
        np.testing.assert_array_equal(drop, [3, 4, 5])

    def test_overlap_invalid_indices_and_existing_database_fail(self):
        x = np.zeros((10, 39), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'x.sqlite'
            for population, sample, audit in [(np.arange(3), np.array([9]), np.arange(4, 6)),
                                               (np.arange(3), np.arange(2), np.arange(2, 5)),
                                               (np.arange(3), np.arange(2), np.array([5, 5]))]:
                with self.assertRaises(ValueError):
                    a.collision_panel(x, FEATURES, population, sample, audit, path)
            path.touch()
            with self.assertRaisesRegex(ValueError, 'Preserve'):
                a.collision_panel(x, FEATURES, np.arange(3), np.arange(2), np.arange(4, 6), path)

    def test_fresh_full39_bridge_and_masked_fit_work(self):
        x, _, _, y = fixture()
        fit_rows, audit = np.arange(320), np.arange(320, 400)
        pop = np.bincount(y, minlength=8)
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=2):
            root = Path(tmp)
            old = runner.prior.fit_one(root, 'historical', 'features', 7, x, y, fit_rows, pop, None, None)
            full = runner.fit_one(root, x, y, fit_rows, pop, FEATURES, 'full39', 7, old)
            self.assertEqual(full['model_sha256'], old['model_sha256'])
            for arm in list(a.ARMS)[1:]:
                spec = runner.fit_one(root, x, y, fit_rows, pop, FEATURES, arm, 7, old)
                self.assertEqual(spec['trees'], 800)
                self.assertEqual(spec['masked_feature_splits'], 0)
                self.assertEqual(spec['weighted_class_mass'], full['weighted_class_mass'])
                model = runner.prior.load_owned(root, spec)
                with patch.object(runner.expert, 'fit', side_effect=AssertionError('Scoring cannot fit')):
                    pred = runner.predict_rows(model, x, audit, FEATURES, arm)
                self.assertEqual(pred.shape, (80,))
                changed = x.copy()
                changed[:, a.columns(FEATURES, arm)] = 999
                np.testing.assert_array_equal(pred, runner.predict_rows(model, changed, audit, FEATURES, arm))

    def test_old_model_mismatch_stops_bridge(self):
        x, _, _, y = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Historical full39'):
                runner.fit_one(Path(tmp), x, y, np.arange(len(y)), np.bincount(y)*2,
                               FEATURES, 'full39', 7, {'model_sha256': 'wrong'})

    def test_plan_rejects_new_inputs_cloud_and_changed_seeds(self):
        p = runner.frozen.read(runner.PLAN)
        runner.validate_plan(p)
        for k in runner.FLAGS:
            with self.assertRaises(ValueError):
                runner.validate_plan(dict(p, **{k: True}))
        with self.assertRaises(ValueError):
            runner.validate_plan(dict(p, seeds=[7]))

    def test_stop_when_any_class_disappears_from_panel(self):
        r = runner.frozen.read(runner.REFERENCE)
        old = r['split']
        classes = dict(old['partitions']['audit']['classes'])
        panel = dict(fit_population_rows=old['partitions']['fit']['rows'], fit_sample_rows=270326,
            original_audit_rows=408027, retained_rows=408027, excluded_rows=0,
            matches_to_sampled_fit_rows=0, matches_only_to_unsampled_fit_rows=0,
            joint_projection_unique_fit_keys=1, drop_features=a.ARMS['number_total_masked'],
            labels_used_for_membership=False, split_reassigned=False, retained_classes=classes,
            excluded_classes={c: 0 for c in classes}, files={'retained_rows.npy': 'a', 'excluded_rows.npy': 'b'})
        runner.verify_panel_record(panel, r)
        bad = copy.deepcopy(panel)
        c = runner.expert.CLASSES[4]
        n = bad['retained_classes'][c]
        bad['retained_classes'][c] = 0
        bad['excluded_classes'][c] = n
        bad['retained_rows'] -= n
        bad['excluded_rows'] += n
        bad['matches_to_sampled_fit_rows'] = n
        with self.assertRaises(ValueError):
            runner.verify_panel_record(bad, r)

    def test_existing_output_is_preserved(self):
        with self.assertRaisesRegex(ValueError, 'Preserve existing'):
            runner.run('absent', 'absent', 'absent', Path(__file__).parent)

    def test_all_seeds_required(self):
        with self.assertRaises(ValueError):
            runner.summarize([])


class PublishedFeatureAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = runner.ROOT/'reports/risk_aware_routing/feature-ablation-results.json'
        if not p.exists():
            raise unittest.SkipTest('Experiment not published yet')
        cls.record = runner.frozen.read(p)

    def test_aggregate_only_audit_and_exact_replay(self):
        with patch.object(np, 'load', side_effect=AssertionError('No private data')), \
                patch.object(runner.expert.joblib, 'load', side_effect=AssertionError('No unpickling')):
            self.assertEqual(runner.audit_record(self.record)['fits'], 9)
        self.assertTrue(self.record['non_timing_replay_exact'])

    def test_tampering_fails(self):
        for key in runner.FLAGS:
            with self.assertRaises(ValueError):
                runner.audit_record(dict(self.record, **{key: True}))
        for key in ['metric', 'mask', 'rows', 'panel', 'budget', 'sources', 'summary']:
            bad = copy.deepcopy(self.record)
            if key == 'metric':
                bad['cases'][0]['primary']['number_masked']['metrics']['macro_f1'] = 1
            elif key == 'mask':
                bad['models'][1]['dropped_features'] = []
            elif key == 'rows':
                bad['models'][1]['original_fit_rows_sha256'] = 'wrong'
            elif key == 'panel':
                bad['common_panel']['retained_rows'] -= 1
            elif key == 'budget':
                bad['models'][1]['trees'] += 1
            elif key == 'sources':
                bad['identity']['source_sha256'] = {}
            else:
                bad['summary']['primary']['methods']['full39']['macro_f1']['mean'] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.audit_record(bad)


if __name__ == '__main__':
    unittest.main()

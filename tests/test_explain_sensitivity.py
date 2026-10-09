"""Synthetic, bounded checks; real training files are never required in CI."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.explain import official_sensitivity as sensitivity
from src.explain import official, official_convergence as mc, official_integration as ig, official_study as study
from tests.test_explain_study import ideal_fixture_integral

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/full_data_extension/explanation-sensitivity-plan.json'
FEATURES = json.loads((PLAN.parent/'explanation-cpu-plan.json').read_text())['features']


class SensitivityTests(unittest.TestCase):
    def test_scope_budget_parent_and_group_guards(self):
        plan = json.loads(PLAN.read_text())
        sensitivity.validate_plan(plan)
        self.assertEqual(plan['study_plan_sha256'], sha256(PLAN.parent/'explanation-study-plan.json'))
        self.assertEqual(plan['study_receipt_sha256'], json.loads((PLAN.parent/'explanation-study-results.json').read_text())['source_receipt_sha256'])
        for k, value in [('settings', {}), ('feature_groups', {}), ('model_trained', True), ('work_budget', {})]:
            bad = copy.deepcopy(plan)
            bad[k] = value
            with self.assertRaises(ValueError):
                sensitivity.validate_plan(bad)

    def test_sampler_unique_disjoint_deterministic_and_global_rng_unchanged(self):
        before = np.random.get_state()
        a = sensitivity.select_background(300, list(range(128)))
        b = sensitivity.select_background(300, list(range(128)))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 128)
        self.assertEqual(len(set(a)), 128)
        self.assertFalse(set(a) & set(range(128)))
        self.assertTrue((np.diff(a) > 0).all())
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        with self.assertRaises(ValueError):
            sensitivity.select_background(200, range(128))

    def test_training_scan_bins_identities_correlations_and_constants(self):
        x = np.random.default_rng(7).normal(size=(24, 39)).astype(np.float32)
        y = np.tile(np.arange(8, dtype=np.int64), 3)
        j = FEATURES.index
        x[:, j('Number')] = np.repeat([10., 100., 7.5], 8)
        x[:, j('IPv')] = -x[:, j('ARP')]
        x[:, j('LLC')] = x[:, j('IPv')]
        x[:, j('Tot size')] = x[:, j('AVG')]
        x[:, j('SMTP')] = 3.125
        settings = dict(json.loads(PLAN.read_text())['settings'], scan_batch_size=7)
        scaler = dict(scale=[1.]*39, mean=[0.]*39)
        r = sensitivity.scan_training(x, y, scaler, FEATURES, settings)
        self.assertEqual(r['class_counts'], dict.fromkeys(CLASSES, 3))
        for v in r['number_by_class'].values():
            self.assertEqual([v['near_10'], v['near_100'], v['other']], [1, 1, 1])
        self.assertTrue(all(v['passing_rows'] == 24 for v in r['identities']))
        self.assertIn('SMTP', r['constant_features'])
        self.assertTrue(all(v['left'] != 'SMTP' and v['right'] != 'SMTP' for v in r['high_correlations']))
        r2 = sensitivity.scan_training(x, y, scaler, FEATURES, dict(settings, scan_batch_size=24))
        self.assertEqual(r['number_by_class'], r2['number_by_class'])
        np.testing.assert_allclose([v['pearson_r'] for v in r['high_correlations']],
                                   [v['pearson_r'] for v in r2['high_correlations']], atol=1e-12)
        x[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid training values'):
            sensitivity.scan_training(x, y, scaler, FEATURES, settings)

    def test_reader_opens_only_training_and_closes_windows_handles(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root/'train_x.npy', np.zeros((8, 39), np.float32))
            np.save(root/'train_y.npy', np.arange(8, dtype=np.int64))
            manifest = dict(train_rows=8, files={p.name: sha256(p) for p in root.glob('*.npy')})
            original, opened = np.load, []
            def guarded(path, *args, **kwargs):
                opened.append(Path(path).name)
                self.assertTrue(Path(path).name.startswith('train_'))
                return original(path, *args, **kwargs)
            with patch.object(np, 'load', side_effect=guarded):
                with sensitivity.training_arrays(root, manifest) as (x, y):
                    self.assertEqual(x.shape, (8, 39))
                self.assertTrue(x._mmap.closed and y._mmap.closed)
            self.assertEqual(opened, ['train_x.npy', 'train_y.npy'])
            (root/'train_y.npy').write_bytes(b'corrupt')
            with patch.object(np, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    with sensitivity.training_arrays(root, manifest):
                        self.fail('Should not open any arrays')
                load.assert_not_called()

    def test_grouped_signed_sum_exposes_cancellation_not_sum_of_importances(self):
        values = np.zeros((2, 39, 8))
        values[:, FEATURES.index('ARP')] = 10
        values[:, FEATURES.index('IPv')] = -9
        r = sensitivity.grouped(values, FEATURES)
        self.assertEqual(r['link_indicators']['mean_gross_magnitude'], 19)
        self.assertEqual(r['link_indicators']['mean_net_magnitude'], 1)
        self.assertAlmostEqual(r['link_indicators']['magnitude_cancellation_fraction'], 18/19)
        self.assertIsNone(r['size_summaries']['magnitude_cancellation_fraction'])

    def test_reference_comparison_keeps_fixed_predictions_and_has_no_agreement_gate(self):
        old = np.ones((8, 39, 8))
        new = old.copy()
        new[:, FEATURES.index('Number')] = -2
        logits = np.eye(8)*2
        logits[0, 1] = 3
        r = sensitivity.compare_backgrounds(old, new, logits, np.zeros(8), np.arange(8),
                                            np.arange(8), FEATURES, 1e-6)
        self.assertIsNone(r['score_feature_difference']['resolution_screen_passed'])
        self.assertTrue(all(c['sign_changed'] for c in r['number_cases']))
        self.assertEqual(len(r['error_cases']), 1)
        self.assertEqual(r['error_cases'][0]['unchanged_logit_margin'], 1)
        self.assertEqual(r['error_cases'][0]['new_background_margin'], 1)

    def test_changed_study_fails_before_any_model_or_training_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'receipt.json').write_text('{}')
            kwargs = dict.fromkeys(['preparation_root', 'convergence_root', 'integration_root',
                'study_root', 'archive_root', 'packed_root'], root)
            kwargs.update(preparation_plan_path=PLAN, convergence_plan_path=PLAN, integration_plan_path=PLAN,
                study_plan_path=PLAN.parent/'explanation-study-plan.json', plan_path=PLAN, output=root/'out')
            with patch.object(sensitivity.study, 'load_evidence') as loader:
                with self.assertRaisesRegex(ValueError, 'Study evidence changed'):
                    sensitivity.run(**kwargs)
                loader.assert_not_called()
            self.assertFalse((root/'out').exists())

    def test_runner_end_to_end_with_small_synthetic_training_and_exact_publication(self):
        # Only this fixture substitutes the 2M scope guard. The production
        # guard is tested separately and all actual runner/file paths execute.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packed, old_study = root/'packed', root/'old-study'
            packed.mkdir()
            old_study.mkdir()
            rng = np.random.default_rng(9)
            train_x = rng.normal(size=(256, 39)).astype(np.float32)
            train_y = np.tile(np.arange(8, dtype=np.int64), 32)
            for name, a in [('train_x', train_x), ('train_y', train_y)]:
                np.save(packed/(name+'.npy'), a)
            scaler = dict(features=FEATURES, fit_split='train', n_samples_seen=256, scale=[1.]*39, mean=[0.]*39)
            (packed/'scaler.json').write_text(json.dumps(scaler))
            manifest = dict(train_rows=256, train_class_counts=dict.fromkeys(CLASSES, 32),
                scaler_sha256=sha256(packed/'scaler.json'),
                files={p.name: sha256(p) for p in packed.glob('*.npy')})
            (packed/'manifest.json').write_text(json.dumps(manifest))
            inputs = dict(background=train_x[:128].copy(), pilot_x=train_x[128:136].copy(),
                          y=np.arange(219) % 8)
            sampling = dict(validation_rows=list(range(219)), pilot_validation_rows=list(range(8)),
                background_train_rows=list(range(128)), background_class_counts=dict.fromkeys(CLASSES, 16))
            models, pilots, records = {}, {}, {}
            for lane, seed in official.ROSTER:
                key = f'{lane}-seed{seed}'
                torch.manual_seed(seed)
                model = torch.nn.Linear(39, 8).eval()
                values = ideal_fixture_integral(model, inputs['background'], inputs['pilot_x'], 2)
                with torch.no_grad():
                    logits = model(torch.from_numpy(inputs['pilot_x'])).numpy()
                    reference = model(torch.from_numpy(inputs['background'])).numpy().mean(0)
                models[key] = model
                pilots[key] = dict(coarse=values, fine=values, logits=logits, reference_logits=reference)
                full = np.tile(values, (28, 1, 1))[:219]
                np.savez_compressed(old_study/f'{key}-study.npz', coarse=full, fine=full)
                records[key] = dict(arrays_sha256=sha256(old_study/f'{key}-study.npz'), checkpoint_sha256='fixture')
            sources = {'explain/'+Path(m.__file__).name: sha256(m.__file__) for m in (official, mc, ig, study)}
            old_plan_path = root/'old-plan.json'
            old_plan_path.write_text('{}')
            saved = dict(status='complete', all_numerical_screens_passed=True, models=records,
                         source_sha256=sources, plan_sha256=sha256(old_plan_path))
            (old_study/'receipt.json').write_text(json.dumps(saved))
            plan = json.loads(PLAN.read_text())
            plan['settings']['training_rows'] = 256
            plan.update(study_plan_sha256=sha256(old_plan_path), study_receipt_sha256=sha256(old_study/'receipt.json'))
            new_plan_path = root/'plan.json'
            new_plan_path.write_text(json.dumps(plan))
            prior = dict(inputs_sha256='fixture', sampling_sha256='fixture', models=records)
            parent = json.loads((PLAN.parent/'explanation-integration-plan.json').read_text())
            evidence = (None, parent, dict(features=FEATURES), prior, inputs, sampling, models, pilots, None, {})
            args = dict(preparation_root=root, convergence_root=root, integration_root=root,
                        study_root=old_study, archive_root=root, packed_root=packed,
                        preparation_plan_path=PLAN, convergence_plan_path=PLAN, integration_plan_path=PLAN,
                        study_plan_path=old_plan_path, plan_path=new_plan_path, output=root/'out')
            with patch.object(sensitivity, 'validate_plan'), \
                 patch.object(study, 'load_evidence', return_value=evidence), \
                 patch.object(ig, 'integrate', side_effect=ideal_fixture_integral) as integration:
                result = sensitivity.run(**args)
                self.assertEqual(integration.call_count, 14)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['new_background_counts'], dict.fromkeys(CLASSES, 16))
                self.assertTrue(all(result[k] is False for k in mc.FLAGS))
                args['output'] = root/'replay'
                sensitivity.run(**args)
            publication = sensitivity.publish_verified(root/'out', root/'replay', root/'publication.json')
            self.assertTrue(publication['audit']['independent_receipt_replay_exact'])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                sensitivity.publish_verified(root/'out', root/'replay', root/'publication.json')
            (root/'replay'/'dirichlet-seed7-sensitivity.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'array replay changed'):
                sensitivity.publish_verified(root/'out', root/'replay', root/'bad-publication.json')
            self.assertFalse((root/'bad-publication.json').exists())

    def test_published_evidence_counts_hashes_and_no_training(self):
        r = json.loads((PLAN.parent/'explanation-sensitivity-results.json').read_text())
        self.assertEqual(r['plan_sha256'], sha256(PLAN))
        self.assertEqual(r['status'], 'complete')
        self.assertTrue(r['audit']['independent_receipt_replay_exact'])
        self.assertTrue(all(r[k] is False for k in mc.FLAGS))
        self.assertEqual(sum(r['new_background_counts'].values()), 128)
        self.assertEqual(len(set(r['background_rows'])), 128)
        audit = r['training_audit']
        self.assertEqual(audit['rows'], 2000000)
        self.assertEqual(sum(audit['class_counts'].values()), 2000000)
        for c, a in audit['number_by_class'].items():
            self.assertEqual(a['near_10']+a['near_100']+a['other'], audit['class_counts'][c])
        self.assertTrue(all(a['passing_rows'] == 2000000 for a in audit['identities']))
        for resolution in ('coarse', 'fine'):
            self.assertEqual(sum(m['numerical_qa']['scores'][resolution]['passing_output_count'] for m in r['models'].values()), 448)
            self.assertEqual(sum(m['numerical_qa']['error_margins'][resolution]['passing_output_count'] for m in r['models'].values()), 29)
        import hashlib
        raw = Path(sensitivity.__file__).read_bytes()
        self.assertIn(r['source_sha256']['explain/official_sensitivity.py'],
                      [hashlib.sha256(b).hexdigest() for b in (raw, raw.replace(b'\r\n', b'\n'))])


if __name__ == '__main__':
    unittest.main()

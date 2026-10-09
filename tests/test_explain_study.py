"""Small synthetic study fixtures: no cloud artifacts or traffic data required."""
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
from src.explain import official_study as study, official_integration as ig
from src.explain import official, official_convergence as mc, official_study_report as report
from tests.test_explain_integration import integration_fixture

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/full_data_extension/explanation-study-plan.json'
PARENT = PLAN.parent/'explanation-integration-plan.json'
FEATURES = [f'f{i}' for i in range(39)]


def example(n=64):
    labels = np.arange(n) % 8
    logits = np.eye(8)[labels]*2
    logits[0, 1] = 3
    reference = np.arange(8)*.1
    values = np.broadcast_to((logits-reference)[:, None, :]/39, (n, 39, 8)).copy()
    return values, logits, reference, labels


def ideal_fixture_integral(model, background, x, segments):
    """Synthetic completeness fixture, NOT an attribution algorithm."""
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).numpy()
        ref = model(torch.from_numpy(background)).numpy().mean(0)
    return np.broadcast_to((logits-ref)[:, None, :]/39, (len(x), 39, 8)).astype(np.float64).copy()


class StudyTests(unittest.TestCase):
    def test_plan_parent_and_budget_are_frozen(self):
        plan, parent = map(lambda p: json.loads(p.read_text()), (PLAN, PARENT))
        study.validate_plan(plan, parent)
        self.assertEqual(plan['integration_plan_sha256'], sha256(PARENT))
        published = json.loads((PLAN.parent/'explanation-integration-results.json').read_text())
        self.assertEqual(plan['integration_receipt_sha256'], published['source_receipt_sha256'])
        for field, value in [('settings', {}), ('work_budget', {}), ('test_evaluated', True),
                             ('full_attribution_run', False), ('feature_interpretation_authorized', True)]:
            bad = copy.deepcopy(plan)
            bad[field] = value
            with self.assertRaises(ValueError):
                study.validate_plan(bad, parent)

    def test_stratum_counts_signed_margin_and_no_correct_self_contrasts(self):
        v, logits, ref, labels = example()
        result = study.describe_stratum(v, v, logits, ref, labels, list(range(64)), FEATURES,
                                       json.loads(PARENT.read_text()))
        self.assertTrue(result['passed'])
        self.assertEqual(result['error_count'], 1)
        self.assertEqual(result['error_validation_rows'], [0])
        self.assertEqual(result['scores']['fine']['passing_output_count'], 512)
        self.assertEqual(result['descriptive']['by_true_class']['Benign']['count'], 8)
        margin = result['descriptive']['error_pairs']['Benign -> DDoS']
        self.assertAlmostEqual(sum(margin['mean_signed']), .9)
        no_errors = study.describe_stratum(v, v, logits, ref, logits.argmax(1), list(range(64)), FEATURES,
                                          json.loads(PARENT.read_text()))
        self.assertIsNone(no_errors['error_margins'])
        self.assertEqual(no_errors['error_count'], 0)

    def test_bad_case_withholds_whole_stratum_without_deleting_it(self):
        v, logits, ref, labels = example()
        coarse = v.copy()
        coarse[17, 0, 3] += 10
        coarse[17, 1, 3] -= 10  # Sum cancels, individual feature errors do not.
        result = study.describe_stratum(coarse, v, logits, ref, labels, list(range(64)), FEATURES,
                                       json.loads(PARENT.read_text()))
        self.assertFalse(result['passed'])
        self.assertIsNone(result['descriptive'])
        self.assertEqual(len(result['cases']), 64)
        self.assertEqual([c['validation_row'] for c in result['cases'] if not c['individual_checks_passed']], [17])
        self.assertEqual(result['scores']['fine']['passing_output_count'], 512)

    def test_margin_can_fail_when_class_scores_pass(self):
        v, logits, ref, labels = example()
        coarse, fine = v.copy(), v.copy()
        # Both class residuals remain below their own threshold. Their
        # difference exceeds the margin threshold and must not be hidden.
        coarse[0, 0, 1] += .25
        coarse[0, 0, 0] -= .25
        fine[:] = coarse
        result = study.describe_stratum(coarse, fine, logits, ref, labels, list(range(64)), FEATURES,
                                       json.loads(PARENT.read_text()))
        self.assertTrue(result['scores']['passed'])
        self.assertFalse(result['error_margins']['passed'])
        self.assertFalse(result['passed'])

    def test_seed_normalization_stability_and_no_cherry_picking(self):
        v, logits, ref, labels = example()
        core = study.describe_stratum(v, v, logits, ref, labels, list(range(64)), FEATURES,
                                     json.loads(PARENT.read_text()))
        models = {f'{lane}-seed{s}': dict(core=copy.deepcopy(core))
                  for lane, seeds in [('light', [7, 17, 27]), ('iid', [7, 17, 27]), ('dirichlet', [7])]
                  for s in seeds}
        result = study.seed_summary(models, FEATURES)
        target = result['light']['targets']['balanced_core']
        self.assertEqual(target['pairwise_top_five_jaccard'], [1.]*3)
        self.assertAlmostEqual(sum(target['normalized_mean']), 1.)
        self.assertIsNone(result['dirichlet']['targets']['balanced_core']['normalized_sample_sd'])
        models['light-seed17']['core']['passed'] = False
        self.assertEqual(study.seed_summary(models, FEATURES)['light']['status'], 'withheld_numerical_failure')

    def test_runner_bounds_chunks_replays_pilot_and_keeps_strata(self):
        plan, parent = map(lambda p: json.loads(p.read_text()), (PLAN, PARENT))
        v, logits, ref, labels = example(219)
        # The first input feature carries its row position for this fixture.
        x = np.zeros((219, 39), np.float32)
        x[:, 0] = np.arange(219)
        keys = [f'{lane}-seed{s}' for lane, seeds in [('light', [7, 17, 27]), ('iid', [7, 17, 27]), ('dirichlet', [7])]
                for s in seeds]
        prior = dict(inputs_sha256='fixture', sampling_sha256='fixture', cpu_reference_sha256='fixture',
                     source_sha256={}, models={k: dict(checkpoint_sha256='fixture') for k in keys})
        inputs = dict(x=x, y=labels, background=np.zeros((128, 39), np.float32))
        sampling = dict(validation_rows=list(range(219)), pilot_validation_rows=list(range(8)),
                        core_rows=list(range(64)))
        models = dict.fromkeys(keys, object())
        pilots = {k: dict(coarse=v[:8], fine=v[:8]) for k in keys}
        endpoints = dict.fromkeys(keys, (logits, ref))
        evidence = (plan, parent, dict(features=FEATURES), prior, inputs, sampling, models, pilots, endpoints, {})
        def integrate(model, bg, selected, segments):
            self.assertLessEqual(len(selected), 8)
            self.assertIn(segments, (2, 4))
            return v[selected[:, 0].astype(int)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez_compressed(root/'inputs.npz', **inputs, validation_rows=np.arange(219))
            (root/'sampling.json').write_text(json.dumps(sampling))
            prior.update(inputs_sha256=sha256(root/'inputs.npz'), sampling_sha256=sha256(root/'sampling.json'),
                         source_sha256={'explain/'+Path(m.__file__).name: sha256(m.__file__) for m in (official, mc, ig)})
            args = dict(preparation_root=folder, convergence_root=folder, integration_root=folder,
                        archive_root=folder, manifest_path=PLAN, preparation_plan_path=PLAN,
                        convergence_plan_path=PLAN, integration_plan_path=PARENT, plan_path=PLAN,
                        output=Path(folder)/'study')
            with patch.object(study, 'load_evidence', return_value=evidence), \
                 patch.object(ig, 'integrate', side_effect=integrate) as call:
                result = study.run(**args)
            self.assertEqual(call.call_count, 7*2*28)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['full_attribution_run'])
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(result['models']['light-seed7']['core']['count'], 64)
            self.assertEqual(result['models']['light-seed7']['supplement']['count'], 155)
            publication = dict(study_root=args['output'], preparation_root=root, plan_path=PLAN,
                               integration_plan_path=PARENT, output=root/'published.json')
            published = report.publish(**publication)
            self.assertTrue(published['audit']['all_diagnostics_recomputed'])
            self.assertEqual(published['models']['light-seed7']['core']['scores']['individual_pass_count'], 512)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                report.publish(**publication)
            publication['output'] = root/'corrupt.json'
            last = args['output']/'dirichlet-seed7-study.npz'
            last.write_bytes(b'corrupt attribution evidence')
            with self.assertRaisesRegex(ValueError, 'Study arrays changed'):
                report.publish(**publication)
            self.assertFalse(publication['output'].exists())
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                study.run(**args)
            args['output'] = Path(folder)/'changed'
            with patch.object(study, 'load_evidence', return_value=evidence), \
                 patch.object(ig, 'integrate', side_effect=lambda m, b, s, n: integrate(m, b, s, n)+1):
                with self.assertRaisesRegex(ValueError, 'pilot attribution replay'):
                    study.run(**args)

    def test_tampered_parent_fails_before_new_calculation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = integration_fixture(root)
            with patch.object(ig, 'integrate', side_effect=ideal_fixture_integral):
                ig.run(**args)
            plan = json.loads(PLAN.read_text())
            plan.update(integration_plan_sha256=sha256(args['plan_path']),
                        integration_receipt_sha256=sha256(args['output']/'receipt.json'))
            path = root/'study-plan.json'
            path.write_text(json.dumps(plan))
            new_args = dict(args, integration_plan_path=args['plan_path'], plan_path=path,
                            integration_root=args['output'], output=root/'study')
            (args['output']/'receipt.json').write_text('{}')
            with patch.object(ig, 'integrate') as call:
                with self.assertRaisesRegex(ValueError, 'Evidence checksum changed'):
                    study.run(**new_args)
                call.assert_not_called()
            self.assertFalse(new_args['output'].exists())

    def test_publication_preserves_failure_identity_and_relative_global_failure(self):
        v, logits, ref, labels = example()
        coarse = v.copy()
        coarse[0, 0, 1] += 10
        coarse[0, 1, 1] -= 10
        checked = study.screen(coarse, v, logits-ref, json.loads(PARENT.read_text()))
        compact = report.compact_screen(checked, list(range(100, 164)), CLASSES)
        self.assertEqual(compact['individual_pass_count'], 511)
        self.assertEqual(compact['failed_outputs'][0]['validation_row'], 100)
        self.assertEqual(compact['failed_outputs'][0]['output'], 'DDoS')
        self.assertFalse(compact['passed'])
        # Individual residual limits can pass while the global relative limit
        # fails; the compact receipt must retain that failure too.
        values = np.zeros((2, 39))
        checked = study.screen(values, values, np.ones(2)*.1, json.loads(PARENT.read_text()))
        compact = report.compact_screen(checked, [7, 8], [])
        self.assertEqual(compact['individual_pass_count'], 2)
        self.assertFalse(compact['passed'])
        self.assertEqual(compact['fine']['relative_mean_abs_residual'], 1.)

    def test_published_study_counts_scope_and_provenance(self):
        result = json.loads((PLAN.parent/'explanation-study-results.json').read_text())
        self.assertEqual(result['plan_sha256'], sha256(PLAN))
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['all_numerical_screens_passed'])
        self.assertTrue(result['full_attribution_run'])
        for flag in ('test_evaluated', 'model_trained', 'model_promoted', 'deployment_authorized',
                     'feature_interpretation_authorized', 'full_validation_rerun'):
            self.assertFalse(result[flag])
        groups = [m[s] for m in result['models'].values() for s in ('core', 'supplement')]
        self.assertEqual(sum(g['count'] for g in groups), 7*219)
        self.assertEqual(sum(g['error_count'] for g in groups), 835)
        for resolution in ('coarse', 'fine'):
            self.assertEqual(sum(g['scores'][resolution]['passing_output_count'] for g in groups), 12264)
            self.assertEqual(sum(g['error_margins'][resolution]['passing_output_count'] for g in groups), 835)
        for group in groups:
            self.assertTrue(group['passed'])
            self.assertEqual(group['scores']['failed_outputs'], [])
            self.assertEqual(group['error_margins']['failed_outputs'], [])
            self.assertEqual(group['individual_passing_cases'], group['count'])
        for module in (study, report):
            raw = Path(module.__file__).read_bytes()
            import hashlib
            expected = result['publisher_sha256'] if module is report else result['source_sha256']['explain/official_study.py']
            self.assertIn(expected, [hashlib.sha256(v).hexdigest() for v in (raw, raw.replace(b'\r\n', b'\n'))])


if __name__ == '__main__':
    unittest.main()

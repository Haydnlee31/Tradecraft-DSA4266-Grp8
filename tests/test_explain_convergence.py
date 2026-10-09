"""Convergence control uses small synthetic receipts, never downloaded traffic."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.official_inventory import sha256
from src.explain import official
from src.explain.official_convergence import array_quality, compare_arrays, contrast_quality, run, validate_plan
from tests.test_explain_official import fixture

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/full_data_extension/explanation-convergence-plan.json'


def synthetic_pilot(model, background, x, settings, limits, complete=False):
    """A numerical fixture, explicitly NOT measured SHAP attributions."""
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).numpy()
        reference = model(torch.from_numpy(background)).numpy().mean(0)
    values = np.zeros((8, 39, 8), dtype=np.float64)
    if complete:
        values[:] = (logits-reference)[:, None, :]/39.
    arrays = dict(repeat_0=values, repeat_1=values.copy(), logits=logits, reference_logits=reference)
    return array_quality(arrays, limits), arrays


def setup_fixture(root):
    previous, _, _ = fixture(root)
    cpu_ref = root/'cpu.json'
    official.audit_bridges(root/'packed', root, root/'plan.json', cpu_ref)
    previous.update(protocol='official39-explanation-cpu-reference-v2',
                    superseded_plan_sha256=sha256(root/'plan.json'), cpu_reference_sha256=sha256(cpu_ref))
    previous['settings']['confusion_parity'] = 'frozen_cpu_reference'
    previous_path = root/'cpu-plan.json'
    previous_path.write_text(json.dumps(previous))
    with patch('src.explain.official.pilot', side_effect=synthetic_pilot):
        official.prepare(root/'packed', root, previous_path, root/'prepared', cpu_ref)
    plan = json.loads(PLAN.read_text())
    plan.update(preparation_plan_sha256=sha256(previous_path),
                preparation_receipt_sha256=sha256(root/'prepared/receipt.json'))
    (root/'convergence-plan.json').write_text(json.dumps(plan))
    return dict(preparation_root=root/'prepared', archive_root=root,
                manifest_path=root/'packed/manifest.json', preparation_plan_path=previous_path,
                plan_path=root/'convergence-plan.json', output=root/'out')


class ConvergenceTests(unittest.TestCase):
    def test_plan_rejects_budget_seed_and_threshold_drift(self):
        plan = json.loads(PLAN.read_text())
        previous = json.loads((PLAN.parent/'explanation-cpu-plan.json').read_text())
        validate_plan(plan, previous)
        for key, value in [('comparison_nsamples', 8192), ('mc_seeds', [1, 2]), ('pilot_rows', 64),
                           ('quality_limits', {}), ('feature_interpretation_authorized', True)]:
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_plan(bad, previous)

    def test_contrast_identity_excludes_correct_and_keeps_failures(self):
        limits = json.loads(PLAN.read_text())['quality_limits']
        labels = np.arange(8)
        logits = np.eye(8)*2
        logits[0, 1], logits[3, 5] = 5, 6  # Exactly two errors.
        reference = np.arange(8)*.1
        phi = np.broadcast_to((logits-reference)[:, None, :]/39, (8, 39, 8)).copy()
        arrays = dict(repeat_0=phi, repeat_1=phi.copy(), logits=logits, reference_logits=reference)
        qa = contrast_quality(arrays, labels, limits)
        self.assertEqual(qa['error_case_count'], 2)
        self.assertEqual(qa['passing_case_count'], 2)
        self.assertTrue(qa['diagnostic_screen_passed'])
        self.assertAlmostEqual(qa['cases'][0]['centered_margin'], 2.9)
        self.assertEqual([c['pilot_position'] for c in qa['cases']], [0, 3])
        broken = copy.deepcopy(arrays)
        broken['repeat_0'][0, :, 1] += 1
        broken['repeat_1'][0, :, 1] += 1
        result = compare_arrays(arrays, broken, labels, limits)
        self.assertFalse(result['comparison']['numerical_screen_passed'])
        self.assertFalse(result['comparison_contrasts']['diagnostic_screen_passed'])
        self.assertEqual(result['output_transitions']['passed_to_fail'], 1)
        self.assertEqual(result['absolute_residual_worsened_outputs'], 1)
        no_errors = dict(arrays, logits=np.eye(8)*2)
        self.assertIsNone(contrast_quality(no_errors, labels, limits)['diagnostic_screen_passed'])
        broken['logits'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'logits changed'):
            compare_arrays(arrays, broken, labels, limits)

    def test_real_loader_fixed_inputs_and_budget_without_opening_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            args = setup_fixture(Path(folder))
            opened = []
            real_load = np.load
            def guarded_load(path, *a, **kw):
                opened.append(Path(path))
                self.assertEqual(Path(path).suffix, '.npz')
                return real_load(path, *a, **kw)
            def candidate(model, background, x, settings, limits):
                self.assertEqual(settings['pilot_nsamples'], 2048)
                self.assertEqual(settings['mc_seeds'], [71, 72])
                return synthetic_pilot(model, background, x, settings, limits, complete=True)
            with patch('src.explain.official.pilot', side_effect=candidate) as pilot, \
                 patch('src.explain.official_convergence.np.load', side_effect=guarded_load):
                result = run(**args)
                self.assertEqual(pilot.call_count, 7)
            self.assertEqual(len(opened), 8)  # Input plus seven baseline files.
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['all_comparison_screens_passed'])
            for flag in ('test_evaluated', 'model_trained', 'feature_interpretation_authorized',
                         'full_attribution_run', 'full_validation_rerun'):
                self.assertFalse(result[flag])
            self.assertEqual(len(result['models']), 7)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                run(**args)

    def test_every_baseline_checked_before_first_attribution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = setup_fixture(root)
            (root/'prepared/dirichlet-seed7-pilot.npz').write_bytes(b'corrupt last model')
            with patch('src.explain.official.pilot') as pilot:
                with self.assertRaisesRegex(ValueError, 'attribution checksum'):
                    run(**args)
                pilot.assert_not_called()
            self.assertFalse(args['output'].exists())

    def test_changed_inputs_fail_before_loading_models(self):
        with tempfile.TemporaryDirectory() as folder:
            args = setup_fixture(Path(folder))
            (args['preparation_root']/'inputs.npz').write_bytes(b'changed input')
            with patch('src.explain.official.load_frozen') as loader, \
                 patch('src.explain.official.pilot') as pilot:
                with self.assertRaisesRegex(ValueError, 'inputs or sampling changed'):
                    run(**args)
                loader.assert_not_called()
                pilot.assert_not_called()

    def test_numerical_warning_remains_and_source_drift_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            args = setup_fixture(Path(folder))
            with patch('src.explain.official.pilot', side_effect=synthetic_pilot):
                result = run(**args)
            self.assertEqual(result['status'], 'numerical_review_required')
            self.assertFalse(result['feature_interpretation_authorized'])
            self.assertFalse(result['all_comparison_screens_passed'])
            # Re-pin a fixture receipt with the wrong source identity: it must
            # still fail on the source check, without any new SHAP calculation.
            receipt_path = args['preparation_root']/'receipt.json'
            receipt = json.loads(receipt_path.read_text())
            receipt['source_sha256']['explain/official.py'] = 'changed'
            receipt_path.write_text(json.dumps(receipt))
            plan = json.loads(args['plan_path'].read_text())
            plan['preparation_receipt_sha256'] = sha256(receipt_path)
            args['plan_path'].write_text(json.dumps(plan))
            args['output'] = Path(folder)/'bad'
            with patch('src.explain.official.pilot') as pilot:
                with self.assertRaisesRegex(ValueError, 'implementation changed'):
                    run(**args)
                pilot.assert_not_called()

    def test_published_plan_pins_original_experiment(self):
        plan = json.loads(PLAN.read_text())
        previous = PLAN.parent/'explanation-cpu-plan.json'
        result = json.loads((PLAN.parent/'explanation-preparation-results.json').read_text())
        self.assertEqual(plan['preparation_plan_sha256'], sha256(previous))
        self.assertEqual(plan['preparation_receipt_sha256'], result['source_receipt_sha256'])
        self.assertEqual(plan['quality_limits'], result['quality_limits'])
        self.assertFalse(plan['full_attribution_run'])

    def test_published_comparison_counts_and_stopping_status(self):
        result = json.loads((PLAN.parent/'explanation-convergence-results.json').read_text())
        previous = json.loads((PLAN.parent/'explanation-cpu-plan.json').read_text())
        self.assertEqual(result['plan_sha256'], sha256(PLAN))
        self.assertEqual(set(result['models']), {m['id'] for m in previous['models']})
        models = list(result['models'].values())
        summary = result['summary']
        self.assertEqual(summary['baseline_passing_outputs'], sum(m['baseline']['passing_output_count'] for m in models))
        self.assertEqual(summary['comparison_passing_outputs'], sum(m['comparison']['passing_output_count'] for m in models))
        self.assertEqual(summary['comparison_passing_outputs'], 443)
        self.assertEqual(summary['total_error_margins'], sum(m['comparison_contrasts']['error_case_count'] for m in models))
        self.assertEqual(summary['comparison_models_passing_all_original_screens'],
                         sum(m['comparison']['numerical_screen_passed'] for m in models))
        for model in models:
            transitions = model['output_transitions']
            self.assertEqual(sum(transitions.values()), 64)
            self.assertEqual(transitions['failed_to_pass']+transitions['still_passed'],
                             model['comparison']['passing_output_count'])
        self.assertEqual(result['status'], 'numerical_review_required')
        self.assertFalse(result['all_comparison_screens_passed'])
        self.assertFalse(result['feature_interpretation_authorized'])
        self.assertFalse(result['test_evaluated'])


if __name__ == '__main__':
    unittest.main()

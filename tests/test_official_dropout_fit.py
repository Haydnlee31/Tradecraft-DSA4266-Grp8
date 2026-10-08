"""Paired control, exact historical bridge, and train-only failure guards."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.data.official_inventory import sha256
from src.eval.official_training_fit import diagnose as training_diagnose, fit_mlp
from src.eval.official_dropout_fit import diagnose, comparable, check_pair, check_protocol
from tests import test_official_training_fit as fixtures


class DropoutFitTests(unittest.TestCase):
    def setUp(self):
        # Reuse the small training-only packed fixture, not any actual dataset.
        self.fixture = fixtures.TrainingFitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.settings = self.fixture.plan['mlp']
        self.x, self.y = self.fixture.x.copy(), self.fixture.y.copy()

    def reference_files(self):
        base_path = self.root/'base-plan.json'
        base_path.write_text(json.dumps(self.fixture.plan))
        ref_path = self.root/'reference.json'
        reference = training_diagnose(self.root, base_path, ref_path)
        plan = json.loads((fixtures.PLAN.parent/'dropout-fit-plan.json').read_text())
        plan.update(reference_receipt_sha256=sha256(ref_path),
                    reference_plan_sha256=reference['plan_sha256'],
                    previous_fitter_sha256=reference['source_sha256']['eval/official_training_fit.py'],
                    panel_sha256=reference['panel_sha256'])
        plan_path = self.root/'dropout-plan.json'
        plan_path.write_text(json.dumps(plan))
        return ref_path, plan_path, reference, plan

    def test_default_and_explicit_reference_are_exact(self):
        a = fit_mlp(self.x, self.y, self.settings, 7)
        b = fit_mlp(self.x, self.y, self.settings, 7, dropout=.2)
        self.assertEqual(comparable(a), comparable(b))

    def test_pair_initial_weights_budget_and_zero_replay(self):
        before = self.x.copy()
        a = fit_mlp(self.x, self.y, self.settings, 7)
        b = fit_mlp(self.x, self.y, self.settings, 7, dropout=0.)
        c = fit_mlp(self.x, self.y, self.settings, 7, dropout=0.)
        check_pair(a, b)
        self.assertEqual(comparable(b), comparable(c))
        self.assertEqual(a['num_parameters'], 5096)
        self.assertEqual(b['examples_processed'], 192)
        self.assertEqual(b['optimizer_steps'], 12)
        self.assertNotEqual(a['final_state_sha256'], b['final_state_sha256'])
        np.testing.assert_array_equal(self.x, before)

    def test_invalid_dropout_is_not_a_sweep(self):
        for value in (-.1, .1, .3, float('nan')):
            with self.assertRaises(ValueError):
                fit_mlp(self.x, self.y, self.settings, 7, dropout=value)

    def test_pair_rejects_changed_initialization_and_other_factors(self):
        a = fit_mlp(self.x, self.y, self.settings, 7)
        b = fit_mlp(self.x, self.y, self.settings, 7, dropout=0.)
        for key, value in (('initial_state_sha256', 'wrong'), ('optimizer_steps', 0), ('seed', 17)):
            bad = copy.deepcopy(b)
            bad[key] = value
            with self.assertRaises(ValueError):
                check_pair(a, bad)
        bad = copy.deepcopy(b)
        bad['model_config']['normalization'] = 'batch'
        with self.assertRaises(ValueError):
            check_pair(a, bad)

    def test_bridge_ignores_only_elapsed_time(self):
        a = {'observations': [{'loss': .3}], 'elapsed_seconds': 5.}
        b = copy.deepcopy(a)
        b['elapsed_seconds'] = 9.
        self.assertEqual(comparable(a), comparable(b))
        b['observations'][0]['loss'] = .4
        self.assertNotEqual(comparable(a), comparable(b))

    def test_complete_without_validation_files_and_no_overwrite(self):
        ref_path, plan_path, reference, plan = self.reference_files()
        check_protocol(plan, reference)
        output = self.root/'control.json'
        r = diagnose(self.root, ref_path, plan_path, output)
        self.assertEqual(r['status'], 'complete')
        self.assertTrue(r['baseline_bridge_exact'])
        self.assertEqual(len(r['paired_changes']), 1)
        self.assertFalse(r['validation_opened'])
        self.assertFalse(r['test_opened'])
        with self.assertRaises(FileExistsError):
            diagnose(self.root, ref_path, plan_path, output)
        for key in ('test_opened', 'validation_opened'):
            bad = copy.deepcopy(plan)
            bad[key] = True
            with self.assertRaises(ValueError):
                check_protocol(bad, reference)

    def test_failed_bridge_prevents_treatment(self):
        ref_path, plan_path, reference, _ = self.reference_files()
        bad_fit = copy.deepcopy(reference['mlp_fits'][0])
        bad_fit['final_state_sha256'] = 'wrong'
        output = self.root/'failed-control.json'
        with patch('src.eval.official_dropout_fit.fit_mlp', return_value=bad_fit) as fitting:
            with self.assertRaisesRegex(ValueError, 'bridge failed'):
                diagnose(self.root, ref_path, plan_path, output)
        self.assertEqual(fitting.call_count, 1)
        self.assertNotIn('dropout', fitting.call_args.kwargs)
        r = json.loads(output.read_text())
        self.assertEqual(r['status'], 'incomplete')
        self.assertFalse(r['baseline_bridge_exact'])
        self.assertEqual(r['zero_dropout_fits'], [])

    def test_bad_reference_hash_stops_before_loading_panel(self):
        ref_path, plan_path, _, plan = self.reference_files()
        plan['reference_receipt_sha256'] = 'wrong'
        plan_path.write_text(json.dumps(plan))
        with patch('src.eval.official_dropout_fit.load_training_panel') as loading:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                diagnose(self.root, ref_path, plan_path, self.root/'bad.json')
        loading.assert_not_called()


if __name__ == '__main__':
    unittest.main()

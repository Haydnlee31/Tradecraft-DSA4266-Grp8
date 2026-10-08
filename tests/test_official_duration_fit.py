"""Duration changes preserve the prefix and continue the existing optimizer."""
import copy
import json
import unittest
from unittest.mock import patch

from src.data.official_inventory import sha256
from src.eval import official_training_fit as training
from src.eval.official_dropout_fit import comparable, diagnose as dropout_diagnose
from src.eval.official_duration_fit import diagnose, check_protocol
from tests import test_official_dropout_fit as fixtures


class DurationFitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DropoutFitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.x, self.y = self.fixture.x, self.fixture.y
        self.short = self.fixture.settings
        self.long = copy.deepcopy(self.short)
        self.long.update(epochs=4, report_epochs=[1, 2, 3, 4])

    def reference_files(self):
        base_path, dropout_plan, _, _ = self.fixture.reference_files()
        ref_path = self.root/'dropout-result.json'
        reference = dropout_diagnose(self.root, base_path, dropout_plan, ref_path)
        plan = json.loads((fixtures.fixtures.PLAN.parent/'duration-fit-plan.json').read_text())
        plan.update(reference_receipt_sha256=sha256(ref_path),
                    reference_plan_sha256=reference['plan_sha256'],
                    previous_fitter_sha256=reference['source_sha256']['eval/official_training_fit.py'],
                    panel_sha256=reference['panel_sha256'], reference_epochs=2, epochs=4,
                    additional_report_epochs=[3, 4])
        plan_path = self.root/'duration-plan.json'
        plan_path.write_text(json.dumps(plan))
        return ref_path, plan_path, reference, plan

    def test_prefix_guard_does_not_change_uninterrupted_training(self):
        for dropout in (.2, 0.):
            with self.subTest(dropout=dropout):
                short = training.fit_mlp(self.x, self.y, self.short, 7, dropout=dropout)
                full = training.fit_mlp(self.x, self.y, self.long, 7, dropout=dropout)
                guarded = training.fit_mlp(self.x, self.y, self.long, 7, dropout=dropout,
                                           prefix_reference=short)
                self.assertTrue(guarded.pop('prefix_bridge_exact'))
                self.assertEqual(guarded.pop('prefix_epoch'), 2)
                self.assertEqual(guarded.pop('prefix_state_sha256'), short['final_state_sha256'])
                self.assertEqual(comparable(full), comparable(guarded))
                self.assertEqual(guarded['examples_processed'], 384)
                self.assertEqual(guarded['optimizer_steps'], 24)
                self.assertEqual(comparable(full)['observations'][:2], comparable(short)['observations'])

    def test_bad_prefix_stops_before_the_first_additional_epoch(self):
        short = training.fit_mlp(self.x, self.y, self.short, 7)
        short['final_state_sha256'] = 'corrupted'
        with patch('src.eval.official_training_fit.train_epoch', wraps=training.train_epoch) as epochs:
            with self.assertRaisesRegex(ValueError, 'prefix differs'):
                training.fit_mlp(self.x, self.y, self.long, 7, prefix_reference=short)
        self.assertEqual(epochs.call_count, 2)

    def test_changed_observation_schedule_stops_before_training(self):
        short = training.fit_mlp(self.x, self.y, self.short, 7)
        bad = copy.deepcopy(self.long)
        bad['report_epochs'] = [1, 3, 4]
        with patch('src.eval.official_training_fit.train_epoch') as epochs:
            with self.assertRaisesRegex(ValueError, 'observation schedule'):
                training.fit_mlp(self.x, self.y, bad, 7, prefix_reference=short)
        epochs.assert_not_called()

    def test_plan_bounds_and_condition_coverage(self):
        _, _, ref, plan = self.reference_files()
        check_protocol(plan, ref)
        for key, value in (('epochs', 301), ('require_exact_prefix', False),
                            ('validation_opened', True), ('test_opened', True),
                            ('additional_report_epochs', []), ('dropout_values', [.2])):
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError):
                check_protocol(bad, ref)
        bad_ref = copy.deepcopy(ref)
        bad_ref['zero_dropout_fits'] = []
        with self.assertRaisesRegex(ValueError, 'seed coverage'):
            check_protocol(plan, bad_ref)

    def test_complete_train_only_receipt_and_no_overwrite(self):
        ref_path, plan_path, _, plan = self.reference_files()
        output = self.root/'duration.json'
        r = diagnose(self.root, ref_path, plan_path, output)
        self.assertEqual(r['status'], 'complete')
        self.assertTrue(r['all_prefixes_exact'])
        self.assertFalse(r['validation_opened'])
        self.assertFalse(r['test_opened'])
        self.assertEqual(len(r['duration_changes']), 2)
        self.assertEqual(r['extended_mlp_settings']['epochs'], 4)
        self.assertEqual(sha256(ref_path), plan['reference_receipt_sha256'])
        for key in ('baseline_fits', 'zero_dropout_fits'):
            self.assertEqual(r[key][0]['examples_processed'], 256)
            self.assertEqual(r[key][0]['optimizer_steps'], 16)
        with self.assertRaises(FileExistsError):
            diagnose(self.root, ref_path, plan_path, output)

    def test_reference_checksum_before_any_data_load(self):
        ref_path, plan_path, _, plan = self.reference_files()
        plan['reference_receipt_sha256'] = 'wrong'
        plan_path.write_text(json.dumps(plan))
        with patch('src.eval.official_duration_fit.load_training_panel') as loading:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                diagnose(self.root, ref_path, plan_path, self.root/'bad.json')
        loading.assert_not_called()

    def test_failed_prefix_leaves_incomplete_receipt(self):
        ref_path, plan_path, ref, plan = self.reference_files()
        ref['baseline_fits'][0]['final_state_sha256'] = 'wrong'
        ref_path.write_text(json.dumps(ref))
        plan['reference_receipt_sha256'] = sha256(ref_path)
        plan_path.write_text(json.dumps(plan))
        output = self.root/'failed.json'
        with self.assertRaisesRegex(ValueError, 'prefix differs'):
            diagnose(self.root, ref_path, plan_path, output)
        r = json.loads(output.read_text())
        self.assertEqual(r['status'], 'incomplete')
        self.assertFalse(r['all_prefixes_exact'])
        self.assertEqual(r['baseline_fits'], [])
        self.assertEqual(r['zero_dropout_fits'], [])


if __name__ == '__main__':
    unittest.main()

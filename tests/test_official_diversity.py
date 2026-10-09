"""Matched exposure, historical replay, bounded loading and sealed-test guards."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import official_diversity as diversity
from src.eval.official_dropout_fit import comparable
from src.eval import official_panel_validation as validation
from src.eval.official_training_fit import fit_mlp, load_training_panel, panel_batches
from tests import test_official_panel_validation as fixtures


class DiversityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PanelValidationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def references(self):
        training, tree, validation_plan, _, _ = self.fixture.reference_files()
        reference_root = self.root/'validation-reference'
        reference = validation.diagnose(self.root, training, tree, validation_plan, reference_root)
        plan = json.loads((Path(__file__).parents[1]/'reports/full_data_extension/diversity-plan.json').read_text())
        ref = reference_root/'receipt.json'
        plan.update(reference_receipt_sha256=sha256(ref), reference_plan_sha256=reference['plan_sha256'],
                    training_reference_sha256=sha256(training), evaluation_batch_size=17)
        plan_path = self.root/'diversity-plan.json'
        plan_path.write_text(json.dumps(plan))
        return ref, training, plan_path, reference, plan

    def test_class_cycles_cover_before_reuse_and_repeat_exactly(self):
        a = diversity.ClassCycle(np.arange(7), 7, 0, 426639)
        b = diversity.ClassCycle(np.arange(7), 7, 0, 426639)
        one = np.concatenate([a.take(3), a.take(0), a.take(18)])
        two = b.take(21)
        np.testing.assert_array_equal(one, two)
        for cycle in np.split(one, 3):
            np.testing.assert_array_equal(np.sort(cycle), np.arange(7))
        with self.assertRaises(ValueError):
            a.take(-1)
        with self.assertRaises(ValueError):
            diversity.ClassCycle([1, 1], 7, 0, 426639)

    def test_schedule_preserves_each_batch_and_rare_position(self):
        _, y, rows, _ = load_training_panel(self.root, 8, 7)
        all_y = self.fixture.training.y
        a = diversity.RowSchedule(all_y, rows, 7, False)
        b = diversity.RowSchedule(all_y, rows, 7, True)
        rare_ids = [CLASSES.index(c) for c in diversity.RARE_CLASSES]
        for epoch in range(1, 5):
            # 64 rows and batch size 21 exercise the merged singleton tail.
            expected = list(panel_batches(rows[:, None], y, 21, 7+epoch))
            control, treatment = list(a.epoch(epoch, 21)), list(b.epoch(epoch, 21))
            self.assertEqual([len(x) for x, _ in control], [21, 21, 22])
            for (original, _), (ar, ay), (br, by) in zip(expected, control, treatment):
                np.testing.assert_array_equal(ar, original[:, 0])
                np.testing.assert_array_equal(ay, by)
                mask = np.isin(ay, rare_ids)
                np.testing.assert_array_equal(ar[mask], br[mask])
        self.assertEqual(a.summary()['label_sequence_sha256'], b.summary()['label_sequence_sha256'])
        self.assertEqual(a.summary()['rare_position_sequence_sha256'], b.summary()['rare_position_sequence_sha256'])
        self.assertNotEqual(a.summary()['row_sequence_sha256'], b.summary()['row_sequence_sha256'])
        for name in CLASSES:
            ac, bc = a.summary()['per_class'][name], b.summary()['per_class'][name]
            self.assertEqual(ac['examples_processed'], 32)
            self.assertEqual(bc['examples_processed'], 32)
            self.assertEqual(bc['unique_rows_seen'], 8 if name in diversity.RARE_CLASSES else 12)
            if name in diversity.RARE_CLASSES:
                self.assertEqual(ac, bc)
        for invalid in (rows[:-1], np.append(rows, rows[0]), np.array([-1])):
            with self.assertRaises(ValueError):
                diversity.RowSchedule(all_y, invalid, 7, True)

    def test_new_control_loop_is_identical_to_archived_fitter(self):
        x, y, rows, _ = load_training_panel(self.root, 8, 7)
        settings = self.fixture.training.plan['mlp']
        old = fit_mlp(x, y, settings, 7)
        new, _, schedule = diversity.fit_condition(self.fixture.training.x, self.fixture.training.y,
            rows, settings, 7, False, 426639)
        self.assertEqual(comparable(old), comparable(new))
        self.assertEqual(schedule.summary()['optimizer_steps'], old['optimizer_steps'])

    def test_complete_repeat_matches_and_test_is_never_loaded(self):
        ref, training, plan, _, _ = self.references()
        real_load, real_hash = np.load, sha256
        def no_test_load(path, **kwargs):
            self.assertFalse(Path(path).name.startswith('test_'))
            if Path(path).name in ('train_x.npy', 'train_y.npy', 'val_x.npy', 'val_y.npy'):
                self.assertEqual(kwargs['mmap_mode'], 'r')
            return real_load(path, **kwargs)
        def no_test_hash(path):
            self.assertFalse(Path(path).name.startswith('test_'))
            return real_hash(path)
        with patch('src.eval.official_diversity.np.load', side_effect=no_test_load), \
             patch('src.eval.official_diversity.sha256', side_effect=no_test_hash):
            a = diversity.diagnose(self.root, ref, training, plan, self.root/'diversity-a')
        b = diversity.diagnose(self.root, ref, training, plan, self.root/'diversity-b')
        for r in (a, b):
            self.assertEqual(r['status'], 'complete')
            self.assertTrue(r['all_control_bridges_exact'])
            self.assertTrue(r['all_pairs_matched'])
            self.assertFalse(r['test_opened'])
            self.assertFalse(r['model_promoted'])
            self.assertEqual(len(r['fits']), 2)
            self.assertEqual(len(r['validation_results']), 2)
            for fit in r['fits']:
                fit['training_fit'].pop('elapsed_seconds')
        self.assertEqual(a, b)
        control, treatment = a['fits']
        # Tampering with any matched factor must reject the paired comparison.
        for area, key, value in (
                ('training_fit', 'initial_state_sha256', 'wrong'),
                ('training_fit', 'optimizer_steps', 999),
                ('exposure', 'label_sequence_sha256', 'wrong'),
                ('exposure', 'rare_position_sequence_sha256', 'wrong')):
            bad = copy.deepcopy(treatment)
            bad[area][key] = value
            with self.assertRaises(ValueError):
                diversity.check_pair(control, bad)
        bad = copy.deepcopy(treatment)
        bad['exposure']['per_class']['Brute Force']['unique_rows_seen'] += 1
        with self.assertRaisesRegex(ValueError, 'Rare row'):
            diversity.check_pair(control, bad)
        bad = copy.deepcopy(treatment)
        bad['exposure']['per_class']['Benign']['unique_rows_seen'] -= 1
        with self.assertRaisesRegex(ValueError, 'maximize coverage'):
            diversity.check_pair(control, bad)
        for e in a['fits']:
            counts = np.load(self.root/'diversity-a'/e['exposure_file'], allow_pickle=False)
            self.assertEqual(int(counts.sum()), e['training_fit']['examples_processed'])
            self.assertEqual(sha256(self.root/'diversity-a'/e['exposure_file']), e['exposure_sha256'])
        with self.assertRaises(FileExistsError):
            diversity.diagnose(self.root, ref, training, plan, self.root/'diversity-a')

    def test_control_failure_blocks_treatment_and_validation(self):
        ref, training, plan_path, reference, plan = self.references()
        reference['neural_fits'][0]['training_fit']['final_state_sha256'] = 'bad'
        ref.write_text(json.dumps(reference))
        plan['reference_receipt_sha256'] = sha256(ref)
        plan_path.write_text(json.dumps(plan))
        with patch.object(diversity, 'fit_condition', wraps=diversity.fit_condition) as fit, \
             patch.object(diversity, 'load_validation') as loading:
            with self.assertRaisesRegex(ValueError, 'Control training bridge'):
                diversity.diagnose(self.root, ref, training, plan_path, self.root/'failed')
        self.assertEqual(fit.call_count, 1)
        self.assertFalse(fit.call_args.args[5])
        loading.assert_not_called()
        r = json.loads((self.root/'failed/receipt.json').read_text())
        self.assertFalse(r['validation_opened'])
        self.assertFalse(r['all_control_bridges_exact'])
        self.assertEqual(r['status'], 'incomplete')

    def test_bad_validation_bridge_blocks_treatment_scoring(self):
        ref, training, plan_path, reference, plan = self.references()
        reference['validation_results'][0]['validation_cross_entropy'] = -1
        ref.write_text(json.dumps(reference))
        plan['reference_receipt_sha256'] = sha256(ref)
        plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'Control validation bridge'):
            diversity.diagnose(self.root, ref, training, plan_path, self.root/'failed')
        r = json.loads((self.root/'failed/receipt.json').read_text())
        self.assertEqual(len(r['fits']), 2)
        self.assertEqual(r['validation_results'], [])

    def test_reference_tampering_blocks_all_loading(self):
        ref, training, plan_path, _, plan = self.references()
        plan['reference_receipt_sha256'] = 'wrong'
        plan_path.write_text(json.dumps(plan))
        with patch.object(diversity, 'load_training_panel') as loading:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                diversity.diagnose(self.root, ref, training, plan_path, self.root/'failed')
        loading.assert_not_called()

    def test_protocol_rejects_scope_expansion(self):
        _, training, _, reference, plan = self.references()
        tr = json.loads(training.read_text())
        for key, value in (('test_opened', True), ('model_promoted', True), ('prediction_rule', 'threshold'),
                           ('conditions', ['panel_control']), ('frozen_rare_classes', ['Brute Force']),
                           ('sampling_seed_namespace', 8), ('require_all_control_bridges_before_treatment', False)):
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError):
                diversity.check_protocol(bad, reference, tr)
        tr['mlp_settings']['epochs'] = 301
        with self.assertRaises(ValueError):
            diversity.check_protocol(plan, reference, tr)


if __name__ == '__main__':
    unittest.main()

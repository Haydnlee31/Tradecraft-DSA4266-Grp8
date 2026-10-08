"""Width changes preserve historical fits, exposure and the train-only boundary."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.data.official_inventory import sha256
from src.eval import official_training_fit as training
from src.eval.official_dropout_fit import comparable, check_pair
from src.eval.official_duration_fit import diagnose as duration_diagnose
from src.eval.official_width_fit import diagnose, check_protocol, check_width_pair, prefix_from_duration
from tests import test_official_duration_fit as fixtures


class WidthFitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DurationFitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.x, self.y = self.fixture.x, self.fixture.y
        self.settings = self.fixture.short

    def reference_files(self):
        ref_path, duration_plan, _, _ = self.fixture.reference_files()
        path = self.root/'duration-result.json'
        reference = duration_diagnose(self.root, ref_path, duration_plan, path)
        plan = json.loads((Path(__file__).parents[1]/'reports/full_data_extension/width-fit-plan.json').read_text())
        plan.update(reference_receipt_sha256=sha256(path), reference_plan_sha256=reference['plan_sha256'],
                    previous_fitter_sha256=reference['source_sha256']['eval/official_training_fit.py'],
                    panel_sha256=reference['panel_sha256'], epochs=4)
        plan_path = self.root/'width-plan.json'
        plan_path.write_text(json.dumps(plan))
        return path, plan_path, reference, plan

    def test_default_and_explicit_small_width_are_exact(self):
        a = training.fit_mlp(self.x, self.y, self.settings, 7)
        b = training.fit_mlp(self.x, self.y, self.settings, 7, hidden_dims=(64, 32))
        self.assertEqual(comparable(a), comparable(b))

    def test_wide_replay_and_dropout_initialization_and_work(self):
        a = training.fit_mlp(self.x, self.y, self.settings, 7)
        b = training.fit_mlp(self.x, self.y, self.settings, 7, hidden_dims=(128, 64))
        c = training.fit_mlp(self.x, self.y, self.settings, 7, hidden_dims=(128, 64), dropout=0.)
        replay = training.fit_mlp(self.x, self.y, self.settings, 7, hidden_dims=[128, 64])
        check_width_pair(a, b)
        check_pair(b, c)
        self.assertEqual(comparable(b), comparable(replay))
        self.assertNotEqual(a['initial_state_sha256'], b['initial_state_sha256'])
        self.assertEqual(b['num_parameters'], 14280)
        self.assertEqual(b['examples_processed'], 192)
        self.assertEqual(b['optimizer_steps'], 12)

    def test_invalid_widths_are_not_a_sweep(self):
        for value in ([], [64], [32, 64], [256, 128], [128, 64, 32]):
            with self.assertRaisesRegex(ValueError, 'bounded wider control'):
                training.fit_mlp(self.x, self.y, self.settings, 7, hidden_dims=value)

    def test_reconstructed_prefix_matches_original_exactly(self):
        short = training.fit_mlp(self.x, self.y, self.settings, 7)
        long = training.fit_mlp(self.x, self.y, self.fixture.long, 7, prefix_reference=short)
        self.assertEqual(prefix_from_duration(long), comparable(short))
        for key, value in (('prefix_epoch', 4), ('prefix_bridge_exact', False), ('optimizer_steps', 25)):
            bad = copy.deepcopy(long)
            bad[key] = value
            with self.assertRaises(ValueError):
                prefix_from_duration(bad)

    def test_width_pair_rejects_changes_besides_width(self):
        small = training.fit_mlp(self.x, self.y, self.settings, 7)
        wide = training.fit_mlp(self.x, self.y, self.settings, 7, hidden_dims=(128, 64))
        for key, value in (('seed', 17), ('num_parameters', 999), ('optimizer_steps', 0), ('examples_processed', 1)):
            bad = copy.deepcopy(wide)
            bad[key] = value
            with self.assertRaises(ValueError):
                check_width_pair(small, bad)
        bad = copy.deepcopy(wide)
        bad['model_config']['dropout'] = 0.
        with self.assertRaisesRegex(ValueError, 'beyond the width'):
            check_width_pair(small, bad)

    def test_protocol_bounds_reference_coverage_and_work(self):
        _, _, reference, plan = self.reference_files()
        check_protocol(plan, reference)
        for key, value in (('epochs', 301), ('treatment_hidden_dims', [256, 128]),
                            ('require_all_exact_baseline_bridges_before_treatment', False),
                            ('validation_opened', True), ('test_opened', True)):
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError):
                check_protocol(bad, reference)
        bad = copy.deepcopy(reference)
        bad['zero_dropout_fits'] = []
        with self.assertRaisesRegex(ValueError, 'seed coverage'):
            check_protocol(plan, bad)
        bad = copy.deepcopy(reference)
        bad['baseline_fits'][0]['optimizer_steps'] += 1
        with self.assertRaisesRegex(ValueError, 'work mismatch'):
            check_protocol(plan, bad)

    def test_complete_train_only_receipt_and_no_overwrite(self):
        path, plan, reference, _ = self.reference_files()
        output = self.root/'width.json'
        r = diagnose(self.root, path, plan, output)
        self.assertEqual(r['status'], 'complete')
        self.assertTrue(r['all_baseline_bridges_exact'])
        self.assertFalse(r['validation_opened'])
        self.assertFalse(r['test_opened'])
        self.assertFalse((self.root/'val_x.npy').exists())
        self.assertFalse((self.root/'test_x.npy').exists())
        for key in ('baseline_fits', 'zero_dropout_fits'):
            self.assertEqual(comparable(r['small_fits'][key][0]), comparable(reference[key][0]))
            self.assertEqual(r['wide_fits'][key][0]['examples_processed'], 256)
            self.assertEqual(r['wide_fits'][key][0]['optimizer_steps'], 16)
        self.assertEqual(len(r['width_changes']), 2)
        with self.assertRaises(FileExistsError):
            diagnose(self.root, path, plan, output)

    def test_bad_reference_checksum_before_loading(self):
        path, plan_path, _, plan = self.reference_files()
        plan['reference_receipt_sha256'] = 'wrong'
        plan_path.write_text(json.dumps(plan))
        with patch('src.eval.official_width_fit.load_training_panel') as loading:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                diagnose(self.root, path, plan_path, self.root/'bad.json')
        loading.assert_not_called()

    def test_last_failed_bridge_blocks_every_treatment(self):
        path, plan_path, reference, plan = self.reference_files()
        # Corrupt the final baseline, not the first: all earlier bridges passing
        # must still be insufficient to authorize any wider-model training.
        reference['zero_dropout_fits'][-1]['final_state_sha256'] = 'wrong'
        path.write_text(json.dumps(reference))
        plan['reference_receipt_sha256'] = sha256(path)
        plan_path.write_text(json.dumps(plan))
        output = self.root/'failed.json'
        with patch('src.eval.official_width_fit.fit_mlp', wraps=training.fit_mlp) as fitting:
            with self.assertRaisesRegex(ValueError, 'Baseline bridge failed'):
                diagnose(self.root, path, plan_path, output)
        self.assertEqual(fitting.call_count, 2)
        self.assertTrue(all('hidden_dims' not in c.kwargs for c in fitting.call_args_list))
        r = json.loads(output.read_text())
        self.assertEqual(r['status'], 'incomplete')
        self.assertFalse(r['all_baseline_bridges_exact'])
        self.assertEqual(r['wide_fits'], {'baseline_fits': [], 'zero_dropout_fits': []})


if __name__ == '__main__':
    unittest.main()

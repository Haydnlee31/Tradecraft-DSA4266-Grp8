"""Synthetic orchestration tests; never train official data or require CUDA."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import official39_ablation_confirmation as confirmation
from src.data.official_inventory import sha256
from src.models import official_streaming as training
from src.models.research import atomic_json
from src.eval.official_ablation_recovery import compare
from tests.test_official_ablation_panel import fixture, make_panel
from tests.test_official_ablation import frozen_partition


class ConfirmationPlanTests(unittest.TestCase):
    def test_frozen_plan_and_total_budgets(self):
        p = confirmation.load_plan(confirmation.DEFAULT_PLAN)
        self.assertEqual(sum(p['work_per_run'][l]['examples_processed'] for l in confirmation.LANES) * 6, 720000000)
        self.assertEqual(sum(p['work_per_run'][l]['optimizer_steps'] for l in confirmation.LANES) * 6, 1409280)
        self.assertEqual(p['panel_rows'], 2047805)
        self.assertEqual(confirmation.cases(17), confirmation.cases(27))
        self.assertEqual(len(confirmation.cases(17)), 9)
        for seed in (7, 0, 42):
            with self.assertRaises(ValueError):
                confirmation.cases(seed)

    def test_mutated_plan_rejected(self):
        original = confirmation.read(confirmation.DEFAULT_PLAN)
        changes = [('confirmation_seeds', [7, 17]), ('maximum_new_runs', 9),
                   ('primary_endpoint', 'best'), ('test_evaluation_allowed', True),
                   ('score_based_stopping_allowed', True), ('new_optimizer_steps', 1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            for key, value in changes:
                p = copy.deepcopy(original)
                p[key] = value
                atomic_json(p, path)
                with self.assertRaises(ValueError):
                    confirmation.load_plan(path)
            p = copy.deepcopy(original)
            p['settings']['epochs_or_rounds'] = 40
            atomic_json(p, path)
            with self.assertRaises(ValueError):
                confirmation.load_plan(path)

    def test_published_seed7_metrics_recompute(self):
        r = confirmation.read(confirmation.REPORTS / 'ablation-seed7-results.json')
        self.assertEqual(r['archive_sha256'], '58cefe5732eb2e232d12fa90f5adae5e745f57872ea066e8554f4a1fec115fa8')
        self.assertNotIn('/Users/', r['archive'])
        self.assertEqual(len(r['runs']), 9)
        self.assertEqual(r['checks']['metric_views_recomputed'], 360)
        for v in r['runs'].values():
            for key in ('final_panel', 'final_all_validation'):
                self.assertEqual(v[key], training.confusion_metrics(np.asarray(v[key]['confusion_matrix'])))
            self.assertEqual(sum(c['support'] for c in v['final_panel']['per_class'].values()), 2047805)


class ConfirmationExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        with redirect_stdout(io.StringIO()):
            cls.data, arrays = fixture(cls.root)
            cls.panel, _ = make_panel(cls.root, cls.data, arrays)
            cls.partitions = frozen_partition(cls.root, cls.data, arrays)
            cls.context = dict(plan_sha256='synthetic-plan', orchestrator_sha256=sha256(confirmation.__file__),
                               seed7_archive_sha256='synthetic', references={}, work_per_run={},
                               assignment_sha256={}, cuda_checked=False)
            cls.base_results = {}
            for lane in confirmation.LANES:
                path = cls.root / ('base-' + lane)
                result = training.run(data_root=cls.data, output=path, lane=lane, epochs=2, clients=4,
                                      batch_size=32, normalization='layer', threads=1,
                                      ablation_arm='full39', validation_panel=cls.panel,
                                      partition_root=cls.partitions if lane == 'dirichlet' else None)
                cls.base_results[lane] = result
                cls.context['references'][lane] = confirmation.read(path / 'environment.json')
                cls.context['work_per_run'][lane] = {k: result[k] for k in ('examples_processed', 'optimizer_steps')}
                if lane != 'light':
                    cls.context['assignment_sha256'][lane] = sha256(path / 'assignments.npz')
            cls.completed = cls.root / 'complete'
            # A separate miniature protocol lets the end-to-end summary test
            # exercise real artifacts without impersonating the production pack.
            cls.fixture_plan = {'work_per_run': cls.context['work_per_run'],
                                'seed7_archive_sha256': 'synthetic',
                                'partitions': {lane + '_assignment_sha256': digest
                                               for lane, digest in cls.context['assignment_sha256'].items()}}
            cls.fixture_plan_path = cls.root / 'fixture-plan.json'
            atomic_json(cls.fixture_plan, cls.fixture_plan_path)
            cls.context['plan_sha256'] = sha256(cls.fixture_plan_path)
            for seed in confirmation.SEEDS:
                confirmation.execute_seed(cls.context, seed, cls.completed, cls.data, cls.panel, cls.partitions)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def execute(self, output, seed=17, resume=False, context=None):
        with redirect_stdout(io.StringIO()):
            return confirmation.execute_seed(context or self.context, seed, output,
                                             self.data, self.panel, self.partitions, resume)

    def test_both_seed_batches_complete_and_seed_init_changes(self):
        initial = []
        for seed in confirmation.SEEDS:
            root = self.completed / f'seed{seed}'
            receipt = confirmation.read(root / 'checks.json')
            self.assertEqual(receipt['status'], 'complete')
            self.assertEqual(len(receipt['runs']), 9)
            self.assertEqual(receipt['total_examples_processed'], 2880)
            self.assertEqual(receipt['total_optimizer_steps'], 120)
            self.assertFalse(receipt['test_evaluated'])
            initial.append(confirmation.read(root / 'light-full39/initialization.json'))
        self.assertNotEqual(initial[0], initial[1])

    def test_completed_resume_does_not_retrain_or_modify_receipt(self):
        before = sha256(self.completed / 'seed17/checks.json')
        with patch.object(training, 'run', side_effect=AssertionError('Must not retrain')):
            self.execute(self.completed, resume=True)
        self.assertEqual(before, sha256(self.completed / 'seed17/checks.json'))

    def test_existing_output_requires_explicit_resume(self):
        with self.assertRaises(FileExistsError):
            self.execute(self.completed)

    def test_seed27_cannot_start_before_seed17(self):
        output = self.root / 'out-of-order'
        with patch.object(training, 'run', side_effect=AssertionError('Must not train')):
            with self.assertRaises(FileNotFoundError):
                self.execute(output, seed=27)
        self.assertFalse(output.exists())

    def test_changed_session_and_completed_artifact_rejected_without_retraining(self):
        changed = copy.deepcopy(self.context)
        changed['plan_sha256'] = 'different'
        with patch.object(training, 'run', side_effect=AssertionError('Must not retrain')):
            with self.assertRaisesRegex(ValueError, 'changed on resume'):
                self.execute(self.completed, resume=True, context=changed)
            path = self.completed / 'seed17/light-full39/result.json'
            original = path.read_bytes()
            try:
                altered = confirmation.read(path)
                altered['optimizer_steps'] += 1
                atomic_json(altered, path)
                with self.assertRaisesRegex(ValueError, 'Completed artifact changed'):
                    self.execute(self.completed, resume=True)
            finally:
                path.write_bytes(original)

    def test_interrupted_checkpoint_resumes_exactly_then_completes_all_cases(self):
        output = self.root / 'interrupted'
        real_run = training.run
        def interrupt(**options):
            real_run(**options, stop_after=1)
            raise RuntimeError('Simulated process interruption after a checkpoint')
        with patch.object(training, 'run', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'Simulated'):
                self.execute(output)
        self.assertFalse((output / 'seed17.lock').exists())
        receipt = self.execute(output, resume=True)
        self.assertEqual(receipt['status'], 'complete')
        for lane, arm in confirmation.cases(17):
            filename = f'seed17/{lane}-{arm}/last.pt'
            a, b = [torch.load(root / filename, map_location='cpu', weights_only=True)
                    for root in (output, self.completed)]
            compare(a, b)

    def test_partial_without_checkpoint_is_not_overwritten(self):
        output = self.root / 'no-checkpoint'
        def interrupt(**options):
            options['output'].mkdir()
            raise RuntimeError('Interrupted before checkpoint')
        with patch.object(training, 'run', side_effect=interrupt):
            with self.assertRaises(RuntimeError):
                self.execute(output)
        with self.assertRaisesRegex(ValueError, 'lacks a recoverable checkpoint'):
            self.execute(output, resume=True)

    def test_lock_blocks_second_process_without_removing_first_lock(self):
        output = self.root / 'locked'
        with confirmation.exclusive_seed_lock(output, 17):
            with self.assertRaises(FileExistsError):
                self.execute(output)
            self.assertTrue((output / 'seed17.lock').exists())
        self.assertFalse((output / 'seed17.lock').exists())

    def test_initial_state_mismatch_rejected(self):
        root = self.completed / 'seed17/light-number_masked'
        expected = confirmation.expected_environment(self.context, 'light', 'number_masked', 17)
        with self.assertRaisesRegex(ValueError, 'Initial model differs'):
            confirmation.verify_run(root, expected, self.context['work_per_run']['light'], {'wrong': True})

    def test_summary_requires_all_paired_endpoints_and_keeps_raw_values(self):
        records = {}
        for lane in confirmation.LANES:
            for arm in confirmation.ARMS:
                # Synthetic seed-7 placeholders test arithmetic only, not an
                # empirical seed-7 ablation claim on this fixture.
                records[7, lane, arm] = self.base_results[lane]['primary_validation_metrics']
                for seed in confirmation.SEEDS:
                    path = self.completed / f'seed{seed}/{lane}-{arm}/result.json'
                    records[seed, lane, arm] = confirmation.read(path)['primary_validation_metrics']
        result = confirmation.paired_summary(records)
        self.assertEqual(result['seed_order'], [7, 17, 27])
        self.assertEqual(len(result['paired_mask_minus_full39']), 6)
        values = result['paired_mask_minus_full39']['light/number_masked-minus-full39']['macro_f1']['values']
        self.assertEqual(values[0], 0.)
        self.assertEqual(values[1], records[17, 'light', 'number_masked']['macro_f1'] - records[17, 'light', 'full39']['macro_f1'])
        del records[27, 'dirichlet', 'number_total_masked']
        with self.assertRaisesRegex(ValueError, 'Need all 27'):
            confirmation.paired_summary(records)

    def test_end_to_end_summary_reverifies_saved_runs_and_exposes_all_27_endpoints(self):
        reports = self.root / 'fixture-reports'
        reports.mkdir()
        runs = {}
        for lane in confirmation.LANES:
            r = self.base_results[lane]
            for arm in confirmation.ARMS:
                runs[f'{lane}/{arm}'] = {
                    'final_panel': r['primary_validation_metrics'],
                    'panel_error_counts': r['final_panel_error_counts'],
                    'examples_processed': r['examples_processed'], 'optimizer_steps': r['optimizer_steps'],
                    'training_validation_seconds': sum(h['elapsed_seconds'] for h in r['history']),
                    'checkpoint_sha256': sha256(self.root / f'base-{lane}/last.pt'),
                    'environment_sha256': sha256(self.root / f'base-{lane}/environment.json'),
                    'result_sha256': sha256(self.root / f'base-{lane}/result.json')}
        atomic_json({'runs': runs}, reports / 'ablation-seed7-results.json')
        with patch.object(confirmation, 'REPORTS', reports), patch.object(confirmation, 'load_plan', return_value=self.fixture_plan):
            summary = confirmation.summarize(self.fixture_plan_path, self.completed)
        self.assertEqual(len(summary['endpoints']), 27)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(set(summary['confirmation_receipts_sha256']), {'17', '27'})
        self.assertFalse(summary['test_evaluated'])
        self.assertEqual(summary['seed_roles']['7'], 'exploratory reference')


if __name__ == '__main__':
    unittest.main()

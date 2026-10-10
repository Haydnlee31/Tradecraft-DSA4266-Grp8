"""Synthetic evaluator gates: never read official arrays or private archives."""
import copy
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import official39_final_evaluation as runner
from scripts import official39_test_panel as panel
from src.data.label_map import CLASSES

PLAN = runner.read(runner.PLAN)
FEATURES = PLAN['data_identity']['features']


def fixture():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(65, 39)).astype(np.float32)
    y = np.arange(65, dtype=np.int64) % len(CLASSES)
    retained = np.ones(65, bool)
    retained[:8] = False  # Every class still has primary support.
    torch.manual_seed(11)
    model = runner.MLPClassifier(39, 8, replace(runner.config_for_variant('light'), normalization='layer'))
    return x, y, retained, model


def synthetic_archive(root, bad=None):
    """Small real Torch archive using production field names, synthetic weights."""
    x, y, keep, model = fixture()
    result = runner.score(model, runner.batches(x, y, FEATURES, 'full39'), 'cpu', keep)
    candidate = copy.deepcopy(PLAN['candidates'][0])
    env = dict(classes=CLASSES, features=FEATURES, packed_manifest_sha256=PLAN['data_identity']['packed_manifest_sha256'],
               settings=dict(lane=candidate['lane'], seed=candidate['seed'], epochs=20, batch_size=512),
               input_transform=dict(arm=candidate['arm'], zero_after_scaling=runner.ARMS[candidate['arm']],
                                    features=FEATURES, value=0.),
               model_config=dict(name='centralized_light', hidden_dims=[64, 32], dropout=.2, normalization='layer'),
               source_sha256={name: runner.digest(runner.ROOT / 'src' / name)
                              for name in ('models/architectures.py', 'models/official_ablation.py')}, test_evaluated=False)
    env_raw = json.dumps(env).encode()
    history = [dict(validation_metrics=result['all']['metrics'], panel_validation_metrics=result['panel']['metrics'])] * 20
    original = dict(test_metrics=None, primary_checkpoint='best.pt' if bad == 'best' else 'last.pt', history=history,
                    final_validation_metrics=result['all']['metrics'], primary_validation_metrics=result['panel']['metrics'])
    checkpoint = dict(step=19 if bad == 'step' else 20, model=model.state_dict(), history=history,
                      environment_sha256=hashlib.sha256(env_raw).hexdigest(), input_transform=env['input_transform'])
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    payloads = {'environment.json': env_raw, 'result.json': json.dumps(original).encode(), 'last.pt': stream.getvalue()}
    path = root / 'synthetic.tgz'
    with tarfile.open(path, 'w:gz') as archive:
        for name, raw in payloads.items():
            member = tarfile.TarInfo('run/' + name)
            member.size = len(raw)
            if bad == 'symlink' and name == 'last.pt':
                member.type, member.linkname, member.size = tarfile.SYMTYPE, '/outside', 0
                archive.addfile(member)
            else:
                archive.addfile(member, io.BytesIO(raw))
        if bad == 'duplicate':
            archive.addfile(tarfile.TarInfo('run/result.json'), io.BytesIO(b''))
    candidate['artifact'] = dict(archive=path.name, archive_sha256=runner.digest(path),
        checkpoint_member='run/last.pt', environment_member='run/environment.json', result_member='run/result.json',
        sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()})
    return candidate, model


class FinalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner.configure('cpu')

    def test_mask_copies_inputs_and_matches_singleton_tail(self):
        x, y, _, _ = fixture()
        original = x.copy()
        for arm, removed in runner.ARMS.items():
            chunks = list(runner.batches(x, y, FEATURES, arm, 16))
            self.assertEqual([len(b) for _, b in chunks], [16, 16, 16, 17])
            actual = np.concatenate([a for a, _ in chunks])
            expected = x.copy()
            for name in removed:
                expected[:, FEATURES.index(name)] = 0.
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(x, original)

    def test_same_runtime_reference_exact_for_every_arm(self):
        x, y, keep, model = fixture()
        for arm in runner.ARMS:
            before = runner.state_digest(model)
            r = runner.score(model, runner.batches(x, y, FEATURES, arm, 16), 'cpu', keep)
            _, full, _, primary = runner.evaluate_views(model, runner.batches(x, y, FEATURES, arm, 16), 'cpu', keep)
            self.assertEqual(r['all']['metrics']['confusion_matrix'], full['confusion_matrix'])
            self.assertEqual(r['panel']['metrics']['confusion_matrix'], primary['confusion_matrix'])
            self.assertEqual(runner.state_digest(model), before)
            self.assertEqual(r['forward_batches'], 4)
            self.assertEqual(r['primary_rows'], 57)

    def test_ties_choose_first_class_and_views_share_one_forward(self):
        x, y, keep, model = fixture()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            r = runner.score(model, runner.batches(x, y, FEATURES, 'full39', 16), 'cpu', keep)
        self.assertEqual(forward.call_count, 4)
        for scope in ('all', 'panel'):
            cm = np.asarray(r[scope]['metrics']['confusion_matrix'])
            self.assertEqual(int(cm[:, 1:].sum()), 0)
            self.assertEqual(set(r[scope]['metrics']['per_class']), set(CLASSES))

    def test_invalid_values_and_targets_stop_before_scoring(self):
        x, y, keep, model = fixture()
        x[0, FEATURES.index('Number')] = np.nan
        with self.assertRaisesRegex(ValueError, 'including masked'):
            list(runner.batches(x, y, FEATURES, 'number_masked'))
        x[0, FEATURES.index('Number')] = 0
        y[0] = 8
        with self.assertRaisesRegex(ValueError, 'Invalid inference batch'):
            runner.score(model, [(x, y)], 'cpu', keep)

    def test_coverage_nonfinite_output_and_model_mutation_rejected(self):
        x, y, keep, model = fixture()
        with self.assertRaisesRegex(ValueError, 'Coverage'):
            runner.score(model, [(x[:-1], y[:-1])], 'cpu', keep)
        with patch.object(model, 'forward', return_value=torch.full((65, 8), float('nan'))):
            with self.assertRaisesRegex(ValueError, 'Invalid model output'):
                runner.score(model, [(x, y)], 'cpu', keep)
        original = model.forward
        def changing(values):
            next(model.parameters()).add_(1)
            return original(values)
        with patch.object(model, 'forward', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'model state changed'):
                runner.score(model, [(x, y)], 'cpu', keep)

    def test_explicit_approvals_precede_all_production_reads(self):
        with patch.object(runner, 'load_plan', side_effect=AssertionError('Must not read data')):
            with self.assertRaisesRegex(ValueError, 'Explicit test-preparation approval'):
                panel.prepare('missing', 'missing', 'missing')
            with self.assertRaisesRegex(ValueError, 'Explicit final-test approval'):
                runner.test_evaluation('missing', 'missing', 'missing', 'missing')

    def test_archive_pins_load_exact_synthetic_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, expected = synthetic_archive(Path(directory))
            store = runner.Artifacts(directory, [candidate])
            try:
                actual, _, _ = store.load(candidate, PLAN, 'cpu')
                self.assertEqual(runner.state_digest(actual), runner.state_digest(expected))
            finally:
                store.close()

    def test_archive_rejects_links_duplicates_best_or_wrong_endpoint(self):
        for bad in ('symlink', 'duplicate', 'best', 'step'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                candidate, _ = synthetic_archive(Path(directory), bad)
                store = None
                try:
                    with self.assertRaises(ValueError):
                        store = runner.Artifacts(directory, [candidate])
                        store.load(candidate, PLAN, 'cpu')
                finally:
                    if store:
                        store.close()

    def test_archive_and_member_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, _ = synthetic_archive(Path(directory))
            altered = copy.deepcopy(candidate)
            altered['artifact']['archive_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Archive checksum'):
                runner.Artifacts(directory, [altered])
            candidate['artifact']['sha256']['last.pt'] = '0' * 64
            store = runner.Artifacts(directory, [candidate])
            try:
                with self.assertRaisesRegex(ValueError, 'Artifact checksum'):
                    store.load(candidate, PLAN, 'cpu')
            finally:
                store.close()

    def test_sessions_preserve_outputs_identity_and_running_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            with runner.session(root, {'fixture': True}, False):
                with self.assertRaises(FileExistsError):
                    with runner.session(root, {'fixture': True}, True):
                        self.fail('Concurrent writer admitted')
                self.assertTrue((root / 'running.lock').exists())
            with self.assertRaises(FileExistsError):
                with runner.session(root, {'fixture': True}, False):
                    self.fail('Output overwritten')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                with runner.session(root, {'fixture': False}, True):
                    self.fail('Changed identity resumed')
            with runner.session(root, {'fixture': True}, True):
                pass
            self.assertFalse((root / 'running.lock').exists())

    def test_resume_journal_checks_hashes_and_never_adopts_orphans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = runner.Journal(root, 'fixture', False)
            target = root / 'case-00.json'
            journal.commit(target, {'fixture': True})
            expected = target.read_bytes()
            resumed = runner.Journal(root, 'fixture', True)
            self.assertEqual(resumed.get(target), {'fixture': True})
            with self.assertRaisesRegex(ValueError, 'Preserve completed'):
                resumed.commit(target, {})
            self.assertEqual(target.read_bytes(), expected)
            target.write_bytes(expected + b'\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                runner.Journal(root, 'fixture', True)
            target.write_bytes(expected)
            runner.save(root / 'case-01.json', {})
            with self.assertRaisesRegex(ValueError, 'Orphan'):
                runner.Journal(root, 'fixture', True)

    def test_incomplete_comparison_cannot_pass_gate(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete historical'):
            runner.verify_comparison(dict(historical_comparison={}, same_runtime_reference_exact=True))
        r = dict(historical_comparison={scope: dict(exact=True, confusion_l1_difference=2, macro_f1_difference=0.)
                                       for scope in ('all', 'panel')}, same_runtime_reference_exact=True)
        with self.assertRaisesRegex(ValueError, 'Invalid historical'):
            runner.verify_comparison(r)
        for view in r['historical_comparison'].values():
            view.update(confusion_l1_difference=0, macro_f1_difference=1e-16)
        runner.verify_comparison(r)  # Reduction rounding is not a count mismatch.

    def test_complete_synthetic_test_campaign_resumes_without_new_forwards(self):
        # Exercise final reporting/recovery with fake inputs and a fake approved
        # gate. This is NOT approval to load the official test split.
        x, y, keep, model = fixture()
        prep = dict(rows=len(y), class_counts=panel.class_counts(y), panel=dict(
            retained_rows=int(keep.sum()), retained_class_counts=panel.class_counts(y[keep])))
        @contextmanager
        def synthetic_arrays(*_):
            yield x, y, keep, prep
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.save(root / 'receipt.json', {'fixture': True})
            with patch.object(runner, 'load_plan', return_value=PLAN), \
                    patch.object(runner, 'verify_replay', return_value='synthetic_gate'), \
                    patch.object(panel, 'prepared_arrays', side_effect=synthetic_arrays), \
                    patch.object(runner, 'Artifacts') as store, redirect_stdout(io.StringIO()):
                store.return_value.load.return_value = (model, {}, {})
                result = runner.test_evaluation(root, root, root, root / 'run', allow_final_test=True)
                self.assertEqual(store.return_value.load.call_count, 27)
                before = {p.name: p.read_bytes() for p in (root / 'run').glob('case-*.json')}
                store.return_value.load.reset_mock()
                again = runner.test_evaluation(root, root, root, root / 'run', resume=True, allow_final_test=True)
                store.return_value.load.assert_not_called()
            self.assertEqual(result, again)
            self.assertEqual(len(result['summary']['panel']['groups']), 9)
            self.assertEqual(len(result['summary']['panel']['paired']), 6)
            self.assertFalse(result['model_promoted'])
            self.assertFalse(result['deployment_authorized'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / 'run').glob('case-*.json')})

    def test_validation_campaign_collects_all_cases_but_blocks_on_one_mismatch(self):
        x, y, keep, model = fixture()
        info = {'receipt_sha256': runner.read(runner.closeout.REPORTS / 'ablation-confirmation-plan.json')['panel_receipt_sha256']}
        @contextmanager
        def synthetic_arrays(*_):
            yield {'val': (x, y), 'train': (x, y)}
        def artifact(candidate, *_):
            r = runner.score(model, runner.batches(x, y, FEATURES, candidate['arm']), 'cpu', keep)
            original = dict(final_validation_metrics=r['all']['metrics'], primary_validation_metrics=r['panel']['metrics'])
            if candidate['id'] == PLAN['candidates'][0]['id']:
                # A one-count discrepancy must not be waived because F1 looks
                # close, and must not prevent checking the remaining cases.
                original['final_validation_metrics']['confusion_matrix'][0][0] += 1
            return model, {'input_transform': {'panel': info}}, original
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'replay'
            with patch.object(runner, 'load_plan', return_value=PLAN), \
                    patch.object(runner, 'verify_pack', return_value={'features': FEATURES}), \
                    patch.object(runner, 'packed_arrays', side_effect=synthetic_arrays), \
                    patch.object(runner, 'load_panel', return_value=(keep, info)), \
                    patch.object(runner, 'Artifacts') as store, redirect_stdout(io.StringIO()):
                store.return_value.load.side_effect = artifact
                receipt = runner.validation_replay('unused', 'unused', 'unused', root)
                self.assertEqual(store.return_value.load.call_count, 27)
                store.return_value.load.reset_mock()
                replayed = runner.validation_replay('unused', 'unused', 'unused', root, resume=True)
                store.return_value.load.assert_not_called()
            self.assertEqual(receipt, replayed)
            self.assertEqual(receipt['status'], 'blocked_replay_mismatch')
            self.assertEqual(receipt['exact_same_runtime_references'], 27)
            self.assertEqual(receipt['exact_historical_endpoints'], 26)
            self.assertFalse(receipt['test_scoring_ready'])
            self.assertFalse(receipt['test_opened'])
            with self.assertRaisesRegex(ValueError, 'Validation gate not satisfied'):
                runner.verify_replay(root, receipt['runtime'])


if __name__ == '__main__':
    unittest.main()

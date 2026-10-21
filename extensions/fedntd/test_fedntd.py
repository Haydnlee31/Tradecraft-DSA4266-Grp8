"""Numerical, recovery and isolation audits; synthetic data only."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from extensions.fedntd import core, data as data_module, run as runner
from src.data.label_map import CLASSES
from src.federated import shared_adam as shared
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion


def original_functions():
    """Compile unchanged original numerical functions, avoiding optional app imports.

    This executes the actual baseline function bodies from disk, not a copied
    reimplementation. The full baseline module imports unavailable Flower/Polars.
    """
    source = data_module.ROOT/'src/eval/shared_optimizer_performance.py'
    names = {'work_plan', 'initial_model', 'make_state', 'sampled_rows', 'record_rows', 'advance'}
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    namespace = dict(copy=copy, np=np, torch=torch, shared=shared, CLASSES=CLASSES,
                     MLPClassifier=MLPClassifier, MLPConfig=MLPConfig, hashlib=runner.hashlib)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class FedNTDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def plan(self):
        plan = json.loads(runner.DEFAULT_PLAN.read_text())
        plan.update(clients=2, fit_rows=40, batch_size=4, local_steps=2,
                    backward_examples_per_arm=48)
        return plan

    def fixture(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(40, 39)).astype(np.float32)
        # Unequal global frequencies exercise the actual weighted CE scale;
        # the first client also lacks seven classes entirely.
        y = np.concatenate([np.zeros(16), np.arange(24) % len(CLASSES)]).astype(np.int64)
        devx = rng.normal(size=(16, 39)).astype(np.float32)
        devy = (np.arange(16) % len(CLASSES)).astype(np.int64)
        data = SimpleNamespace(arrays={'fit': (x, y), 'development': (devx, devy)},
                               parts=[np.arange(13), np.arange(13, 40)])
        data.batch = lambda rows: (torch.from_numpy(x[rows].copy()), torch.from_numpy(y[rows].copy()))
        criterion = build_official_criterion('sqrt_weighted_ce',
                    dict(zip(CLASSES, np.bincount(y, minlength=8).tolist())), 'fixed_train_mean')
        return data, criterion

    def state(self, plan):
        return {'model': runner.initial_model(plan), 'round': 0, 'work': {}, 'sample_chain_sha256': '0'*64}

    def assert_model_equal(self, a, b):
        for key in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]), key)

    def test_kl_direction_temperature_true_exclusion_and_teacher_detachment(self):
        student = torch.tensor([[2., -1., 4., 0.], [1., 2., 3., 4.]], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor([[8., 2., -2., 1.], [4., 3., 2., 1.]], dtype=torch.float64, requires_grad=True)
        y, temp = torch.tensor([0, 2]), 2.5
        actual = core.not_true_kl(student, teacher, y, temp)
        terms = []
        for s, t, label in zip(student, teacher.detach(), y):
            keep = [i for i in range(4) if i != int(label)]
            p, q = (t[keep]/temp).softmax(0), (s[keep]/temp).softmax(0)
            terms.append((p*(p.log()-q.log())).sum()*temp**2)
        torch.testing.assert_close(actual, torch.stack(terms).mean())
        changed = student.detach().clone()
        changed[torch.arange(2), y] += 10000
        torch.testing.assert_close(actual, core.not_true_kl(changed, teacher, y, temp), rtol=0, atol=0)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(student.grad[torch.arange(2), y], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_extreme_logits_single_class_batch_and_missing_local_classes(self):
        s = torch.tensor([[10000., -10000., 2000., 0., 1., 2., 3., 4.]]*3, requires_grad=True)
        t = s.detach().flip(1)
        loss = core.not_true_kl(s, t, torch.zeros(3, dtype=torch.long))
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(s.grad).all())

    def test_invalid_distillation_inputs(self):
        logits, target = torch.ones(2, 8), torch.zeros(2, dtype=torch.long)
        for temp in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                core.not_true_kl(logits, logits, target, temp)
        for labels in (torch.tensor([-1, 0]), torch.tensor([8, 0]), torch.zeros(2), torch.zeros(3, dtype=torch.long)):
            with self.assertRaises(ValueError):
                core.not_true_kl(logits, logits, labels)
        for beta in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                core.validate_settings(beta, 1)

    def test_zero_beta_bit_exact_original_fedavg_multiple_rounds_and_rng(self):
        plan, (data, criterion) = self.plan(), self.fixture()
        original = original_functions()
        old_plan = copy.deepcopy(plan)
        old_plan['mime'] = {'local_steps': plan['local_steps']}
        reference = original['make_state'](old_plan, 'fedavg_adam', plan['learning_rate'])
        state = self.state(plan)
        rng = torch.get_rng_state().clone()
        with patch.object(core, 'frozen_teacher', side_effect=AssertionError('zero beta used teacher')):
            for _ in range(3):
                original['advance'](reference, data, old_plan, 'fedavg_adam', plan['learning_rate'], criterion)
                runner.advance(state, data, plan, criterion, 0.)
                self.assert_model_equal(state['model'], reference['model'])
                self.assertEqual(state['sample_chain_sha256'], reference['sample_chain_sha256'])
                for key in ('backward_examples', 'optimizer_steps', 'estimated_payload_bytes', 'logical_barriers'):
                    self.assertEqual(state['work'][key], reference['work'][key])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(state['work']['teacher_forward_examples'], 0)

    def test_teacher_frozen_round_start_input_unchanged_and_positive_beta_effect(self):
        plan, (data, criterion) = self.plan(), self.fixture()
        model = runner.initial_model(plan)
        before = copy.deepcopy(model)
        teachers = []
        def capture(m):
            t = core.frozen_teacher_original(m)
            teachers.append(t)
            return t
        # Keep the implementation reference outside the patched name.
        with patch.object(core, 'frozen_teacher_original', core.frozen_teacher, create=True):
            with patch.object(core, 'frozen_teacher', side_effect=capture):
                updated, work = core.round_update(model, [13, 27],
                    lambda i, j: data.batch(data_module.sampled_rows(data, plan, 0, i, j)), criterion,
                    {**plan['adam'], 'lr': plan['learning_rate']}, 2, plan['seed'], 1.)
        self.assertEqual(len(teachers), 1)
        self.assert_model_equal(model, before)
        self.assert_model_equal(teachers[0], before)
        self.assertFalse(teachers[0].training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teachers[0].parameters()))
        baseline = self.state(plan)
        runner.advance(baseline, data, plan, criterion, 0.)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(updated.parameters(), baseline['model'].parameters())))
        self.assertEqual(work['teacher_forward_examples'], 16)
        self.assertEqual(work['estimated_payload_bytes'], baseline['work']['estimated_payload_bytes'])

    def test_plan_budget_and_invalid_plan(self):
        plan = json.loads(runner.DEFAULT_PLAN.read_text())
        self.assertEqual(runner.validate_plan(plan), 108)
        self.assertEqual(runner.initial_model(plan).num_parameters(), 5096)
        for field, value in [('betas', [.1]), ('backward_examples_per_arm', 3), ('clients', 0)]:
            bad = copy.deepcopy(plan)
            bad[field] = value
            with self.assertRaises(ValueError):
                runner.validate_plan(bad)

    def write_bundle(self, root):
        data, _ = self.fixture()
        for split, arrays in data.arrays.items():
            for suffix, array in zip(('x', 'y'), arrays):
                np.save(root/f'{split}_{suffix}.npy', array)
            np.save(root/f'{split}_row_ids.npy', np.arange(len(arrays[1])) + (0 if split == 'fit' else 100))
        np.savez(root/'clients.npz', **{f'client_{i}': r for i, r in enumerate(data.parts)})
        (root/'scaler.json').write_text('{}')
        manifest = {'status': 'complete', 'plan_sha256': data_module.sha256(data_module.PARENT_PLAN),
                    'classes': CLASSES, 'features': [f'f{i}' for i in range(39)],
                    'files': {n: data_module.sha256(root/n) for n in data_module.ALLOWED_FILES},
                    'counts': {s: dict(zip(CLASSES, np.bincount(y, minlength=8).tolist()))
                               for s, (_, y) in data.arrays.items()}, 'fit_client_counts': [13, 27]}
        # Poison entry: a generic hash-all loader would try to open this file.
        manifest['files']['confirmation_x.npy'] = 'never-open'
        (root/'manifest.json').write_text(json.dumps(manifest))

    def test_data_adapter_memory_maps_whitelist_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_bundle(root)
            data = data_module.DevelopmentData(root, self.plan())
            self.assertIsInstance(data.arrays['fit'][0], np.memmap)
            self.assertNotIn('confirmation_x.npy', data.file_hashes)
            for i in range(2):
                self.assertTrue(np.isin(data_module.sampled_rows(data, self.plan(), 3, i, 1), data.parts[i]).all())
            with (root/'fit_x.npy').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                data_module.DevelopmentData(root, self.plan())

    def test_recovery_round_boundary_complete_runner_and_contract_refusal(self):
        plan, (data, criterion) = self.plan(), self.fixture()
        contract = {'sources': runner.source_hashes(), 'test_fixture': True}
        with tempfile.TemporaryDirectory() as tmp:
            uninterrupted, interrupted = Path(tmp)/'full', Path(tmp)/'resume'
            uninterrupted.mkdir()
            interrupted.mkdir()
            expected = runner.run_locked(data, uninterrupted, plan, 3, criterion, contract)
            real_advance = runner.advance
            calls = 0
            def fail_after_checkpoint(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 5:  # baseline done, positive-beta round one saved
                    raise RuntimeError('simulated process interruption')
                return real_advance(*args, **kwargs)
            with patch.object(runner, 'advance', side_effect=fail_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    runner.run_locked(data, interrupted, plan, 3, criterion, contract)
            actual = runner.run_locked(data, interrupted, plan, 3, criterion, contract)
            for a, b in zip(expected['results'], actual['results']):
                for key in ('development_metrics', 'work', 'sample_chain_sha256'):
                    self.assertEqual(a[key], b[key])
                folder = f'beta-{a["beta"]:g}'
                ca, cb = [torch.load(p/folder/'last.pt', weights_only=True) for p in (uninterrupted, interrupted)]
                for key in ca['model']:
                    self.assertTrue(torch.equal(ca['model'][key], cb['model'][key]))
                self.assertEqual(set(a['development_metrics']['per_class']), set(CLASSES))
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                runner.run_locked(data, interrupted, plan, 3, criterion, {**contract, 'different': True})
            with self.assertRaisesRegex(ValueError, 'Completed run'):
                runner.run_locked(data, interrupted, plan, 3, criterion, contract)

    def test_output_guard_and_no_original_reverse_imports(self):
        with self.assertRaisesRegex(ValueError, 'Output must'):
            runner.run('/not-opened', data_module.ROOT/'reports/old-results')
        for directory in ('src', 'scripts'):
            for path in (data_module.ROOT/directory).rglob('*.py'):
                self.assertNotIn('extensions.fedntd', path.read_text(), str(path))
        config = (data_module.ROOT/'pyproject.toml').read_text()
        self.assertNotIn('extensions.fedntd', config)

    def test_full_entrypoint_on_disk_bundle_and_metrics_match_original(self):
        runs = runner.HERE/'runs'
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs) as output_tmp, tempfile.TemporaryDirectory() as data_tmp:
            root = Path(data_tmp)
            self.write_bundle(root)
            plan_path = root/'plan.json'
            plan_path.write_text(json.dumps(self.plan()))
            result = runner.run(root, Path(output_tmp)/'screen', plan_path)
            self.assertEqual(len(result['results']), 3)
            self.assertFalse((Path(output_tmp)/'screen/RUNNING.lock').exists())
            source = data_module.ROOT/'src/models/official_streaming.py'
            node = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'confusion_metrics')
            namespace = {'np': np, 'CLASSES': CLASSES}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
            for arm in result['results']:
                metrics = dict(arm['development_metrics'])
                self.assertEqual(metrics.pop('decision_rule'), 'argmax')
                self.assertEqual(metrics, namespace['confusion_metrics'](np.array(metrics['confusion_matrix'])))
                self.assertFalse(arm['test_opened'] or arm['confirmation_scored'])

    def test_data_adapter_rejects_overlap_and_bad_ownership_even_with_updated_hash(self):
        for filename, replacement, error in (
            ('development_row_ids.npy', np.arange(16), 'row overlap'),
            ('clients.npz', None, 'Overlapping client'),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.write_bundle(root)
                if replacement is None:
                    np.savez(root/filename, client_0=np.arange(13), client_1=np.arange(12, 40))
                else:
                    np.save(root/filename, replacement)
                manifest = json.loads((root/'manifest.json').read_text())
                manifest['files'][filename] = data_module.sha256(root/filename)
                (root/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, error):
                    data_module.DevelopmentData(root, self.plan())


if __name__ == '__main__':
    unittest.main()

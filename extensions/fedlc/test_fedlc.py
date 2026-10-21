"""Audit tests for the FedLC extension. Run from the repository root:

    python -m unittest extensions.fedlc.test_fedlc -v

The first group needs only NumPy/PyTorch. The second group drives the unchanged
original FedAvg/block128 loops and the real runner on tiny synthetic data, so it
needs the full project environment (the one used for the shared-Adam
reproduction); it is skipped, with the import error shown, when that is absent.
Synthetic scores are software checks, not evidence of an IDS improvement.
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from extensions.fedlc.core import ClientRoutedData, FedLCCriterion, client_class_counts, fedlc_margins
from src.data.label_map import CLASSES

try:
    from extensions.fedlc import run as runner
    from extensions.fedlc.data import sha256
    from src.eval import shared_optimizer_performance as training
    from src.eval.federated_class_safeguard import state_digest
    from src.federated import block_shared_adam as codec
    from src.models.official_losses import build_official_criterion
    FULL_ENVIRONMENT, MISSING = True, ''
except ImportError as error:  # pragma: no cover - depends on the environment
    FULL_ENVIRONMENT, MISSING = False, str(error)

ROOT = Path(__file__).resolve().parents[2]
K = len(CLASSES)


class MemoryData:
    """Minimal in-memory stand-in with the attributes the original loops use."""

    def __init__(self, x, y, parts):
        self.arrays, self.parts, self.manifest = {'fit': (x, y)}, parts, {}

    def batch(self, rows):
        x, y = self.arrays['fit']
        return torch.from_numpy(np.array(x[rows], copy=True)), torch.from_numpy(np.array(y[rows], copy=True))


def synthetic_fit(rows=400, clients=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, rows).astype(np.int64)
    y[:K] = np.arange(K)  # every class present in the pooled fit set
    x = rng.normal(size=(rows, 39)).astype(np.float32)
    x[np.arange(rows), y % 39] += 3.  # learnable signal
    # Unequal, label-skewed ownership: sort by label, then cut unevenly.
    order = np.argsort(y, kind='stable')
    cuts = [0, 40, 140, 260, rows]
    parts = [np.sort(order[a:b]) for a, b in zip(cuts, cuts[1:])]
    return x, y, parts


def tiny_parent(clients=4):
    return {'seed': 7, 'clients': clients, 'fit_rows': 400, 'batch_size': 5,
            'architecture': {'hidden_dims': [64, 32], 'normalization': 'layer', 'dropout': 0.0},
            'adam': {'betas': [0.9, 0.99], 'eps': 0.001, 'weight_decay': 0.0},
            'mime': {'beta1': 0.9, 'beta2': 0.99, 'eps': 0.001, 'initial_v': 0.0001, 'local_steps': 2, 'rounds': 2},
            'anchor_batch_size': 64, 'backward_examples_per_arm': 80}


class MarginTests(unittest.TestCase):
    def test_formula_floor_and_validation(self):
        counts = np.array([10000, 16, 1, 0, 81, 0, 5, 2], dtype=np.int64)
        m = fedlc_margins(counts, 2.0, count_floor=1)
        expected = 2.0*np.maximum(counts, 1).astype(np.float64)**-0.25
        np.testing.assert_allclose(m.numpy(), expected, rtol=0, atol=1e-15)
        self.assertEqual(float(m[3]), 2.0)  # absent class uses the floor
        self.assertGreater(float(fedlc_margins(counts, 1.0, count_floor=16)[2]),
                           0.0)  # larger floors stay finite
        self.assertTrue(torch.equal(fedlc_margins(counts, 0.0), torch.zeros(K, dtype=torch.float64)))
        for bad in (dict(tau=-1.), dict(tau=float('nan')), dict(tau=True), dict(count_floor=0),
                    dict(count_floor=1.5)):
            with self.assertRaises(ValueError):
                fedlc_margins(counts, bad.get('tau', 1.), bad.get('count_floor', 1))
        with self.assertRaises(ValueError):
            fedlc_margins(counts[:7], 1.)
        with self.assertRaises(ValueError):
            fedlc_margins(np.array([-1, *counts[1:]]), 1.)

    def test_client_counts(self):
        y = np.array([0, 0, 1, 7, 7, 7], dtype=np.int64)
        counts = client_class_counts(y, [np.array([0, 1, 2]), np.array([3, 4, 5])])
        self.assertEqual(counts[0].tolist(), [2, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(counts[1].tolist(), [0]*7+[3])


class CriterionTests(unittest.TestCase):
    def setUp(self):
        self.counts = [np.array([5000, 300, 0, 40, 1, 7, 900, 60]), np.array([1]*K)]
        self.base = torch.nn.CrossEntropyLoss()

    def test_matches_independent_formula(self):
        torch.manual_seed(0)
        logits = torch.randn(6, K, dtype=torch.float64)
        target = torch.tensor([0, 2, 4, 4, 6, 1])
        crit = FedLCCriterion(self.base, self.counts, 1.5)
        crit.select(0)
        got = crit(logits, target)
        adjusted = logits - 1.5*torch.from_numpy(np.maximum(self.counts[0], 1).astype(np.float64)**-0.25)
        manual = -(adjusted.gather(1, target[:, None]).squeeze(1) - torch.logsumexp(adjusted, 1)).mean()
        self.assertTrue(torch.allclose(got, manual, rtol=0, atol=1e-12))

    def test_zero_tau_is_the_untouched_base_path(self):
        logits, target = torch.randn(9, K), torch.randint(0, K, (9,))
        weighted = torch.nn.CrossEntropyLoss(weight=torch.rand(K)+.1)
        crit = FedLCCriterion(weighted, self.counts, 0.0)
        crit.select(1)
        self.assertTrue(torch.equal(crit(logits, target), weighted(logits, target)))

    def test_rare_target_is_penalized_more(self):
        # Margins shrink logits of rare local classes most; fitting a rare-class
        # row therefore costs more than under plain CE (the intended pressure).
        logits = torch.zeros(1, K)
        crit = FedLCCriterion(self.base, self.counts, 1.0)
        rare, common = 4, 0  # counts 1 and 5000 on client 0
        crit.select(0)
        rare_loss = crit(logits, torch.tensor([rare]))
        crit.select(0)
        common_loss = crit(logits, torch.tensor([common]))
        plain = self.base(logits, torch.tensor([rare]))
        self.assertGreater(float(rare_loss), float(plain))
        self.assertGreater(float(rare_loss), float(common_loss))

    def test_finite_gradients_absent_class_and_extreme_logits(self):
        logits = torch.tensor([[80., -80., 0., 1e3, -1e3, 3., 2., 1.]]*4, requires_grad=True)
        crit = FedLCCriterion(self.base, self.counts, 3.0)
        crit.select(0)
        loss = crit(logits, torch.tensor([2, 2, 4, 0]))  # class 2 is absent on client 0
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)) and bool(torch.isfinite(logits.grad).all()))

    def test_selection_is_required_and_single_use(self):
        crit = FedLCCriterion(self.base, self.counts, 1.0)
        logits, target = torch.randn(2, K), torch.tensor([0, 1])
        with self.assertRaises(ValueError):
            crit(logits, target)
        crit.select(1)
        crit(logits, target)
        with self.assertRaises(ValueError):
            crit(logits, target)
        for bad in (-1, 2, 1.0, True):
            with self.assertRaises(ValueError):
                crit.select(bad)


class RoutingTests(unittest.TestCase):
    def test_routes_owner_and_returns_identical_batch(self):
        x, y, parts = synthetic_fit()
        data = MemoryData(x, y, parts)
        crit = FedLCCriterion(torch.nn.CrossEntropyLoss(), client_class_counts(y, parts), 1.0)
        view = ClientRoutedData(data, crit)
        for client, rows in enumerate(parts):
            sample = rows[[0, -1, 1]]
            got = view.batch(sample)
            self.assertEqual(crit.client, client)
            for a, b in zip(got, data.batch(sample)):
                self.assertTrue(torch.equal(a, b))
        with self.assertRaises(ValueError):
            view.batch(np.array([parts[0][0], parts[1][0]]))

    def test_rejects_bad_ownership(self):
        x, y, parts = synthetic_fit()
        crit = FedLCCriterion(torch.nn.CrossEntropyLoss(), [np.ones(K, dtype=np.int64)]*4, 1.0)
        with self.assertRaises(ValueError):
            ClientRoutedData(MemoryData(x, y, [parts[0], parts[0], *parts[2:]]), crit)
        with self.assertRaises(ValueError):
            ClientRoutedData(MemoryData(x, y, parts[:3]), crit)


@unittest.skipUnless(FULL_ENVIRONMENT, f'Full project environment needed: {MISSING}')
class OriginalLoopTests(unittest.TestCase):
    """FedLC goes through the unchanged original loops; tau=0 must be bit-exact."""

    def setUp(self):
        self.x, self.y, self.parts = synthetic_fit()
        self.base = tiny_parent()
        counts = dict(zip(CLASSES, np.bincount(self.y, minlength=K).tolist()))
        self.criterion = build_official_criterion('sqrt_weighted_ce', counts, 'fixed_train_mean')
        self.client_counts = client_class_counts(self.y, self.parts)

    def trajectory(self, method, tau):
        data = MemoryData(self.x, self.y, self.parts)
        if tau is None:
            criterion = self.criterion
        else:
            criterion = FedLCCriterion(self.criterion, self.client_counts, tau)
            data = ClientRoutedData(data, criterion)
        state = runner.make_state(method, self.base, 0.003)
        units, _ = runner.units_for(self.base, method)
        for _ in range(units):
            runner.advance(method, state, data, self.base, 0.003, criterion)
        return state

    def test_zero_tau_reproduces_original_bit_exactly(self):
        for method in ('fedavg_adam', 'block128'):
            with self.subTest(method=method):
                original, routed = self.trajectory(method, None), self.trajectory(method, 0.0)
                model = runner.evaluated_model(method, original)
                self.assertEqual(state_digest(model), state_digest(runner.evaluated_model(method, routed)))
                self.assertEqual(state_digest(original['model']), state_digest(routed['model']))
                self.assertEqual(original['sample_chain_sha256'], routed['sample_chain_sha256'])
                self.assertEqual(original['work'], routed['work'])

    def test_positive_tau_changes_weights_not_samples(self):
        for method in ('fedavg_adam', 'block128'):
            with self.subTest(method=method):
                original, treated = self.trajectory(method, None), self.trajectory(method, 1.0)
                self.assertNotEqual(state_digest(original['model']), state_digest(treated['model']))
                self.assertEqual(original['sample_chain_sha256'], treated['sample_chain_sha256'])
                self.assertEqual(original['work'], treated['work'])

    def test_snapshot_restore_continues_identically(self):
        for method in ('fedavg_adam', 'block128'):
            with self.subTest(method=method):
                full = self.trajectory(method, 1.0)
                criterion = FedLCCriterion(self.criterion, self.client_counts, 1.0)
                data = ClientRoutedData(MemoryData(self.x, self.y, self.parts), criterion)
                state = runner.make_state(method, self.base, 0.003)
                runner.advance(method, state, data, self.base, 0.003, criterion)
                saved = copy.deepcopy(runner.snapshot(method, state))
                state = runner.restore(method, saved, self.base, 0.003)
                while state['unit'] < runner.units_for(self.base, method)[0]:
                    runner.advance(method, state, data, self.base, 0.003, criterion)
                self.assertEqual(state_digest(runner.evaluated_model(method, full)),
                                 state_digest(runner.evaluated_model(method, state)))
                self.assertEqual(full['sample_chain_sha256'], state['sample_chain_sha256'])


@unittest.skipUnless(FULL_ENVIRONMENT, f'Full project environment needed: {MISSING}')
class RunnerTests(unittest.TestCase):
    """The actual disk-backed runner on a tiny synthetic bundle (not a real study)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.runs = runner.HERE/'runs'
        self.output = self.runs/f'test-{self.tmp.name}'
        self.addCleanup(shutil.rmtree, self.output, ignore_errors=True)
        parent = tiny_parent()
        self.parent_path = self.tmp/'parent.json'
        self.parent_path.write_text(json.dumps(parent))
        plan = json.loads(runner.DEFAULT_PLAN.read_text())
        plan.update(seeds=[7], expected_backward_examples_per_run=parent['backward_examples_per_arm'])
        self.plan_path = self.tmp/'plan.json'
        self.plan_path.write_text(json.dumps(plan))
        self.data = self.tmp/'data'
        self.write_bundle()

    def write_bundle(self):
        d = self.data
        d.mkdir()
        x, y, parts = synthetic_fit()
        rng = np.random.default_rng(1)
        dev_y = np.repeat(np.arange(K), 200).astype(np.int64)
        dev_x = rng.normal(size=(len(dev_y), 39)).astype(np.float32)
        np.save(d/'fit_x.npy', x); np.save(d/'fit_y.npy', y)
        np.save(d/'development_x.npy', dev_x); np.save(d/'development_y.npy', dev_y)
        np.save(d/'fit_row_ids.npy', np.arange(len(y), dtype=np.int64))
        np.save(d/'development_row_ids.npy', np.arange(len(dev_y), dtype=np.int64)+10**6)
        np.savez(d/'clients.npz', **{f'client_{i}': p for i, p in enumerate(parts)})
        (d/'scaler.json').write_text('{}')
        files = {name: sha256(d/name) for name in ('fit_x.npy', 'fit_y.npy', 'development_x.npy',
                 'development_y.npy', 'fit_row_ids.npy', 'development_row_ids.npy', 'clients.npz', 'scaler.json')}
        manifest = {'status': 'complete', 'plan_sha256': sha256(self.parent_path), 'classes': CLASSES,
                    'features': [f'f{i}' for i in range(39)], 'files': files,
                    'counts': {'fit': dict(zip(CLASSES, np.bincount(y, minlength=K).tolist())),
                               'development': dict(zip(CLASSES, np.bincount(dev_y, minlength=K).tolist()))},
                    'fit_client_counts': [len(p) for p in parts]}
        (d/'manifest.json').write_text(json.dumps(manifest))

    def run_screen(self):
        return runner.run(self.data, self.output, self.plan_path, self.parent_path)

    def test_end_to_end_pairing_metrics_and_protection(self):
        summary = self.run_screen()
        self.assertEqual(len(summary['results']), 4)
        self.assertFalse(summary['contract']['real_study'])
        self.assertEqual(summary['published_bridge_checked'], 0)
        by_arm = {r['arm']: r for r in summary['results']}
        for method in ('fedavg_adam', 'block128'):
            control, treated = by_arm[f'seed7-{method}-control'], by_arm[f'seed7-{method}-fedlc']
            self.assertEqual(control['sample_chain_sha256'], treated['sample_chain_sha256'])
            self.assertIsNone(control['client_margins'])
            self.assertEqual(len(treated['client_margins']), 4)
            self.assertEqual(len(control['evaluation']['operating_points']), 4)
            primary = next(p for p in treated['evaluation']['operating_points']
                           if p['calibration']['target_basis_points'] == 500)
            cm = np.array(primary['audit_metrics']['confusion_matrix'])
            self.assertAlmostEqual(primary['audit_metrics']['macro_f1'],
                                   float(np.mean([2*cm[i, i]/(cm[i].sum()+cm[:, i].sum()) if cm[i].sum()+cm[:, i].sum() else 0.
                                                  for i in range(K)])), places=12)
            rows = summary['fedlc_minus_control'][method]['7']
            self.assertEqual([r['target_basis_points'] for r in rows], [100, 500, 1000, 2000])
            self.assertIn('supports_fedlc', summary['assessment'][method])
        with self.assertRaises(ValueError):
            self.run_screen()  # completed runs are protected

    def test_resume_after_losing_results(self):
        first = self.run_screen()
        (self.output/'summary.json').unlink()
        (self.output/'seed7-block128-fedlc'/'result.json').unlink()
        second = self.run_screen()  # restores the final checkpoint; no retraining
        a = {r['arm']: r['model_sha256'] for r in first['results']}
        b = {r['arm']: r['model_sha256'] for r in second['results']}
        self.assertEqual(a, b)

    def test_rejects_tampered_data_and_bad_output(self):
        with self.assertRaises(ValueError):
            runner.run(self.data, self.tmp/'outside', self.plan_path, self.parent_path)
        np.save(self.data/'fit_y.npy', np.zeros(400, dtype=np.int64))
        with self.assertRaises(ValueError):
            self.run_screen()

    def test_plan_validation(self):
        plan = json.loads(runner.DEFAULT_PLAN.read_text())
        runner.validate_plan(plan)
        for change in ({'base_loss': 'focal'}, {'fedlc': {'tau': 0.0, 'count_floor': 1}},
                       {'methods': {'block128': plan['methods']['block128']}},
                       {'primary_target_basis_points': 300}, {'seeds': [7, 7]}):
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                runner.validate_plan({**plan, **change})


class IsolationTests(unittest.TestCase):
    def test_original_code_never_imports_the_extension(self):
        for path in list((ROOT/'src').rglob('*.py')) + list((ROOT/'scripts').rglob('*.py')):
            self.assertNotIn('extensions.fedlc', path.read_text(encoding='utf-8'), str(path))


if __name__ == '__main__':
    unittest.main()

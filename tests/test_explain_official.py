"""Official-39 explanation preparation uses fixtures, never downloaded data."""
import copy
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

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.explain.official import (
    ROSTER, audit_bridges, freeze_rows, load_frozen, numerical_quality, pilot, prepare, validate_plan,
)
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/full_data_extension/explanation-preparation-plan.json'


def fixture(root):
    """Synthetic completed-run-shaped receipts; no claimed real training occurs."""
    plan = json.loads(PLAN.read_text())
    data = root/'packed'
    data.mkdir()
    rng = np.random.default_rng(4266)
    y = np.repeat(np.arange(8), 16).astype(np.int64)
    for split in ('train', 'val'):
        np.save(data/f'{split}_x.npy', rng.normal(size=(128, 39)).astype(np.float32))
        np.save(data/f'{split}_y.npy', y)
    features = [f'f{i}' for i in range(39)]
    scaler = dict(fit_split='train', features=features, n_samples_seen=128,
                  mean=[0.]*39, scale=[1.]*39, var=[1.]*39)
    (data/'scaler.json').write_text(json.dumps(scaler))
    manifest = dict(status='complete', test_opened=False, features=features, classes=CLASSES,
                    train_rows=128, val_rows=128, train_class_counts=dict.fromkeys(CLASSES, 16),
                    scaler_sha256=sha256(data/'scaler.json'),
                    files={p.name: sha256(p) for p in data.glob('*.npy')})
    (data/'manifest.json').write_text(json.dumps(manifest))
    plan.update(train_rows=128, validation_rows=128, features=features,
                train_class_counts=dict.fromkeys(CLASSES, 16),
                validation_class_counts=dict.fromkeys(CLASSES, 16),
                packed_manifest_sha256=sha256(data/'manifest.json'), archives={}, models=[])
    source_hashes = {name: sha256(ROOT/'src'/name) for name in ('models/architectures.py', 'data/label_map.py')}
    models = {}
    torch.set_num_threads(2)
    with tarfile.open(root/'fixture.tgz', 'w:gz') as archive:
        for lane, seed in ROSTER:
            key = f'{lane}-seed{seed}'
            torch.manual_seed(seed)
            model = MLPClassifier(39, 8, MLPConfig(**plan['model_config'])).eval()
            models[key] = model
            pred = model(torch.from_numpy(np.load(data/'val_x.npy'))).detach().numpy().argmax(1)
            cm = np.zeros((8, 8), dtype=np.int64)
            np.add.at(cm, (y, pred), 1)
            metrics = confusion_metrics(cm)
            env = dict(settings=dict(plan['training_settings'], lane=lane, seed=seed),
                       classes=CLASSES, features=features, model_config=plan['model_config'],
                       packed_manifest_sha256=plan['packed_manifest_sha256'], source_sha256=source_hashes,
                       test_evaluated=False)
            partition_hash = 'a'*64 if lane != 'light' else None
            if lane == 'dirichlet':
                env['partition_control'] = dict(policy='nested-anchor-class-shares-v1', cohort='2m',
                                               assignment_sha256=partition_hash, plan_sha256='b'*64)
            eb = json.dumps(env).encode()
            history = [dict(step=step, validation_metrics=metrics, examples_processed=128,
                            optimizer_steps=2) for step in range(1, 21)]
            result = dict(history=history, final_validation_metrics=metrics, test_metrics=None,
                          examples_processed=2560, optimizer_steps=40, num_parameters=5096)
            buffer = io.BytesIO()
            torch.save(dict(step=20, model=model.state_dict(), history=history,
                            environment_sha256=hashlib.sha256(eb).hexdigest(), partition_sha256=partition_hash), buffer)
            raw = {'environment.json': eb, 'last.pt': buffer.getvalue(),
                   'status.json': json.dumps(dict(status='complete', step=20)).encode(),
                   'result.json': json.dumps(result).encode()}
            spec = dict(id=key, lane=lane, seed=seed, archive='fixture', member_root='runs/'+key,
                        examples_processed=2560, optimizer_steps=40, partition_sha256=partition_hash,
                        files_sha256={name: hashlib.sha256(value).hexdigest() for name, value in raw.items()})
            if lane == 'dirichlet':
                spec['partition_control'] = env['partition_control']
            plan['models'].append(spec)
            for name, value in raw.items():
                member = tarfile.TarInfo(spec['member_root']+'/'+name)
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
    plan['archives']['fixture'] = dict(file='fixture.tgz', sha256=sha256(root/'fixture.tgz'))
    (root/'plan.json').write_text(json.dumps(plan))
    return plan, manifest, models


class OfficialExplanationTests(unittest.TestCase):
    def test_sampling_is_shared_order_independent_and_handles_absence(self):
        settings = json.loads(PLAN.read_text())['settings']
        y = np.repeat(np.arange(8), 30)
        predictions = dict(perfect=y.copy(), all_benign=np.zeros_like(y))
        a = freeze_rows(y, y, predictions, settings)
        self.assertEqual(a, freeze_rows(y, y, dict(reversed(list(predictions.items()))), settings))
        self.assertEqual(len(a['core_rows']), 64)
        self.assertEqual(np.bincount(y[a['core_rows']]).tolist(), [8]*8)
        self.assertEqual(len(a['validation_rows']), len(set(a['validation_rows'])))
        self.assertFalse(set(a['core_rows']) & set(a['supplement_rows']))
        self.assertEqual(np.bincount(y[a['pilot_validation_rows']]).tolist(), [1]*8)
        self.assertEqual(len(set(a['background_train_rows'])), 128)
        for c in CLASSES[1:]:
            self.assertIsNone(a['error_groups']['all_benign'][c+'/correct']['selected_row'])
            self.assertEqual(a['error_groups']['all_benign'][c+'/missed_as_benign']['eligible_count'], 30)
        for groups in a['error_groups'].values():
            self.assertEqual(sum(g['eligible_count'] for g in groups.values()), len(y))

    def test_numerical_screens_reject_incomplete_and_unstable_attribution(self):
        limits = json.loads(PLAN.read_text())['quality_limits']
        a = np.ones((2, 39, 8))
        baseline = np.zeros(8)
        outputs = a.sum(1)
        self.assertTrue(numerical_quality([a, a], outputs, baseline, limits)['numerical_screen_passed'])
        self.assertFalse(numerical_quality([a, a], outputs+10, baseline, limits)['numerical_screen_passed'])
        self.assertFalse(numerical_quality([a*0, a*2], outputs, baseline, limits)['numerical_screen_passed'])
        a[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            numerical_quality([a, a], outputs, baseline, limits)

    def test_actual_shap_linear_identity_and_unchanged_weights(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = torch.nn.Linear(39, 8).eval()
        x = np.random.default_rng(7).normal(size=(2, 39)).astype(np.float32)
        settings = json.loads(PLAN.read_text())['settings']
        settings['pilot_nsamples'] = 8  # Analytic fixture, not the real-data plan.
        qa, arrays = pilot(model, np.zeros((4, 39), dtype=np.float32), x, settings,
                           json.loads(PLAN.read_text())['quality_limits'])
        expected = x[:, :, None]*model.weight.detach().numpy().T[None]
        np.testing.assert_allclose(arrays['repeat_0'], expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(qa['numerical_screen_passed'])

    def test_loader_validates_hashes_and_final_model_not_best(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, manifest, models = fixture(root)
            for spec in plan['models']:
                model, _ = load_frozen(spec, root, plan, manifest)
                for k, tensor in models[spec['id']].state_dict().items():
                    self.assertTrue(torch.equal(tensor, model.state_dict()[k]))
            plan['models'][0]['files_sha256']['last.pt'] = 'changed'
            with patch('src.explain.official.torch.load') as loader:
                with self.assertRaisesRegex(ValueError, 'artifact'):
                    load_frozen(plan['models'][0], root, plan, manifest)
                loader.assert_not_called()

    def test_protocol_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan, manifest, _ = fixture(Path(folder))
            validate_plan(plan, manifest)
            for key, value in [('checkpoint_step', 16), ('pilot_nsamples', 2048), ('confusion_parity', 'tolerant')]:
                altered = copy.deepcopy(plan)
                altered['settings'][key] = value
                with self.assertRaises(ValueError):
                    validate_plan(altered, manifest)
            altered = copy.deepcopy(plan)
            altered['models'][0]['id'] = '../elsewhere'
            with self.assertRaises(ValueError):
                validate_plan(altered, manifest)
            altered = copy.deepcopy(plan)
            altered['models'].pop()
            with self.assertRaises(ValueError):
                validate_plan(altered, manifest)

    def test_prepare_checks_all_bridges_before_pilot_and_keeps_test_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            real_load = np.load

            def guarded_load(path, *args, **kwargs):
                self.assertFalse(Path(path).name.startswith('test'))
                return real_load(path, *args, **kwargs)

            def pilot_stub(model, background, x, settings, limits):
                self.assertEqual(x.shape, (8, 39))
                self.assertEqual(background.shape, (128, 39))
                return {'numerical_screen_passed': True}, {'fixture': np.zeros(1)}

            with patch('src.explain.official.np.load', side_effect=guarded_load), \
                 patch('src.explain.official.pilot', side_effect=pilot_stub) as explainer:
                result = prepare(root/'packed', root, root/'plan.json', root/'out')
                self.assertEqual(explainer.call_count, 7)
            self.assertEqual(result['status'], 'complete')
            self.assertFalse(result['test_evaluated'])
            self.assertFalse(result['full_attribution_run'])
            self.assertFalse(result['feature_interpretation_authorized'])
            with self.assertRaises(ValueError):
                prepare(root/'packed', root, root/'plan.json', root/'out')
            # A failed full-validation bridge must never start SHAP.
            from src.explain.official import scan_validation
            def broken(model, data, size):
                pred, cm = scan_validation(model, data, size)
                cm[0, 0] += 1
                return pred, cm
            with patch('src.explain.official.scan_validation', side_effect=broken), \
                 patch('src.explain.official.pilot') as explainer:
                with self.assertRaisesRegex(ValueError, 'bridge failed'):
                    prepare(root/'packed', root, root/'plan.json', root/'failure')
                explainer.assert_not_called()
                self.assertFalse((root/'failure').exists())

    def test_numerical_failure_is_not_marked_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            with patch('src.explain.official.pilot', return_value=(
                {'numerical_screen_passed': False}, {'fixture': np.zeros(1)})):
                result = prepare(root/'packed', root, root/'plan.json', root/'out')
            self.assertEqual(result['status'], 'numerical_review_required')
            self.assertFalse(result['all_numerical_screens_passed'])
            self.assertFalse(result['feature_interpretation_authorized'])

    def test_v2_requires_exact_cpu_prediction_replay_and_preserves_gpu_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _, _ = fixture(root)
            baseline = root/'cpu-reference.json'
            audit = audit_bridges(root/'packed', root, root/'plan.json', baseline)
            self.assertFalse(audit['attribution_run'])
            plan.update(protocol='official39-explanation-cpu-reference-v2',
                        superseded_plan_sha256=sha256(root/'plan.json'), cpu_reference_sha256=sha256(baseline))
            plan['settings']['confusion_parity'] = 'frozen_cpu_reference'
            revised = root/'plan-v2.json'
            revised.write_text(json.dumps(plan))
            stub = ({'numerical_screen_passed': True}, {'fixture': np.zeros(1)})
            with patch('src.explain.official.pilot', return_value=stub):
                result = prepare(root/'packed', root, revised, root/'v2', baseline)
            self.assertEqual(result['status'], 'complete')
            for key, bridge in result['bridges'].items():
                self.assertEqual(bridge['historical_validation_metrics'], audit['models'][key]['historical_validation_metrics'])
                self.assertEqual(bridge['cpu_prediction_sha256'], audit['models'][key]['cpu_prediction_sha256'])
            # Corrupting only a per-row digest leaves the confusion counts the
            # same, but must still fail. Aggregate parity is not enough for v2.
            audit['models']['light-seed7']['cpu_prediction_sha256'] = 'wrong'
            baseline.write_text(json.dumps(audit))
            plan['cpu_reference_sha256'] = sha256(baseline)
            revised.write_text(json.dumps(plan))
            with patch('src.explain.official.pilot') as explainer:
                with self.assertRaisesRegex(ValueError, 'prediction replay'):
                    prepare(root/'packed', root, revised, root/'bad-reference', baseline)
                explainer.assert_not_called()

    def test_published_plans_and_readiness_do_not_require_local_data(self):
        folder = PLAN.parent
        first = json.loads(PLAN.read_text())
        revised = json.loads((folder/'explanation-cpu-plan.json').read_text())
        reference = json.loads((folder/'explanation-cpu-reference.json').read_text())
        result = json.loads((folder/'explanation-preparation-results.json').read_text())
        self.assertEqual(revised['superseded_plan_sha256'], sha256(PLAN))
        self.assertEqual(revised['cpu_reference_sha256'], sha256(folder/'explanation-cpu-reference.json'))
        self.assertEqual(reference['plan_sha256'], sha256(PLAN))
        self.assertEqual(result['plan_sha256'], sha256(folder/'explanation-cpu-plan.json'))
        self.assertEqual(first['models'], revised['models'])  # Same checkpoint identities.
        self.assertFalse(reference['all_historical_confusions_exact'])
        self.assertEqual(result['status'], 'numerical_review_required')
        self.assertFalse(result['feature_interpretation_authorized'])
        self.assertFalse(result['full_attribution_run'])
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(result['core_rows']+result['supplement_rows'], 219)
        self.assertEqual(sum(result['background_class_counts'].values()), 128)
        self.assertTrue(all(b['explanation_reference_exact'] for b in result['bridges'].values()))
        for counts in result['error_group_population_counts'].values():
            self.assertEqual(sum(counts.values()), revised['validation_rows'])


if __name__ == '__main__':
    unittest.main()

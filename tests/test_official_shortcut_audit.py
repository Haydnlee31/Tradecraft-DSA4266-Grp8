"""Small deterministic fixtures; CI never requires research data or a GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.data import official_shortcut_audit as audit
from src.data import official_shortcut_report as report
from src.data.label_map import CLASSES
from src.data.official_inventory import sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/full_data_extension/shortcut-audit-plan.json'
FEATURES = json.loads((PLAN.parent/'explanation-cpu-plan.json').read_text())['features']


class ShortcutAuditTests(unittest.TestCase):
    def test_frozen_plan_guards(self):
        p = json.loads(PLAN.read_text())
        audit.validate_plan(p)
        self.assertEqual(p['prior_sensitivity_results_sha256'], sha256(PLAN.parent/'explanation-sensitivity-results.json'))
        for key, value in [('test_opened', True), ('model_trained', True), ('rows', {}),
                           ('settings', {}), ('projections', {}), ('packed_manifest_sha256', 'x')]:
            bad = copy.deepcopy(p)
            bad[key] = value
            with self.assertRaises(ValueError):
                audit.validate_plan(bad)

    def test_canonical_exact_bytes_and_signed_zero(self):
        x = np.array([[0., 2.], [-0., 2.], [0., np.nextafter(np.float32(2.), np.float32(3.))]], np.float32)
        a, b, c = audit.canonical(x)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 8)
        with self.assertRaises(ValueError):
            audit.canonical([[float('inf')]])
        with self.assertRaises(ValueError):
            audit.check_labels(np.array([8], np.int64))

    def test_proxy_identity_counterexample_and_invalid_denominator(self):
        x = np.zeros((24, 39), np.float32)
        j = FEATURES.index
        x[:, j('Number')] = 10
        x[:, j('Tot sum')] = 100
        x[:, j('AVG')] = x[:, j('Tot size')] = 10
        x[8:16, j('Tot sum')] = 200  # Not every plausible ratio is an identity.
        x[16:, j('AVG')] = x[16:, j('Tot size')] = 0
        y = np.tile(np.arange(8, dtype=np.int64), 3)
        r = audit.proxy_scan(x, y, dict(mean=[0.]*39, scale=[1.]*39), FEATURES,
                             dict(audit.SETTINGS, batch_size=7))
        self.assertEqual(r['AVG'], r['Tot size'])
        for v in r['AVG'].values():
            self.assertEqual(v['rows'], 3)
            self.assertEqual(v['valid_denominator'], 2)
            self.assertEqual(v['within_tolerance'], 1)
            self.assertEqual(v['near_10_or_100'], 3)
            self.assertEqual(v['window_group_agreement'], 2)
            self.assertEqual(v['mean_abs_error_valid'], 5)
            self.assertEqual(v['max_abs_error'], 10)

    def fixture(self):
        # A has pre-existing cross-split overlap. B acquires overlap after
        # Number removal. C has only same-split duplicates. D stays unique.
        tr = np.zeros((5, 39), np.float32)
        tr[:, 0] = [1, 2, 3, 3, 1]
        tr[:, FEATURES.index('Number')] = [10, 10, 10, 10, 10]
        va = np.zeros((4, 39), np.float32)
        va[:, 0] = [1, 2, 4, 1]
        va[:, FEATURES.index('Number')] = [10, 100, 10, 100]
        return dict(train=(tr, np.array([0, 1, 2, 2, 0], np.int64)),
                    val=(va, np.array([0, 3, 4, 5], np.int64)))

    def test_collision_groups_expansions_conflicts_and_exact_row_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            arrays = self.fixture()
            r = audit.collision_scan(arrays, FEATURES, Path(folder)/'vectors.sqlite',
                                     dict(audit.SETTINGS, batch_size=2))
            a, b = r['full39'], r['without_number']
            self.assertEqual(a['train_rows'], 5)
            self.assertEqual(a['val_rows'], 4)
            self.assertEqual(a['train_duplicate_excess_rows'], 2)
            self.assertEqual(a['val_duplicate_excess_rows'], 0)
            self.assertEqual(a['cross_split_groups'], 1)
            self.assertEqual(a['cross_split_train_rows'], 2)
            self.assertEqual(a['cross_split_val_rows'], 1)
            self.assertEqual(b['cross_split_groups'], 2)
            self.assertEqual(b['cross_split_train_rows'], 3)
            self.assertEqual(b['cross_split_val_rows'], 3)
            self.assertEqual(b['cross_split_row_pairs'], 5)
            self.assertEqual(b['cross_split_different_label_pairs'], 3)
            self.assertEqual(b['cross_split_conflicting_label_groups'], 2)
            self.assertEqual(b['val_conflicting_label_groups'], 1)
            self.assertEqual(b['cross_split_rows_by_class'][CLASSES[0]], {'train': 2, 'val': 1})
            self.assertEqual(r['new_overlap'], dict(train_rows=1, val_rows=2,
                                projected_groups_without_any_baseline_shared_vector=1))
            r2 = audit.collision_scan(arrays, FEATURES, Path(folder)/'replay.sqlite',
                                      dict(audit.SETTINGS, batch_size=9))
            self.assertEqual(r, r2)

    def test_collision_results_match_independent_python_dictionary(self):
        rng = np.random.default_rng(8)
        x = np.zeros((150, 39), np.float32)
        x[:, :2] = rng.integers(0, 4, (150, 2))
        x[:, FEATURES.index('Number')] = rng.integers(0, 3, 150)
        y = rng.integers(0, 8, 150, dtype=np.int64)
        arrays = dict(train=(x[:80], y[:80]), val=(x[80:], y[80:]))
        with tempfile.TemporaryDirectory() as folder:
            r = audit.collision_scan(arrays, FEATURES, Path(folder)/'vectors.sqlite',
                                     dict(audit.SETTINGS, batch_size=17))
        for table, drop in audit.PROJECTIONS.items():
            cols = [i for i, name in enumerate(FEATURES) if name not in drop]
            groups = {}
            for split, (xs, ys) in arrays.items():
                for row, label in zip(xs[:, cols], ys):
                    groups.setdefault(tuple(row), dict(train=[], val=[]))[split].append(int(label))
            shared = [g for g in groups.values() if g['train'] and g['val']]
            self.assertEqual(r[table]['unique_vectors'], len(groups))
            self.assertEqual(r[table]['cross_split_groups'], len(shared))
            self.assertEqual(r[table]['cross_split_row_pairs'], sum(len(g['train'])*len(g['val']) for g in shared))
            self.assertEqual(r[table]['cross_split_different_label_pairs'], sum(a != b for g in shared for a in g['train'] for b in g['val']))
            self.assertEqual(r[table]['conflicting_label_groups'], sum(len(set(g['train']+g['val'])) > 1 for g in groups.values()))

    def test_reader_verifies_before_open_and_closes_only_train_val(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            m = dict(files={}, train_rows=5, val_rows=4)
            for split, pair in self.fixture().items():
                for key, a in zip(('x', 'y'), pair):
                    name = f'{split}_{key}.npy'
                    np.save(root/name, a)
                    m['files'][name] = sha256(root/name)
            real_load, names = np.load, []
            def guarded(path, **kwargs):
                names.append(Path(path).name)
                self.assertNotIn('test', str(path))
                return real_load(path, **kwargs)
            with patch.object(audit.np, 'load', side_effect=guarded):
                with audit.packed_arrays(root, m) as arrays:
                    handles = [a for pair in arrays.values() for a in pair]
                self.assertTrue(all(a._mmap.closed for a in handles))
            self.assertEqual(names, ['train_x.npy', 'train_y.npy', 'val_x.npy', 'val_y.npy'])
            (root/'val_y.npy').write_bytes(b'bad')
            with patch.object(audit.np, 'load') as mocked:
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    with audit.packed_arrays(root, m):
                        self.fail('Corrupt arrays must not open')
                mocked.assert_not_called()

    def test_refuses_existing_database_and_unsupported_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'audit.sqlite'
            db = audit.open_index(path, 16)
            try:
                with self.assertRaises(FileExistsError):
                    audit.open_index(path, 16)
                with self.assertRaises(ValueError):
                    audit.add_rows(db, 'full39', [], np.array([], np.int64), 'test')
                with self.assertRaises(ValueError):
                    audit.summarize(db, 'test')
            finally:
                db.close()

    def test_synthetic_end_to_end_receipt_replay_and_input_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packed = root/'packed'
            packed.mkdir()
            m = dict(status='complete', classes=CLASSES, test_opened=False,
                     features=FEATURES, train_rows=5, val_rows=4, files={})
            for split, pair in self.fixture().items():
                for key, a in zip(('x', 'y'), pair):
                    name = f'{split}_{key}.npy'
                    np.save(packed/name, a)
                    m['files'][name] = sha256(packed/name)
            s = dict(features=FEATURES, fit_split='train', n_samples_seen=5,
                     mean=[0.]*39, scale=[1.]*39)
            (packed/'scaler.json').write_text(json.dumps(s))
            m['scaler_sha256'] = sha256(packed/'scaler.json')
            (packed/'manifest.json').write_text(json.dumps(m))
            (root/'explanation-sensitivity-results.json').write_text('{}')
            p = json.loads(PLAN.read_text())
            p.update(rows={'train': 5, 'val': 4}, packed_manifest_sha256=sha256(packed/'manifest.json'),
                     prior_sensitivity_results_sha256=sha256(root/'explanation-sensitivity-results.json'))
            plan_path = root/'plan.json'
            plan_path.write_text(json.dumps(p))
            # Only the real-cohort allowlist is bypassed; IO/provenance/receipt
            # logic still runs against the synthetic fixture unchanged.
            with patch.object(audit, 'validate_plan'):
                a = audit.run(plan_path, packed, root/'first')
                b = audit.run(plan_path, packed, root/'replay')
                self.assertEqual(a, b)
                self.assertEqual(a['status'], 'complete')
                self.assertTrue(all(a[k] is False for k in audit.FLAGS))
                self.assertEqual(a['auditor_sha256'], sha256(audit.__file__))
                with self.assertRaises(FileExistsError):
                    audit.run(plan_path, packed, root/'first')
                (packed/'scaler.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'scaler'):
                    audit.run(plan_path, packed, root/'bad')
                self.assertFalse((root/'bad').exists())

    def test_ablation_design_preserves_reference_settings_and_pending_gates(self):
        p = json.loads((PLAN.parent/'shortcut-ablation-plan.json').read_text())
        ref = json.loads((PLAN.parent/'explanation-cpu-plan.json').read_text())
        self.assertEqual(p['historical_reference_plan_sha256'], sha256(PLAN.parent/'explanation-cpu-plan.json'))
        self.assertEqual(p['packed_manifest_sha256'], ref['packed_manifest_sha256'])
        receipt = json.loads((PLAN.parent/'shortcut-audit-results.json').read_text())
        self.assertEqual(p['shortcut_audit_source_receipt_sha256'], receipt['source_receipt_sha256'])
        self.assertTrue(all(p[k] is False for k in (*audit.FLAGS, 'model_promoted', 'deployment_authorized')))
        self.assertEqual([a['zero_after_scaling'] for a in p['arms']], [[], ['Number'], ['Number', 'Tot sum']])
        for key, value in ref['training_settings'].items():
            key = 'epochs_or_rounds' if key == 'epochs' else key
            self.assertEqual(p['settings'][key], value)
        for model in ref['models']:
            work = p['work_per_arm_seed'][model['lane']]
            self.assertEqual(work['examples_processed'], model['examples_processed'])
            self.assertEqual(work['optimizer_steps'], model['optimizer_steps'])
        self.assertEqual(p['initial_maximum_runs_after_gates'], len(p['arms'])*len(p['settings']['lanes']))
        self.assertIn('pending', p['status'])

    def test_publication_rejects_same_file_mismatch_and_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, b, out = root/'a.json', root/'b.json', root/'published.json'
            a.write_text('{}')
            b.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'separate'):
                report.publish(a, a, PLAN, out)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                report.publish(a, b, PLAN, out)
            b.write_text('{"status": "incomplete"}')
            with self.assertRaisesRegex(ValueError, 'identical'):
                report.publish(a, b, PLAN, out)
            self.assertFalse(out.exists())

    def test_published_receipt_conservation_and_tamper_checks(self):
        r = json.loads((PLAN.parent/'shortcut-audit-results.json').read_text())
        report.validate(r, PLAN)
        self.assertEqual(r['source_receipt_sha256'], r['independent_replay_receipt_sha256'])
        for split in ('train', 'val'):
            for values in r['proxy'][split].values():
                for stats in values.values():
                    self.assertEqual(stats['within_tolerance'], stats['rows'])
        for key, value in [('model_trained', True), ('status', 'incomplete'), ('auditor_sha256', 'x')]:
            bad = copy.deepcopy(r)
            bad[key] = value
            with self.assertRaises(ValueError):
                report.validate(bad, PLAN)
        bad = copy.deepcopy(r)
        bad['collisions']['new_overlap']['val_rows'] += 1
        with self.assertRaisesRegex(ValueError, 'overlap'):
            report.validate(bad, PLAN)

    def test_windows_receipt_publication_keeps_raw_byte_checksum(self):
        r = json.loads((PLAN.parent/'shortcut-audit-results.json').read_text())
        for key in ('source_receipt_sha256', 'independent_replay_receipt_sha256', 'source_newline', 'publication_scope'):
            r.pop(key)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            encoded = (json.dumps(r, indent=2)+'\n').replace('\n', '\r\n').encode()
            for name in ('a.json', 'b.json'):
                (root/name).write_bytes(encoded)
            published = report.publish(root/'a.json', root/'b.json', PLAN, root/'published.json')
            self.assertEqual(published['source_receipt_sha256'], sha256(root/'a.json'))
            report.validate(published, PLAN)
            report.validate(json.loads((root/'published.json').read_text()), PLAN)


if __name__ == '__main__':
    unittest.main()

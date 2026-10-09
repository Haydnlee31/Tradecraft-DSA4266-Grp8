"""Exact projected overlap and label-blind membership on small CPU fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.data import official_ablation_panel as panel
from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/full_data_extension/ablation-panel-plan.json'
FEATURES = json.loads((PLAN.parent/'explanation-cpu-plan.json').read_text())['features']


def fixture(root):
    """Real pack schema, synthetic values only, with all classes retained."""
    packed = root/'packed'
    packed.mkdir()
    rng = np.random.default_rng(77)
    x = rng.normal(size=(160, 39)).astype(np.float32)
    v = rng.normal(size=(80, 39)).astype(np.float32)
    v[:8] = x[:8]  # Existing full-feature matches.
    v[8:12] = x[8:12]
    v[8:12, FEATURES.index('Number')] += 2  # Number-only matches.
    v[12:16] = x[12:16]
    v[12:16, FEATURES.index('Number')] += 3
    v[12:16, FEATURES.index('Tot sum')] += 4  # New joint-only matches.
    # A within-validation duplicate stays in the panel twice, not deduplicated.
    v[-1] = v[-2]
    labels = dict(train=np.arange(160, dtype=np.int64) % 8, val=np.arange(80, dtype=np.int64) % 8)
    arrays = dict(train=(x, labels['train']), val=(v, labels['val']))
    for split, (xs, ys) in arrays.items():
        np.save(packed/f'{split}_x.npy', xs)
        np.save(packed/f'{split}_y.npy', ys)
    panel.write_json(packed/'scaler.json', dict(features=FEATURES, fit_split='train', n_samples_seen=160,
                                            mean=[0.]*39, scale=[1.]*39))
    m = dict(status='complete', classes=CLASSES, features=FEATURES, test_opened=False,
             train_rows=160, val_rows=80, train_class_counts=dict.fromkeys(CLASSES, 20),
             scaler_sha256=sha256(packed/'scaler.json'),
             files={p.name: sha256(p) for p in packed.glob('*.npy')})
    panel.write_json(packed/'manifest.json', m)
    return packed, arrays


def make_panel(root, packed, arrays, name='panel'):
    out = root/name
    out.mkdir()
    r = dict(protocol='official39-ablation-panel-v1', status='prepared_not_scored',
             packed_manifest_sha256=sha256(packed/'manifest.json'), drop_features=panel.DROP,
             source_validation_rows=len(arrays['val'][1]), classes=CLASSES,
             **dict.fromkeys(panel.FLAGS, False))
    r.update(panel.build(arrays, FEATURES, out, batch_size=13, cache_mib=16))
    panel.write_json(out/'receipt.json', r)
    return out, r


class AblationPanelTests(unittest.TestCase):
    def test_plan_pins_parents_and_scope(self):
        p = json.loads(PLAN.read_text())
        panel.validate_plan(p)
        self.assertEqual(p['ablation_design_sha256'], sha256(PLAN.parent/'shortcut-ablation-plan.json'))
        self.assertEqual(p['prior_audit_results_sha256'], sha256(PLAN.parent/'shortcut-audit-results.json'))
        for key, value in [('model_scored', True), ('drop_features', ['Number']), ('rows', {}), ('batch_size', 3)]:
            bad = copy.deepcopy(p)
            bad[key] = value
            with self.assertRaises(ValueError):
                panel.validate_plan(bad)

    def test_membership_matches_independent_union_and_preserves_multiplicity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packed, arrays = fixture(root)
            out, r = make_panel(root, packed, arrays)
            expected = set()
            for drop in ([], ['Number'], panel.DROP):
                cols = [i for i, name in enumerate(FEATURES) if name not in drop]
                train_keys = {tuple(row) for row in arrays['train'][0][:, cols]}
                expected.update(i for i, row in enumerate(arrays['val'][0][:, cols]) if tuple(row) in train_keys)
            np.testing.assert_array_equal(np.load(out/'excluded_rows.npy'), sorted(expected))
            self.assertEqual(expected, set(range(16)))
            self.assertEqual(r['retained_rows'], 64)
            self.assertEqual(r['joint_projection']['val_duplicate_excess_rows'], 1)
            self.assertTrue(r['all_classes_retained'])
            self.assertEqual(r['retained_class_counts'], dict.fromkeys(CLASSES, 8))

    def test_label_permutation_cannot_change_membership_or_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packed, arrays = fixture(root)
            a, r1 = make_panel(root, packed, arrays, 'a')
            changed = {s: (x, (y+3) % 8) for s, (x, y) in arrays.items()}
            b, _ = make_panel(root, packed, changed, 'b')
            for name in panel.FILES:
                self.assertEqual(sha256(a/name), sha256(b/name))
            c = root/'c'
            c.mkdir()
            r2 = panel.build(arrays, FEATURES, c, batch_size=80, cache_mib=16)
            self.assertEqual(r1['joint_projection'], r2['joint_projection'])
            self.assertEqual(r1['files'], r2['files'])

    def test_panel_loader_guards_indices_counts_cohort_and_immutability(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packed, arrays = fixture(root)
            out, r = make_panel(root, packed, arrays)
            data = PackedData(packed)
            try:
                mask, meta = panel.load_panel(out, data)
                self.assertEqual(int(mask.sum()), 64)
                self.assertEqual(meta['receipt_sha256'], sha256(out/'receipt.json'))
                for key, value in [('model_scored', True), ('packed_manifest_sha256', 'x'), ('retained_rows', 1),
                                   ('retained_class_counts', dict.fromkeys(CLASSES, 0))]:
                    bad = copy.deepcopy(r)
                    bad[key] = value
                    panel.write_json(out/'receipt.json', bad)
                    with self.assertRaises(ValueError):
                        panel.load_panel(out, data)
                panel.write_json(out/'receipt.json', r)
                rows = np.load(out/'retained_rows.npy')
                rows[1] = rows[0]
                np.save(out/'retained_rows.npy', rows)
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    panel.load_panel(out, data)
                r['files']['retained_rows.npy'] = sha256(out/'retained_rows.npy')
                panel.write_json(out/'receipt.json', r)
                with self.assertRaisesRegex(ValueError, 'sorted unique'):
                    panel.load_panel(out, data)
            finally:
                for pair in data.arrays.values():
                    for a in pair:
                        a._mmap.close()

    def test_fails_on_nonfinite_masked_column_and_missing_class_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, arrays = fixture(root)
            arrays['train'][0][0, FEATURES.index('Number')] = np.nan
            out = root/'bad'
            out.mkdir()
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                panel.build(arrays, FEATURES, out, batch_size=8, cache_mib=16)
            arrays['train'][0][0, FEATURES.index('Number')] = 0
            y = arrays['val'][1].copy()
            y[y == 7] = 0
            arrays['val'] = (arrays['val'][0], y)
            out2 = root/'missing'
            out2.mkdir()
            r = panel.build(arrays, FEATURES, out2, batch_size=16, cache_mib=16)
            self.assertFalse(r['all_classes_retained'])

    def test_synthetic_run_publication_replay_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packed, _ = fixture(root)
            p = json.loads(PLAN.read_text())
            previous = {'collisions': {'full39': {'cross_split_train_rows': 8, 'cross_split_val_rows': 8}}}
            panel.write_json(root/'shortcut-audit-results.json', previous)
            panel.write_json(root/'shortcut-ablation-plan.json', {})
            p.update(rows={'train': 160, 'val': 80}, packed_manifest_sha256=sha256(packed/'manifest.json'),
                     prior_audit_results_sha256=sha256(root/'shortcut-audit-results.json'),
                     ablation_design_sha256=sha256(root/'shortcut-ablation-plan.json'))
            plan = root/'plan.json'
            panel.write_json(plan, p)
            with patch.object(panel, 'validate_plan'):
                a = panel.run(plan, packed, root/'first')
                b = panel.run(plan, packed, root/'replay')
                self.assertEqual(a, b)
                self.assertEqual(a['additional_overlap_rows_vs_full39'], {'train': 8, 'val': 8})
                published = panel.publish(root/'first', root/'replay', plan, root/'published.json')
                self.assertEqual(published['source_receipt_sha256'], published['independent_replay_receipt_sha256'])
                with self.assertRaises(FileExistsError):
                    panel.run(plan, packed, root/'first')
                with self.assertRaises(ValueError):
                    panel.publish(root/'first', root/'first', plan, root/'bad.json')
                (root/'replay'/'retained_rows.npy').write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError, 'indices'):
                    panel.publish(root/'first', root/'replay', plan, root/'bad.json')


if __name__ == '__main__':
    unittest.main()

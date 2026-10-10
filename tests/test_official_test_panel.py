"""Label-blind overlap screening on tiny synthetic data; official test closed."""
import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import official39_final_evaluation as evaluation
from scripts import official39_test_panel as panel
from src.data.label_map import CLASSES

FEATURES = evaluation.read(evaluation.PLAN)['data_identity']['features']


def fixture():
    rng = np.random.default_rng(9)
    references = {name: rng.normal(size=(16, 39)).astype(np.float32) for name in ('train', 'val')}
    x = rng.normal(size=(64, 39)).astype(np.float32)
    y = np.arange(64, dtype=np.int64) % len(CLASSES)
    x[:8] = references['train'][:8]
    x[8:16] = references['val'][:8]
    x[16:24] = references['train'][8:16]
    x[16:24, FEATURES.index('Number')] += 20
    x[24:32] = references['val'][8:16]
    x[24:32, FEATURES.index('Number')] += 20
    x[24:32, FEATURES.index('Tot sum')] += 30
    x[-1] = x[-2]  # Preserve both rows even when their class labels differ.
    return references, x, y


class TestPanelTests(unittest.TestCase):
    def test_membership_matches_independent_exact_projection_union(self):
        refs, x, y = fixture()
        union = set()
        with tempfile.TemporaryDirectory() as directory:
            result = panel.screening(refs, x, y, FEATURES, directory, 7)
            for arm, drop in panel.ARMS.items():
                columns = [i for i, name in enumerate(FEATURES) if name not in drop]
                matched = {}
                for split, source in refs.items():
                    keys = {tuple(row) for row in source[:, columns]}
                    matched[split] = {i for i, row in enumerate(x[:, columns]) if tuple(row) in keys}
                    self.assertEqual(result['projections'][arm]['matched_rows'][split], len(matched[split]))
                combined = matched['train'] | matched['val']
                self.assertEqual(result['projections'][arm]['matched_rows']['either'], len(combined))
                union |= combined
                self.assertEqual(result['projections'][arm]['duplicate_excess_rows'], 1)
            self.assertEqual(union, set(range(32)))
            retained = np.load(Path(directory) / 'retained.npy', allow_pickle=False)
            np.testing.assert_array_equal(np.flatnonzero(~retained), sorted(union))
            self.assertTrue(retained[-2:].all())
            self.assertTrue(result['all_classes_retained'])
            self.assertEqual(result['retained_class_counts'], dict.fromkeys(CLASSES, 4))

    def test_labels_and_batch_size_cannot_change_membership(self):
        refs, x, y = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('a', 'b', 'c'):
                (root / name).mkdir()
            a = panel.screening(refs, x, y, FEATURES, root / 'a', 7)
            panel.screening(refs, x, (y + 3) % 8, FEATURES, root / 'b', 19)
            c = panel.screening(refs, x, y, FEATURES, root / 'c', 64)
            for name in ('b', 'c'):
                self.assertEqual((root / 'a' / 'retained.npy').read_bytes(), (root / name / 'retained.npy').read_bytes())
            self.assertEqual(a, c)

    def test_signed_zero_and_both_reference_sources(self):
        refs, x, y = fixture()
        refs['train'][0, :] = 0.
        refs['val'][0, :] = -0.
        x[0, :] = -0.
        with tempfile.TemporaryDirectory() as directory:
            r = panel.screening(refs, x, y, FEATURES, directory)
            p = r['projections']['full39']['matched_rows']
            self.assertGreater(p['train'] + p['val'], p['either'])
            self.assertFalse(np.load(Path(directory) / 'retained.npy')[0])

    def test_missing_primary_class_blocks_readiness_without_redraw(self):
        refs, x, y = fixture()
        refs['train'] = x[y == 0].copy()
        with tempfile.TemporaryDirectory() as directory:
            r = panel.screening(refs, x, y, FEATURES, directory)
            self.assertFalse(r['all_classes_retained'])
            self.assertEqual(r['retained_class_counts'][CLASSES[0]], 0)

    def test_nonfinite_masked_value_and_existing_index_rejected(self):
        refs, x, y = fixture()
        x[0, FEATURES.index('Number')] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                panel.screening(refs, x, y, FEATURES, directory)
            with self.assertRaisesRegex(ValueError, 'Preserve existing'):
                panel.screening(refs, x, y, FEATURES, directory)

    def test_saved_scaler_uses_train_statistics_without_fit(self):
        record = dict(features=FEATURES, fit_split='train', n_samples_seen=2000000,
                      mean=list(np.linspace(.1, 2, 39)), var=[4.] * 39, scale=[2.] * 39)
        values = np.random.default_rng(4).normal(size=(19, 39)).astype(np.float64)
        with patch.object(panel.StandardScaler, 'fit', side_effect=AssertionError('Never fit on held-out rows')):
            scaler = panel.scaler_from_record(record, FEATURES)
            actual = scaler.transform(values).astype(np.float32)
        expected = ((values - np.asarray(record['mean'])) / np.asarray(record['scale'])).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        record['n_samples_seen'] = 500000
        with self.assertRaisesRegex(ValueError, 'fixed 2M'):
            panel.scaler_from_record(record, FEATURES)

    def test_synthetic_preparation_pins_arrays_and_loader_rejects_tampering(self):
        refs, x, y = fixture()
        @contextmanager
        def synthetic_references(*_):
            yield {name: (values, np.arange(len(values), dtype=np.int64) % 8) for name, values in refs.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('manifest.json', 'protocol.json'):
                evaluation.save(root / name, {'synthetic': True})
            evaluation.save(root / 'scaler.json', dict(features=FEATURES, fit_split='train', n_samples_seen=2000000,
                                                       mean=[0.] * 39, var=[1.] * 39, scale=[1.] * 39))
            plan = copy.deepcopy(evaluation.read(evaluation.PLAN))
            identity = plan['data_identity']
            identity.update(parent_shards_manifest_sha256=evaluation.digest(root / 'manifest.json'),
                            split_protocol_sha256=evaluation.digest(root / 'protocol.json'),
                            scaler_sha256=evaluation.digest(root / 'scaler.json'),
                            test_rows_before_overlap_filter=len(y), test_class_counts_before_overlap_filter=panel.class_counts(y))
            out = root / 'prepared'
            with patch.object(evaluation, 'load_plan', return_value=plan), \
                    patch.object(evaluation, 'verify_pack', return_value={}), \
                    patch.object(panel, 'OfficialBatches') as loader, \
                    patch.object(panel, 'packed_arrays', side_effect=synthetic_references):
                loader.return_value.rows, loader.return_value.features = len(y), FEATURES
                loader.return_value.batches.side_effect = lambda _: iter([(x[:32], y[:32]), (x[32:], y[32:])])
                r = panel.prepare(root, root, out, allow_test_preparation=True)
                loader.assert_called_once_with(root, 'test', batch_size=panel.BATCH_SIZE, allow_test=True)
                self.assertEqual(loader.return_value.batches.call_count, 2)
            self.assertEqual(r['status'], 'prepared_not_scored')
            self.assertFalse(r['test_evaluated'])
            with panel.prepared_arrays(out, plan) as (actual_x, actual_y, keep, receipt):
                np.testing.assert_array_equal(actual_x, x)
                np.testing.assert_array_equal(actual_y, y)
                self.assertEqual(int(keep.sum()), 32)
                self.assertEqual(receipt, r)
            bad = copy.deepcopy(r)
            bad['panel']['retained_class_counts'][CLASSES[0]] += 1
            evaluation.save(out / 'receipt.json', bad)
            with self.assertRaisesRegex(ValueError, 'supports changed'):
                with panel.prepared_arrays(out, plan):
                    self.fail('Changed population admitted')
            evaluation.save(out / 'receipt.json', r)
            np.save(out / 'retained.npy', np.ones(len(y), bool), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, 'checksum changed'):
                with panel.prepared_arrays(out, plan):
                    self.fail('Changed membership admitted')


if __name__ == '__main__':
    unittest.main()

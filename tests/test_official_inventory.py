"""Synthetic input tests; these do not establish official-data readiness."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.data.label_map import LABEL_TO_CLASS
from src.data.official_inventory import inventory, iter_staged_batches


class OfficialInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.features = [f'f{i}' for i in range(46)]
        self.labels = list(LABEL_TO_CLASS)

    def csv(self, name='a.csv', rows=80, bad=None):
        with (self.raw / name).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow([*self.features, 'label'])
            for i in range(rows):
                values = [float(i + j) for j in range(46)]
                if bad == 'nan':
                    values[0] = 'nan'
                writer.writerow([*values, 'unknown' if bad == 'label' else self.labels[i % 34]])

    def test_shards_preserve_all_rows_provenance_and_bounds(self):
        self.csv()
        self.csv('b.csv', rows=35)
        out = self.root / 'stage'
        result = inventory(self.raw, out, self.features, 4096, 7, True)
        self.assertEqual(result['rows'], 115)
        self.assertEqual(sum(result['class_counts'].values()), 115)
        self.assertFalse(result['training_ready'])
        batches = list(iter_staged_batches(out, 3))
        self.assertTrue(all(len(batch) <= 3 for batch in batches))
        rows = [row for batch in batches for row in batch.to_pylist()]
        self.assertEqual([(r['source_file_id'], r['source_row']) for r in rows],
                         [(0, i) for i in range(80)] + [(1, i) for i in range(35)])
        self.assertEqual([r['f0'] for r in rows], list(map(float, range(80))) + list(map(float, range(35))))
        self.assertTrue(all(s['rows'] <= 7 for s in result['shards']))
        # Source inventory is independent of parser block boundaries.
        other = inventory(self.raw, self.root / 'other', self.features, 8192)
        self.assertEqual(result['raw_label_counts'], other['raw_label_counts'])

    def test_unknown_label_keeps_incomplete_receipt(self):
        self.csv(bad='label')
        out = self.root / 'failed'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            inventory(self.raw, out, self.features)
        self.assertEqual(json.loads((out / 'inventory.json').read_text())['status'], 'incomplete')
        with self.assertRaises(ValueError):
            list(iter_staged_batches(out))

    def test_nonfinite_inventory_and_staging_rejection(self):
        self.csv(rows=2, bad='nan')
        result = inventory(self.raw, self.root / 'counts', self.features)
        self.assertEqual(result['files'][0]['nonfinite_by_feature']['f0'], 2)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            inventory(self.raw, self.root / 'stage', self.features, write_shards=True)

    def test_refuse_overwrite_and_detect_shard_drift(self):
        self.csv(rows=2)
        out = self.root / 'stage'
        result = inventory(self.raw, out, self.features, write_shards=True)
        with self.assertRaises(FileExistsError):
            inventory(self.raw, out, self.features)
        (out / result['shards'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            list(iter_staged_batches(out))

    def test_schema_and_empty_source_rejected_before_output(self):
        out = self.root / 'stage'
        with self.assertRaises(FileNotFoundError):
            inventory(self.raw, out, self.features)
        self.csv(rows=2)
        with self.assertRaisesRegex(ValueError, 'schema'):
            inventory(self.raw, out, ['wrong', *self.features[1:]])
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()

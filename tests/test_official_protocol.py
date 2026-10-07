"""Splits are feature-group dependent, deterministic and label independent."""
import hashlib
import unittest
import json
from pathlib import Path
import tempfile

from src.data.official_protocol import split_for_digest, freeze
from src.data.official_quality import open_index, add_vectors, summarize_index
from src.data.label_map import CLASSES


class OfficialProtocolTests(unittest.TestCase):
    def test_freeze_counts_conflicts_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = open_index(root / 'vectors.sqlite')
            for i, name in enumerate(CLASSES):
                hashes = [hashlib.sha256(f'{name}:{j}'.encode()).digest() for j in range(200)]
                add_vectors(db, hashes, 1 << i, 1 << i, i)
            conflict = hashlib.sha256(b'conflict').digest()
            add_vectors(db, [conflict], 1, 1, 0)
            add_vectors(db, [conflict], 2, 2, 1)
            db.commit()
            summary = {'status': 'complete', 'features': [f'f{i}' for i in range(39)],
                       'totals': {'float32_overflow_rows': 0}, 'duplicates': summarize_index(db)}
            db.close()
            (root / 'summary.json').write_text(json.dumps(summary))
            output = root / 'protocol.json'
            result = freeze(root, output)
            self.assertEqual(sum(result['unique_vectors_by_split'].values()), 1600)
            self.assertFalse(result['training_ready'])
            self.assertFalse(result['test_evaluated'])
            with self.assertRaises(FileExistsError):
                freeze(root, output)

    def test_group_assignment_is_deterministic(self):
        digest = hashlib.sha256(b'example').digest()
        self.assertEqual(split_for_digest(digest), split_for_digest(digest))
        values = [split_for_digest(hashlib.sha256(str(i).encode()).digest()) for i in range(10000)]
        self.assertEqual(set(values), {'train', 'val', 'test'})
        self.assertTrue(7800 < values.count('train') < 8200)
        self.assertTrue(800 < values.count('val') < 1200)
        self.assertTrue(800 < values.count('test') < 1200)


if __name__ == '__main__':
    unittest.main()

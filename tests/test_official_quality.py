"""Guard label normalization and label-independent numerical duplicate checks."""
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from collections import namedtuple

import numpy as np

from src.data.label_map import official_raw_label, to_class
from src.data.official_quality import fingerprints, open_index, add_vectors, summarize_index, run
from src.data.official_schema_audit import audit


class OfficialQualityTests(unittest.TestCase):
    def test_interrupted_file_rolls_back_and_resume_matches_full_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            folder = source / 'Benign_Final'
            folder.mkdir(parents=True)
            features = [f'f{i}' for i in range(39)]
            for i in range(2):
                (folder / f'{i}.csv').write_text(','.join(features) + '\n' + ','.join([str(i)] * 39) + '\n')
            receipt = root / 'schema.json'
            receipt.write_text(json.dumps(audit(source, features)))
            calls = 0

            def interrupted(*args):
                nonlocal calls
                calls += 1
                add_vectors(*args)
                if calls == 2:
                    raise RuntimeError('simulated interruption')

            usage = namedtuple('Usage', 'total used free')(100*1024**3, 0, 100*1024**3)
            with patch('src.data.official_quality.shutil.disk_usage', return_value=usage):
                with patch('src.data.official_quality.add_vectors', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'simulated'):
                        run(source, receipt, root / 'resumed', cache_mib=16)
                resumed = run(source, receipt, root / 'resumed', resume=True, cache_mib=16)
                full = run(source, receipt, root / 'full', cache_mib=16)
            self.assertEqual(resumed['files'], full['files'])
            self.assertEqual(resumed['duplicates'], full['duplicates'])
            self.assertEqual(resumed['totals'], full['totals'])

    def test_full_quality_audit_matches_structural_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            folder = source / 'Benign_Final'
            folder.mkdir(parents=True)
            features = [f'f{i}' for i in range(39)]
            good = ','.join(['1'] * 39)
            invalid = ','.join(['nan'] + ['1'] * 38)
            (folder / 'example.csv').write_text(','.join(features) + '\n' +
                                               good + '\n' + good + '\n' + invalid + '\n1,2\n')
            receipt = root / 'schema.json'
            receipt.write_text(json.dumps(audit(source, features)))
            usage = namedtuple('Usage', 'total used free')(100*1024**3, 0, 100*1024**3)
            with patch('src.data.official_quality.shutil.disk_usage', return_value=usage):
                result = run(source, receipt, root / 'quality')
            self.assertEqual(result['totals']['rows'], 3)
            self.assertEqual(result['totals']['nonfinite_rows'], 1)
            self.assertEqual(result['totals']['malformed'], 1)
            self.assertEqual(result['duplicates']['unique_vectors'], 1)
            self.assertEqual(result['duplicates']['duplicate_excess_rows'], 1)
            self.assertEqual(result['files'][0]['path'], 'Benign_Final/example.csv')
            self.assertFalse(result['split_frozen'])

    def test_only_observed_aliases_and_legacy_contract_unchanged(self):
        self.assertEqual(official_raw_label('Benign_Final'), 'BenignTraffic')
        self.assertEqual(official_raw_label('BENIGN'), 'BenignTraffic')
        self.assertEqual(official_raw_label('DDOS-PSHACK_FLOOD'), 'DDoS-PSHACK_Flood')
        with self.assertRaises(KeyError):
            official_raw_label('some-benign-file.csv')
        with self.assertRaises(KeyError):
            to_class('BENIGN')

    def test_canonical_zero_and_nonfinite_rejection(self):
        keys = fingerprints([[0, 1], [-0.0, 1], [0, 2]])
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[1], keys[2])
        with self.assertRaises(ValueError):
            fingerprints([[np.nan, 1]])

    def test_cross_file_and_conflicting_labels_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            db = open_index(Path(directory) / 'index.sqlite')
            try:
                key, other = fingerprints([[1, 2], [3, 4]])
                add_vectors(db, [key, key, other], 1, 1, 0)
                add_vectors(db, [key], 2, 2, 1)
                result = summarize_index(db)
                self.assertEqual(result['unique_vectors'], 2)
                self.assertEqual(result['finite_rows'], 4)
                self.assertEqual(result['duplicate_excess_rows'], 2)
                self.assertEqual(result['cross_file_vectors'], 1)
                self.assertEqual(result['class_conflict_vectors'], 1)
                self.assertEqual(result['raw_label_conflict_vectors'], 1)
            finally:
                db.close()


if __name__ == '__main__':
    unittest.main()

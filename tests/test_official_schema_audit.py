"""An incompatible layout must be reported rather than relabelled or padded."""
import tempfile
import unittest
from pathlib import Path

from src.data.official_schema_audit import audit


class SchemaAuditTests(unittest.TestCase):
    def test_malformed_rows_are_counted_not_silently_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.csv').write_text('x,y\n1,2\n3\n4,5\n')
            result = audit(root, ['x', 'y'])
            self.assertEqual(result['rows'], 2)
            self.assertEqual(result['malformed_records'], 1)
            self.assertEqual(result['files'][0]['malformed_records']['examples'][0]['actual_columns'], 1)
            self.assertFalse(result['training_ready'])

    def test_mixed_schemas_and_folder_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.csv').write_text('x,y,label\n1,2,BenignTraffic\n')
            (root / 'attack').mkdir()
            (root / 'attack' / 'b.csv').write_text('x,Time_To_Live\n1,64\n2,32\n')
            result = audit(root, ['x', 'y'])
            self.assertEqual(result['rows'], 3)
            self.assertEqual(result['folder_rows']['attack'], 2)
            profiles = result['schema_profiles']
            self.assertTrue(profiles[0]['exact_training_schema'])
            self.assertFalse(profiles[1]['exact_training_schema'])
            self.assertEqual(profiles[1]['missing_model_features'], ['y'])
            self.assertFalse(result['labels_assigned'])
            self.assertFalse(result['training_ready'])


if __name__ == '__main__':
    unittest.main()

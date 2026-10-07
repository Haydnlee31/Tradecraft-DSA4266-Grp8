"""End-to-end shard exclusions, replay checks and bounded loader parity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.label_map import CLASSES, LABEL_TO_CLASS
from src.data.official_schema_audit import audit
from src.data.official_quality import run
from src.data.official_protocol import freeze
from src.data.official_shards import materialize, verify_file
from src.data.official_loader import OfficialBatches
from src.eval.official_loader_check import check


class OfficialShardTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'raw'
        features = [f'f{i}' for i in range(39)]
        for class_id, name in enumerate(CLASSES):
            label = next(label for label, target in LABEL_TO_CLASS.items() if target == name)
            folder = source / label
            folder.mkdir(parents=True)
            rows = [','.join([str(class_id*1000+j)]*39) for j in range(1, 201)]
            # Repeated identical vectors collapse; a shared conflicting vector is excluded.
            rows += [rows[0], ','.join(['-999']*39), ','.join(['nan']+['1']*38), '1,2']
            (folder / 'a.csv').write_text(','.join(features)+'\n'+'\n'.join(rows)+'\n')
            # A later source file contributes no new vector; cross-file copies
            # must not sneak into a different split or be counted twice.
            (folder / 'b.csv').write_text(','.join(features)+'\n'+rows[0]+'\n')
        receipt = root / 'schema.json'
        receipt.write_text(json.dumps(audit(source, features)))
        run(source, receipt, root / 'audit', cache_mib=16)
        freeze(root / 'audit', root / 'protocol.json')
        return source

    def test_materialize_loader_and_corruption_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            result = materialize(source, root/'audit', root/'protocol.json', root/'shards')
            self.assertTrue(result['exact_digest_disjointness_verified'])
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(sum(sum(v.values()) for v in result['counts_by_split_and_class'].values()), 1600)
            data = OfficialBatches(root/'shards', 'train', batch_size=17)
            batches = list(data.raw_batches())
            x = np.concatenate([b[0] for b in batches])
            y = np.concatenate([b[1] for b in batches])
            scaler = data.fit_scaler()
            reference = StandardScaler().fit(x)
            np.testing.assert_allclose(scaler.mean_, reference.mean_)
            np.testing.assert_allclose(scaler.var_, reference.var_)
            actual = list(data.batches(scaler, shuffle_buffer_rows=51))
            np.testing.assert_allclose(np.concatenate([b[0] for b in actual]), reference.transform(x).astype(np.float32))
            np.testing.assert_array_equal(np.concatenate([b[1] for b in actual]), y)
            first = list(data.batches(scaler, seed=7, shuffle_buffer_rows=51))
            second = list(data.batches(scaler, seed=7, shuffle_buffer_rows=51))
            np.testing.assert_array_equal(np.concatenate([b[0] for b in first]), np.concatenate([b[0] for b in second]))
            self.assertTrue(all(len(b[0]) <= 17 for b in first))
            self.assertEqual(sum(len(b[0]) for b in first), len(x))
            np.testing.assert_array_equal(np.sort(np.concatenate([b[0][:, 0] for b in first])), np.sort(reference.transform(x)[:, 0].astype(np.float32)))
            with self.assertRaisesRegex(ValueError, 'sealed'):
                OfficialBatches(root/'shards', 'test')
            with self.assertRaisesRegex(ValueError, 'training'):
                OfficialBatches(root/'shards', 'val').fit_scaler()
            with self.assertRaises(FileExistsError):
                materialize(source, root/'audit', root/'protocol.json', root/'shards')
            checked = check(root/'shards', root/'loader-check.json', batch_size=17, buffer_rows=51)
            self.assertEqual(checked['status'], 'complete')
            self.assertFalse(checked['test_opened'])
            path = root/'shards'/result['files'][0]['shards'][0]['path']
            with path.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_file(root/'shards', result['files'][0], result['features'], 400000)
            # The streaming loader also rejects a changed shard before using it.
            affected = result['files'][0]['shards'][0]['split']
            with self.assertRaisesRegex(ValueError, 'checksum'):
                list(OfficialBatches(root/'shards', affected, allow_test=True).raw_batches())

    def test_interruption_resume_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'cap'):
                materialize(source, root/'audit', root/'protocol.json', root/'tiny', cap=1)
            calls = 0

            def interrupt(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('interrupted')
                return verify_file(*args)

            with patch('src.data.official_shards.verify_file', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    materialize(source, root/'audit', root/'protocol.json', root/'shards')
            result = materialize(source, root/'audit', root/'protocol.json', root/'shards', resume=True)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['files']), 16)


if __name__ == '__main__':
    unittest.main()

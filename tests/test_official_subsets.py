"""Nested subsets match an independent in-memory rank selection on fixtures."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from src.data.label_map import CLASSES
from src.data.official_loader import OfficialBatches
from src.data.official_shards import materialize
from src.data.official_subsets import build, choose_recipe, quotas, rank_key, SubsetBatches
from src.eval.official_subset_check import check
import tests.test_official_shards as fixtures


class OfficialSubsetTests(unittest.TestCase):
    def test_quotas(self):
        counts = [874352, 9707930, 2985317, 545933, 19699, 10450, 363217, 1967314]
        sizes = [500000, 2000000, 5000000]
        result = quotas(counts, sizes)
        self.assertEqual([sum(row) for row in result], sizes)
        self.assertTrue(all(0 < a <= b <= c for a, b, c in zip(*result)))
        for invalid in ([500000, 500000], [200, 100], [sum(counts)+1], [8]):
            with self.assertRaises(ValueError):
                quotas(counts, invalid)

    def test_nested_selection_order_independence_and_scaler_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixtures.OfficialShardTests().fixture(root)
            materialize(source, root/'audit', root/'protocol.json', root/'parent')
            parent = OfficialBatches(root/'parent', 'train')
            sizes = [80, 400, 1000]
            result = build(root/'parent', root/'subsets', sizes)
            self.assertTrue(result['exact_nesting_verified'])
            self.assertFalse(result['test_opened'])
            recipe = json.loads((root/'subsets'/'recipe.json').read_text())
            parent.shards.reverse()
            reverse = choose_recipe(parent, sizes)
            self.assertEqual(recipe['thresholds_hex'], reverse['thresholds_hex'])
            with self.assertRaisesRegex(ValueError, 'cap'):
                choose_recipe(parent, sizes, boundary_cap=0)
            with self.assertRaises(FileExistsError):
                build(root/'parent', root/'subsets', sizes)
            groups = [[] for _ in CLASSES]
            for shard in parent.shards:
                table = pq.read_table(parent.root/shard['path'], columns=['_digest', '_class_id'])
                for digest, target in zip(table['_digest'].to_pylist(), table['_class_id'].to_pylist()):
                    groups[target].append(digest)
            for group in groups:
                group.sort(key=rank_key)
            previous = set()
            for size, allocation in zip(sizes, recipe['quotas']):
                data = SubsetBatches(root/'parent', root/'subsets', size, batch_size=13)
                actual = set()
                for shard in data.shards:
                    values = pq.read_table(data.root/shard['path'], columns=['_digest'])['_digest'].to_pylist()
                    self.assertTrue(actual.isdisjoint(values))
                    actual.update(values)
                expected = {digest for i, name in enumerate(CLASSES) for digest in groups[i][:allocation[name]]}
                self.assertEqual(actual, expected)
                self.assertLessEqual(previous, actual)
                previous = actual
                raw = list(data.raw_batches())
                x = np.concatenate([item[0] for item in raw])
                y = np.concatenate([item[1] for item in raw])
                self.assertEqual(Counter(CLASSES[i] for i in y), allocation)
                self.assertEqual(data.manifest['counts_by_split_and_class']['train'], allocation)
                scaler = data.fit_scaler()
                reference = StandardScaler().fit(x)
                self.assertEqual(int(scaler.n_samples_seen_), size)
                np.testing.assert_allclose(scaler.mean_, reference.mean_)
                np.testing.assert_allclose(scaler.var_, reference.var_)
                batches = list(data.batches(scaler, shuffle_buffer_rows=39))
                np.testing.assert_allclose(np.concatenate([item[0] for item in batches]), reference.transform(x).astype(np.float32))
            with self.assertRaisesRegex(ValueError, 'not frozen'):
                SubsetBatches(root/'parent', root/'subsets', 123)
            checked = check(root/'parent', root/'subsets', root/'checks')
            self.assertEqual(checked['status'], 'complete')
            self.assertFalse(checked['test_opened'])
            self.assertEqual([v['scaler_fit_rows'] for v in checked['sizes'].values()], sizes)
            # A change to the selection recipe must fail before any training read.
            (root/'subsets'/'recipe.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                SubsetBatches(root/'parent', root/'subsets', 80)


if __name__ == '__main__':
    unittest.main()

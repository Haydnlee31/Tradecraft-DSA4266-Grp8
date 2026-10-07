"""Disk-backed batch parity, exact client coverage and step recovery on CPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.data.label_map import CLASSES
from src.data.official_shards import materialize
from src.data.official_subsets import build
from src.data.official_packed import prepare, PackedData
from src.eval.official_subset_check import check
from src.models.official_streaming import run, partitions, train_epoch, confusion_metrics
from src.models.architectures import MLPClassifier, LIGHT_CONFIG
from src.models.losses import build_criterion
from src.models.train import Trainer
from src.utils.seed import set_seed
from src.eval.official_packed_check import check as check_packed
from src.eval.official_streaming_recovery import pilot
import tests.test_official_shards as fixtures


class OfficialStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        source = fixtures.OfficialShardTests().fixture(cls.root)
        materialize(source, cls.root/'audit', cls.root/'protocol.json', cls.root/'parent')
        build(cls.root/'parent', cls.root/'subsets', [80, 400, 1000])
        check(cls.root/'parent', cls.root/'subsets', cls.root/'checks')
        prepare(cls.root/'parent', cls.root/'subsets', cls.root/'checks', 1000, cls.root/'packed')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_batches_partitions_and_metrics(self):
        data = PackedData(self.root/'packed')
        x, y = data.arrays['train']
        order = np.arange(len(y))
        np.random.default_rng(17).shuffle(order)
        batches = list(data.batches('train', 333, 17))
        self.assertEqual([len(b[1]) for b in batches], [333, 333, 334])
        np.testing.assert_array_equal(np.concatenate([b[0] for b in batches]), x[order])
        for lane in ('iid', 'dirichlet'):
            a, _ = partitions(y, lane, 4, 7, .5)
            b, _ = partitions(y, lane, 4, 7, .5)
            np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(len(y)))
            for left, right in zip(a, b):
                np.testing.assert_array_equal(left, right)
        true = np.arange(40) % 8
        pred = np.arange(40) % 7
        metrics = confusion_metrics(confusion_matrix(true, pred, labels=range(8)))
        self.assertAlmostEqual(metrics['macro_f1'], f1_score(true, pred, labels=range(8), average='macro'))

    def test_streaming_optimizer_matches_existing_in_memory_trainer(self):
        torch.set_num_threads(2)
        data = PackedData(self.root/'packed')
        set_seed(42)
        model = MLPClassifier(39, 8, LIGHT_CONFIG)
        reference = copy.deepcopy(model)
        criterion = build_criterion('sqrt_weighted_ce', data.manifest['train_class_counts'])
        trainer = Trainer(reference, copy.deepcopy(criterion), device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
        batches = list(data.batches('train', 128, 5))
        set_seed(123)
        observed, count, steps = train_epoch(model, optimizer, criterion, iter(batches), 'cpu')
        set_seed(123)
        expected, _ = trainer._run_epoch([(torch.from_numpy(x), torch.from_numpy(y)) for x, y in batches], train=True)
        self.assertEqual(observed, expected)
        self.assertEqual(count, 1000)
        self.assertEqual(steps, len(batches))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, reference.state_dict()[name]), name)

    def test_packed_check_and_corruption_guard(self):
        result = check_packed(self.root/'packed', self.root/'packed-check.json', clients=4)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['model_trained'])
        self.assertFalse(result['test_opened'])
        shutil.copytree(self.root/'packed', self.root/'tampered')
        path = self.root/'tampered'/'train_x.npy'
        with path.open('ab') as stream:
            stream.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            PackedData(self.root/'tampered')

    def test_recovery_pilot(self):
        result = pilot(self.root/'packed', self.root/'pilot', clients=4, batch_size=128)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['test_evaluated'])
        self.assertTrue(all(v['recovery_exact'] for v in result['lanes'].values()))

    def test_all_lanes_recover_exactly_and_reject_changed_settings(self):
        for lane in ('light', 'heavy', 'iid', 'dirichlet'):
            options = dict(data_root=self.root/'packed', lane=lane, epochs=2, clients=4, batch_size=128)
            full = run(output=self.root/f'{lane}-full', **options)
            run(output=self.root/f'{lane}-resumed', stop_after=1, **options)
            with self.assertRaisesRegex(ValueError, 'changed'):
                run(output=self.root/f'{lane}-resumed', resume=True, **(options | {'lr': .002}))
            resumed = run(output=self.root/f'{lane}-resumed', resume=True, **options)
            self.assertEqual(full['validation_metrics'], resumed['validation_metrics'])
            self.assertIsNone(resumed['test_metrics'])
            self.assertEqual(resumed['examples_processed'], 2000)
            for a, b in zip(full['history'], resumed['history']):
                self.assertEqual({k: v for k, v in a.items() if k != 'elapsed_seconds'},
                                 {k: v for k, v in b.items() if k != 'elapsed_seconds'})
            saved = [torch.load(self.root/f'{lane}-{mode}'/'last.pt', weights_only=True) for mode in ('full', 'resumed')]
            for key in saved[0]['model']:
                self.assertTrue(torch.equal(saved[0]['model'][key], saved[1]['model'][key]), (lane, key))
            self.assertEqual(saved[0]['optimizer']['param_groups'], saved[1]['optimizer']['param_groups'])
            for key in saved[0]['optimizer']['state']:
                for field, value in saved[0]['optimizer']['state'][key].items():
                    other = saved[1]['optimizer']['state'][key][field]
                    self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)


if __name__ == '__main__':
    unittest.main()

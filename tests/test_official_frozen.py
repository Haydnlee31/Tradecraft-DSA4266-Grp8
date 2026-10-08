"""Frozen assignments must preserve training and bind recovery to their plan."""
import json
import shutil
import unittest

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.federated.official_frozen import load_frozen
from src.models.official_streaming import run, partitions
from src.models.research import atomic_json
from src.eval.official_streaming_recovery import same
import tests.test_official_streaming as fixtures


class FrozenPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse only the miniature packed-data fixture, not the other test cases.
        fixtures.OfficialStreamingTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def make_plan(self, name):
        data = PackedData(self.root/'packed')
        parts, _ = partitions(data.arrays['train'][1], 'dirichlet', 4, 7, .5)
        root = self.root/name
        root.mkdir()
        for cohort in ('500k', '2m'):
            np.savez_compressed(root/f'{cohort}-assignments.npz', **{f'client_{i}': p for i, p in enumerate(parts)})
        np.save(root/'small_to_large.npy', np.arange(data.manifest['train_rows']))
        # Cohort labels reuse the production schema, but rows are tiny fixtures.
        counts = [np.bincount(data.arrays['train'][1][p], minlength=8).tolist() for p in parts]
        plan = {'status': 'prepared_not_trained', 'classes': CLASSES, 'test_opened': False,
                'class_support_preserved': True, 'anchor_counts': counts, 'expanded_counts': counts,
                'packed_manifest_sha256': {'500k': sha256(data.root/'manifest.json'), '2m': '0'*64},
                'files': {p.name: sha256(p) for p in root.iterdir()}}
        atomic_json(plan, root/'plan.json')
        return root, data, plan

    def test_native_bridge_and_recovery_exact(self):
        root, data, _ = self.make_plan('bridge-plan')
        options = dict(data_root=data.root, lane='dirichlet', clients=4, epochs=2,
                       normalization='layer', batch_size=128)
        native, full, resumed = [self.root/name for name in ('native', 'frozen', 'recovered')]
        run(output=native, **options)
        run(output=full, partition_root=root, **options)
        run(output=resumed, partition_root=root, stop_after=1, **options)
        # The default path must not silently resume a frozen-partition experiment.
        with self.assertRaisesRegex(ValueError, 'changed'):
            run(output=resumed, resume=True, **options)
        run(output=resumed, partition_root=root, resume=True, **options)
        states = [torch.load(p/'last.pt', weights_only=True) for p in (native, full, resumed)]
        for observed in states[1:]:
            for k in ('model', 'optimizer', 'rng', 'best', 'best_score', 'best_step'):
                self.assertTrue(same(states[0][k], observed[k]), k)
            for a, b in zip(states[0]['history'], observed['history']):
                self.assertEqual({k:v for k,v in a.items() if k!='elapsed_seconds'},
                                 {k:v for k,v in b.items() if k!='elapsed_seconds'})
        self.assertNotIn('partition_control', json.loads((native/'environment.json').read_text()))
        self.assertEqual(json.loads((full/'environment.json').read_text())['partition_control']['cohort'], '500k')

    def test_plan_change_blocks_resume_without_overwriting(self):
        root, data, plan = self.make_plan('changed-plan')
        out = self.root/'changed-resume'
        options = dict(data_root=data.root, output=out, lane='dirichlet', clients=4,
                       epochs=2, batch_size=128, partition_root=root, normalization='layer')
        run(**options, stop_after=1)
        digest = sha256(out/'last.pt')
        plan['note'] = 'Changed after checkpoint'
        atomic_json(plan, root/'plan.json')
        with self.assertRaisesRegex(ValueError, 'changed'):
            run(**options, resume=True)
        self.assertEqual(sha256(out/'last.pt'), digest)

    def test_loader_rejects_cohort_counts_clients_and_corruption(self):
        root, data, plan = self.make_plan('invalid-plan')
        with self.assertRaisesRegex(ValueError, 'client count'):
            load_frozen(root, data, 5)
        plan['packed_manifest_sha256']['500k'] = '1'*64
        atomic_json(plan, root/'plan.json')
        with self.assertRaisesRegex(ValueError, 'different cohort'):
            load_frozen(root, data, 4)
        plan['packed_manifest_sha256']['500k'] = sha256(data.root/'manifest.json')
        plan['anchor_counts'][0][0] += 1
        atomic_json(plan, root/'plan.json')
        with self.assertRaisesRegex(ValueError, 'class counts'):
            load_frozen(root, data, 4)
        with (root/'500k-assignments.npz').open('ab') as f:
            f.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_frozen(root, data, 4)

    def test_duplicate_rows_rejected_even_with_updated_checksum(self):
        root, data, plan = self.make_plan('duplicate-plan')
        with np.load(root/'500k-assignments.npz') as f:
            parts = {key: f[key].copy() for key in f.files}
        parts['client_0'][1] = parts['client_0'][0]
        np.savez_compressed(root/'500k-assignments.npz', **parts)
        plan['files']['500k-assignments.npz'] = sha256(root/'500k-assignments.npz')
        atomic_json(plan, root/'plan.json')
        with self.assertRaisesRegex(ValueError, 'sorted unique'):
            load_frozen(root, data, 4)

    def test_relocation_and_scope(self):
        root, data, _ = self.make_plan('portable-plan')
        copy = self.root/'portable-copy'
        shutil.copytree(root, copy)
        a, meta_a = load_frozen(root, data, 4)
        b, meta_b = load_frozen(copy, data, 4)
        self.assertEqual(meta_a, meta_b)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))
        for overrides in ({'lane': 'iid'}, {'lane': 'light'}, {'partition_seed': 17}, {'local_max_batches': 1}):
            options = dict(data_root=data.root, output=self.root/'invalid-scope', lane='dirichlet',
                           partition_root=root, clients=4) | overrides
            with self.assertRaisesRegex(ValueError, 'Frozen control requires'):
                run(**options)
        self.assertFalse((self.root/'invalid-scope').exists())

    def test_pilot_wrapper(self):
        from src.eval.official_frozen_recovery import pilot
        root, data, _ = self.make_plan('wrapper-plan')
        receipt = pilot(data.root, root, self.root/'wrapper-output', clients=4, batch_size=128)
        self.assertEqual(receipt['status'], 'complete')
        self.assertTrue(receipt['recovery_exact'])
        self.assertTrue(receipt['same_code_anchor_bridge_exact'])
        self.assertFalse(receipt['test_evaluated'])


if __name__ == '__main__':
    unittest.main()

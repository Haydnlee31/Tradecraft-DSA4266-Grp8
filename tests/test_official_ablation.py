"""Synthetic bridge, mask, paired initialization and recovery gates on CPU."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.data.official_ablation_panel import write_json
from src.eval.official_streaming_recovery import same
from src.models import official_ablation as ablation
from src.models.official_streaming import run, partitions, evaluate
from src.models.architectures import MLPClassifier, config_for_variant
from tests.test_official_ablation_panel import fixture, make_panel, FEATURES


def frozen_partition(root, packed, arrays):
    directory = root/'frozen'
    directory.mkdir()
    parts, _ = partitions(arrays['train'][1], 'dirichlet', 4, 7, .5)
    for cohort in ('500k', '2m'):
        np.savez_compressed(directory/f'{cohort}-assignments.npz', **{f'client_{i}': p for i, p in enumerate(parts)})
    np.save(directory/'small_to_large.npy', np.arange(len(arrays['train'][1]), dtype=np.int64))
    counts = [np.bincount(arrays['train'][1][p], minlength=len(CLASSES)).tolist() for p in parts]
    write_json(directory/'plan.json', dict(status='prepared_not_trained', classes=CLASSES,
        test_opened=False, class_support_preserved=True, anchor_counts=counts, expanded_counts=counts,
        packed_manifest_sha256={'500k': sha256(packed/'manifest.json'), '2m': '0'*64},
        files={p.name: sha256(p) for p in directory.iterdir()}))
    return directory


def deterministic_history(history, bridge=False):
    omit = {'elapsed_seconds'}
    if bridge:
        omit |= {'panel_validation_metrics', 'panel_val_loss'}
    return [{k: v for k, v in row.items() if k not in omit} for row in history]


class AblationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.packed, cls.arrays = fixture(cls.root)
        cls.panel, _ = make_panel(cls.root, cls.packed, cls.arrays)
        cls.frozen = frozen_partition(cls.root, cls.packed, cls.arrays)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def options(self, lane):
        return dict(data_root=self.packed, lane=lane, clients=4, epochs=2, batch_size=32,
                    normalization='layer', partition_root=self.frozen if lane == 'dirichlet' else None)

    def test_transform_only_selected_columns_no_alias_or_source_mutation(self):
        source = self.arrays['train'][0]
        original = source.copy()
        for arm, names in ablation.ARMS.items():
            masked = ablation.transform_inputs(source, FEATURES, arm)
            self.assertFalse(np.shares_memory(masked, source))
            for j, name in enumerate(FEATURES):
                np.testing.assert_array_equal(masked[:, j], np.zeros(len(source)) if name in names else source[:, j])
        np.testing.assert_array_equal(source, original)
        with self.assertRaises(ValueError):
            ablation.transform_inputs(source, FEATURES, 'other')
        with self.assertRaises(ValueError):
            ablation.transform_inputs(source.astype(np.float64), FEATURES, 'full39')

    def test_dual_view_uses_one_forward_and_exact_legacy_full_metrics(self):
        data = PackedData(self.packed)
        try:
            masked = ablation.AblationData(data, 'number_total_masked', self.panel)
            model = MLPClassifier(39, 8, config_for_variant('light'))
            expected = evaluate(model, masked.batches('val', 32), 'cpu')
            calls = []
            hook = model.register_forward_hook(lambda m, inputs, result: calls.append(result.detach().clone()))
            state = torch.random.get_rng_state().clone()
            full_loss, full_metrics, _, subset_metrics = ablation.evaluate_views(
                model, masked.batches('val', 32), 'cpu', masked.panel_mask)
            hook.remove()
            self.assertEqual((full_loss, full_metrics), expected)
            self.assertEqual(len(calls), 3)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            predictions = torch.cat(calls).argmax(1).numpy()
            take = masked.panel_mask
            expected_cm = np.bincount(data.arrays['val'][1][take]*8+predictions[take], minlength=64).reshape(8,8)
            self.assertEqual(subset_metrics['confusion_matrix'], expected_cm.tolist())
            self.assertEqual(sum(v['support'] for v in subset_metrics['per_class'].values()), 64)
        finally:
            for pair in data.arrays.values():
                for a in pair:
                    a._mmap.close()

    def test_all_lanes_empty_mask_bridge_preserves_weights_optimizer_rng_and_history(self):
        for lane in ('light', 'iid', 'dirichlet'):
            baseline, masked = self.root/f'base-{lane}', self.root/f'empty-{lane}'
            a = run(output=baseline, **self.options(lane))
            b = run(output=masked, ablation_arm='full39', validation_panel=self.panel, **self.options(lane))
            left, right = [torch.load(p/'last.pt', weights_only=True) for p in (baseline, masked)]
            for key in ('model', 'optimizer', 'rng', 'best', 'best_score', 'best_step'):
                self.assertTrue(same(left[key], right[key]), (lane, key))
            self.assertEqual(deterministic_history(a['history']), deterministic_history(b['history'], bridge=True))
            self.assertEqual(a['final_validation_metrics'], b['final_validation_metrics'])
            self.assertNotIn('input_transform', left)
            self.assertEqual(right['input_transform']['zero_after_scaling'], [])

    def test_three_arms_three_lanes_recover_exactly_and_share_initialization_work_and_parts(self):
        original = {p.name: sha256(p) for p in self.packed.iterdir()}
        for lane in ('light', 'iid', 'dirichlet'):
            initial, work, partition_hash = None, None, None
            for arm in ablation.ARMS:
                a, b = self.root/f'{lane}-{arm}-full', self.root/f'{lane}-{arm}-resume'
                options = dict(**self.options(lane), ablation_arm=arm, validation_panel=self.panel)
                full = run(output=a, **options)
                run(output=b, stop_after=1, **options)
                resumed = run(output=b, resume=True, **options)
                left, right = [torch.load(p/'last.pt', weights_only=True) for p in (a,b)]
                for key in ('model', 'optimizer', 'rng', 'best', 'best_score', 'best_step', 'input_transform'):
                    self.assertTrue(same(left[key], right[key]), (lane, arm, key))
                self.assertEqual(deterministic_history(full['history']), deterministic_history(resumed['history']))
                self.assertEqual(full['primary_validation_metrics'], full['history'][-1]['panel_validation_metrics'])
                self.assertEqual(full['primary_checkpoint'], 'last.pt')
                self.assertIsNone(full['test_metrics'])
                current = json.loads((a/'initialization.json').read_text())
                self.assertEqual(current, json.loads((b/'initialization.json').read_text()))
                counters = (full['examples_processed'], full['optimizer_steps'])
                self.assertEqual(counters[0], 320)
                if initial is not None:
                    self.assertEqual(current, initial)
                    self.assertEqual(counters, work)
                initial, work = current, counters
                if lane != 'light':
                    digest = sha256(a/'assignments.npz')
                    self.assertEqual(digest, sha256(b/'assignments.npz'))
                    if partition_hash is not None:
                        self.assertEqual(digest, partition_hash)
                    partition_hash = digest
                best = torch.load(a/'best.pt', weights_only=True)
                self.assertEqual(best['input_transform']['zero_after_scaling'], ablation.ARMS[arm])
        self.assertEqual(original, {p.name: sha256(p) for p in self.packed.iterdir()})

    def test_resume_rejects_arm_panel_and_initialization_changes_without_overwrite(self):
        import shutil
        out = self.root/'guarded'
        options = dict(**self.options('light'), ablation_arm='number_masked', validation_panel=self.panel)
        run(output=out, stop_after=1, **options)
        before = sha256(out/'last.pt')
        with self.assertRaisesRegex(ValueError, 'changed'):
            run(output=out, resume=True, **(options | {'ablation_arm': 'number_total_masked'}))
        copy = self.root/'panel-copy'
        shutil.copytree(self.panel, copy)
        receipt = json.loads((copy/'receipt.json').read_text())
        receipt['note'] = 'changed but still structurally valid'
        write_json(copy/'receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'changed'):
            run(output=out, resume=True, **(options | {'validation_panel': copy}))
        write_json(out/'initialization.json', {'model_state_sha256': 'bad'})
        with self.assertRaisesRegex(ValueError, 'Initial model'):
            run(output=out, resume=True, **options)
        self.assertEqual(sha256(out/'last.pt'), before)

    def test_rejects_partial_requests_heavy_native_noniid_and_batch_cap(self):
        for overrides in ({'ablation_arm': 'full39'}, {'validation_panel': self.panel},
                          {'lane': 'heavy'}, {'lane': 'dirichlet'}, {'local_max_batches': 1},
                          {'normalization': 'batch'}, {'ablation_arm': 'other'}):
            options = dict(data_root=self.packed, output=self.root/'invalid', lane='light',
                           ablation_arm='full39', validation_panel=self.panel, normalization='layer')
            if len(overrides) == 1 and set(overrides) in ({'ablation_arm'}, {'validation_panel'}) and 'other' not in overrides.values():
                options['validation_panel' if 'ablation_arm' in overrides else 'ablation_arm'] = None
            with self.assertRaises(ValueError):
                run(**(options | overrides))
            self.assertFalse((self.root/'invalid').exists())

    def test_pilot_wrapper_work_budget_and_reviewed_panel_gate(self):
        from src.eval.official_ablation_recovery import pilot
        out = self.root/'pilot'
        options = dict(data=self.packed, panel=self.panel, partition_root=self.frozen,
                       reviewed_panel_sha256=sha256(self.panel/'receipt.json'), clients=4, batch_size=32)
        with self.assertRaisesRegex(ValueError, 'reviewed'):
            pilot(output=out, **(options | {'reviewed_panel_sha256': 'bad'}))
        self.assertFalse(out.exists())
        r = pilot(output=out, **options)
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['total_examples_processed'], 42*160)
        self.assertFalse(r['test_evaluated'])
        self.assertFalse(r['comparison_training_launched'])
        for lane in r['lanes'].values():
            self.assertTrue(lane['default_path_bridge_exact'])
            self.assertEqual(set(lane['arms']), set(ablation.ARMS))
            self.assertTrue(all(arm['recovery_exact'] for arm in lane['arms'].values()))
        with self.assertRaises(FileExistsError):
            pilot(output=out, **options)


if __name__ == '__main__':
    unittest.main()

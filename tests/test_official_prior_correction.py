"""Fixed training priors, exact replay and inference-only split boundaries."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import official_diversity as diversity
from src.eval import official_prior_correction as correction
from tests import test_official_diversity as fixtures


class PriorCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DiversityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def references(self):
        # Deliberately nonuniform training counts, all still large enough to
        # supply the existing eight-row-per-class synthetic fitting panel.
        base = self.fixture.fixture.training
        counts = dict(zip(CLASSES, [20, 16, 12, 12, 10, 10, 8, 8]))
        base.y = np.repeat(np.arange(len(CLASSES), dtype=np.int64), list(counts.values()))
        np.save(self.root/'train_y.npy', base.y)
        base.manifest['train_class_counts'] = counts
        base.manifest['files']['train_y.npy'] = sha256(self.root/'train_y.npy')
        base.write_manifest()
        base.plan['packed_manifest_sha256'] = sha256(self.root/'manifest.json')
        ref, training, diversity_plan, _, _ = self.fixture.references()
        ref_root = self.root/'diversity-reference'
        reference = diversity.diagnose(self.root, ref, training, diversity_plan, ref_root)
        plan = json.loads((Path(__file__).parents[1]/'reports/full_data_extension/prior-correction-plan.json').read_text())
        plan.update(reference_receipt_sha256=sha256(ref_root/'receipt.json'),
                    reference_plan_sha256=reference['plan_sha256'], seeds=[7], epochs=4,
                    evaluation_batch_size=17, validation_rows=96,
                    validation_class_counts=self.fixture.fixture.counts)
        plan_path = self.root/'prior-plan.json'
        plan_path.write_text(json.dumps(plan))
        return ref_root, plan_path, reference, plan

    def test_prior_counts_are_ordered_canonical_and_not_smoothed(self):
        counts = {c: i+1 for i, c in reversed(list(enumerate(CLASSES)))}
        prior, offsets = correction.prior_offsets(counts)
        np.testing.assert_array_equal(prior, np.arange(1, 9)/36)
        np.testing.assert_array_equal(offsets, np.log(prior*8).astype('float32'))
        for value in (0, -1, 1.5, True, float('nan')):
            bad = dict(counts)
            bad[CLASSES[0]] = value
            with self.assertRaises(ValueError):
                correction.prior_offsets(bad)
        with self.assertRaises(ValueError):
            correction.prior_offsets({'Benign': 10})

    def test_uniform_priors_are_exact_identity_including_ties(self):
        prior, offsets = correction.prior_offsets(dict.fromkeys(CLASSES, 10))
        np.testing.assert_array_equal(prior, np.full(8, .125))
        np.testing.assert_array_equal(offsets, np.zeros(8, dtype='float32'))
        logits = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.], [0., 1e-7, 0., 0., 0., 0., 0., 0.]])
        model = correction.PriorAdjustedModel(torch.nn.Identity(), offsets)
        self.assertTrue(torch.equal(model(logits), logits))
        self.assertTrue(torch.equal(model(logits).argmax(1), torch.tensor([0, 1])))

    def test_logit_rule_matches_posterior_reweighting_direction(self):
        counts = dict(zip(CLASSES, [100, 50, 25, 20, 2, 1, 15, 10]))
        prior, offsets = correction.prior_offsets(counts)
        logits = torch.tensor([[0., 0., 0., 0., 2., 2., 0., 0.]], dtype=torch.float32)
        adjusted = correction.PriorAdjustedModel(torch.nn.Identity(), offsets)(logits)
        expected = logits.double().softmax(1)*torch.from_numpy(prior/.125)
        expected /= expected.sum(1, keepdim=True)
        torch.testing.assert_close(adjusted.double().softmax(1), expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(logits.argmax(1).item(), CLASSES.index('Web-based'))
        self.assertEqual(adjusted.argmax(1).item(), CLASSES.index('Benign'))
        for invalid in ([0.]*7, [float('nan')]*8, [[0.]*8]):
            with self.assertRaises(ValueError):
                correction.PriorAdjustedModel(torch.nn.Identity(), invalid)

    def test_complete_repeat_never_loads_training_or_test_and_never_fits(self):
        ref, plan_path, reference, _ = self.references()
        real_load, real_hash = np.load, sha256
        def val_only(path, **kwargs):
            self.assertIn(Path(path).name, ('val_x.npy', 'val_y.npy'))
            self.assertEqual(kwargs['mmap_mode'], 'r')
            return real_load(path, **kwargs)
        def no_forbidden_hash(path):
            self.assertNotIn(Path(path).name, ('train_x.npy', 'train_y.npy', 'test_x.npy', 'test_y.npy'))
            return real_hash(path)
        with patch('src.eval.official_prior_correction.np.load', side_effect=val_only), \
             patch('src.eval.official_prior_correction.sha256', side_effect=no_forbidden_hash), \
             patch('torch.optim.Adam', side_effect=AssertionError('No optimizer allowed')), \
             patch('src.eval.official_diversity.fit_condition', side_effect=AssertionError('No training allowed')):
            a = correction.diagnose(self.root, ref, plan_path, self.root/'prior-a')
        b = correction.diagnose(self.root, ref, plan_path, self.root/'prior-b')
        self.assertEqual(a, b)
        self.assertEqual(a['status'], 'complete')
        self.assertTrue(a['all_raw_bridges_exact'])
        for key in ('training_arrays_opened', 'test_opened', 'model_trained', 'model_promoted', 'independent_assessment'):
            self.assertFalse(a[key])
        self.assertEqual(len(a['raw_results']), 2)
        self.assertEqual(len(a['corrected_results']), 2)
        for r, old, adjusted in zip(a['raw_results'], reference['validation_results'], a['corrected_results']):
            self.assertEqual(r['validation_metrics'], old['validation_metrics'])
            self.assertEqual(r['checkpoint_sha256'], adjusted['checkpoint_sha256'])
            self.assertEqual(r['final_state_sha256'], adjusted['final_state_sha256'])
            self.assertEqual(sum(c['support'] for c in adjusted['validation_metrics']['per_class'].values()), 96)
        self.assertNotEqual(a['raw_results'][0]['validation_metrics'], a['corrected_results'][0]['validation_metrics'])
        with self.assertRaises(FileExistsError):
            correction.diagnose(self.root, ref, plan_path, self.root/'prior-a')

    def test_last_raw_failure_blocks_every_correction(self):
        ref, plan_path, reference, plan = self.references()
        reference['validation_results'][-1]['validation_cross_entropy'] = -1
        (ref/'receipt.json').write_text(json.dumps(reference))
        plan['reference_receipt_sha256'] = sha256(ref/'receipt.json')
        plan_path.write_text(json.dumps(plan))
        with patch.object(correction, 'PriorAdjustedModel') as adjust:
            with self.assertRaisesRegex(ValueError, 'Raw bridge failed'):
                correction.diagnose(self.root, ref, plan_path, self.root/'failed')
        adjust.assert_not_called()
        receipt = json.loads((self.root/'failed/receipt.json').read_text())
        self.assertEqual(receipt['corrected_results'], [])
        self.assertFalse(receipt['all_raw_bridges_exact'])

    def test_corrupt_checkpoint_fails_before_validation_load(self):
        ref, plan_path, reference, _ = self.references()
        checkpoint = ref/reference['fits'][-1]['checkpoint']
        checkpoint.write_bytes(checkpoint.read_bytes()+b'corrupt')
        with patch.object(correction, 'load_validation') as loading:
            with self.assertRaisesRegex(ValueError, 'Checkpoint checksum'):
                correction.diagnose(self.root, ref, plan_path, self.root/'failed')
        loading.assert_not_called()

    def test_checkpoint_path_and_metadata_are_checked(self):
        ref, _, reference, _ = self.references()
        entry = copy.deepcopy(reference['fits'][0])
        manifest = json.loads((self.root/'manifest.json').read_text())
        entry['checkpoint'] = '../escaped.pt'
        with self.assertRaisesRegex(ValueError, 'checkpoint path'):
            correction.load_model(ref, entry, reference, manifest)
        entry = copy.deepcopy(reference['fits'][0])
        path = ref/entry['checkpoint']
        saved = torch.load(path, weights_only=True)
        saved['condition'] = 'wrong'
        torch.save(saved, path)
        entry['checkpoint_sha256'] = sha256(path)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            correction.load_model(ref, entry, reference, manifest)

    def test_bad_reference_blocks_model_and_data_loading(self):
        ref, plan_path, _, plan = self.references()
        plan['reference_receipt_sha256'] = 'wrong'
        plan_path.write_text(json.dumps(plan))
        with patch.object(correction, 'load_model') as model, patch.object(correction, 'load_validation') as data:
            with self.assertRaisesRegex(ValueError, 'Reference checksum'):
                correction.diagnose(self.root, ref, plan_path, self.root/'failed')
        model.assert_not_called()
        data.assert_not_called()

    def test_scope_nonuniform_exposure_and_wrong_counts_are_rejected(self):
        _, _, reference, plan = self.references()
        manifest = json.loads((self.root/'manifest.json').read_text())
        for key, value in (('coefficient', .5), ('prior_source', 'validation'), ('model_trained', True),
                           ('test_opened', True), ('model_promoted', True), ('seeds', [7, 17]),
                           ('require_all_raw_bridges_before_correction', False)):
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError):
                correction.check_protocol(bad, reference, manifest)
        bad = copy.deepcopy(reference)
        bad['fits'][0]['exposure']['per_class']['Benign']['examples_processed'] += 1
        with self.assertRaisesRegex(ValueError, 'Uniform training exposure'):
            correction.check_protocol(plan, bad, manifest)
        bad = copy.deepcopy(manifest)
        bad['train_class_counts']['Benign'] += 1
        with self.assertRaisesRegex(ValueError, 'sum to'):
            correction.check_protocol(plan, reference, bad)
        bad = copy.deepcopy(reference)
        bad['fits'][0]['training_fit']['class_weights'][0] = 2.
        with self.assertRaisesRegex(ValueError, 'Uniform training exposure'):
            correction.check_protocol(plan, bad, manifest)


if __name__ == '__main__':
    unittest.main()

"""Exercise replay bookkeeping with synthetic predictions, never Laya itself."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from scripts import official39_laya_repeatability as r


class LayaRepeatabilityTests(unittest.TestCase):
    def test_fixed_subset_review_and_changed_prediction_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, repeat = root/'prepared', root/'repeat'
            prepared.mkdir()
            labels = np.repeat(np.arange(8), 128)
            np.save(prepared/'labels.npy', labels, allow_pickle=False)
            request = dict(plan=r.read(r.pilot.PLAN), checkpoint_files={'model': 'fixture'},
                states={arm: [f'fixture {i}' for i in range(1024)] for arm in r.pilot.rep.SERIALIZERS})
            r.save(prepared/'inputs.json', request)
            baseline = r.pilot.old.prior.prior.scored(labels, labels.astype(np.uint8), np.ones(8, dtype=int)*128)
            ident = {'fixture': True}
            r.save(prepared/'preparation.json', dict(identity=ident, baseline=baseline,
                calibration_population_counts=[128]*8,
                files={n: r.digest(prepared/n) for n in ['labels.npy', 'inputs.json']}))
            with patch.object(r.pilot, 'identity', return_value=ident):
                r.prepare(prepared, repeat)
                indices = np.load(repeat/'indices.npy', allow_pickle=False)
                np.testing.assert_array_equal(np.bincount(labels[indices], minlength=8), [16]*8)
                full = dict(status='complete', test_opened=False, fine_tuned=False,
                    input_sha256=r.digest(prepared/'inputs.json'), cpu_fallback_count=0,
                    checkpoint_files=request['checkpoint_files'], runtime={'fixture': True},
                    device='mps', inference_dtype='torch.float32',
                    source_sha256=r.digest(r.ROOT/'scripts/official39_laya_inference.py'),
                    arms={arm: dict(rows=1024, predictions=[0]*1024,
                         rounded_probabilities=[[1., 0., 0., 0., 0., 0., 0., 0.]]*1024)
                         for arm in r.pilot.rep.SERIALIZERS})
                second = copy.deepcopy(full)
                second['input_sha256'] = r.digest(repeat/'inputs.json')
                for arm in second['arms'].values():
                    arm['rows'] = 128
                    for key in ['predictions', 'rounded_probabilities']:
                        arm[key] = [arm[key][i] for i in indices]
                r.save(root/'first.json', full)
                r.save(root/'second.json', second)
                r.review(prepared, repeat, root/'first.json', root/'second.json', root/'result.json')
                result = r.read(root/'result.json')
                self.assertTrue(result['pivot_authorized'])
                self.assertTrue(result['repeat_subset_exact'])
                self.assertFalse(result['full_replay_performed'])
                second['arms']['values']['predictions'][0] = 1
                r.save(root/'bad.json', second)
                with self.assertRaisesRegex(ValueError, 'Repeat subset predictions changed'):
                    r.review(prepared, repeat, root/'first.json', root/'bad.json', root/'rejected.json')
                self.assertFalse((root/'rejected.json').exists())


if __name__ == '__main__':
    unittest.main()

"""Synthetic receipt checks; no real training data or original test access."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import official39_contrastive as run


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ident = {'synthetic': True}
        y = np.arange(409918, dtype=np.int64) % 8
        pred = np.zeros(len(y), np.uint8)
        pop = [100000]*8
        files = {}
        for name, a in [('fit-rows.npy', np.arange(16)), ('calibration-rows.npy', np.arange(len(y))+16),
                        ('fit-groups.npy', np.arange(16)), ('calibration-labels.npy', y)]:
            files[name] = run.previous.save_array(self.root, name, a)
        run.frozen.save(self.root/'edges.json', [[0, 1]])
        files['edges.json'] = run.digest(self.root/'edges.json')
        recipe = dict(identity=self.ident, files=copy.deepcopy(files), fit_population_counts=pop,
                      calibration_supports=dict(zip(run.prior.CLASSES, np.bincount(y).tolist())))
        models = []
        for arm, coefficient in zip(run.ARMS, [0., .1]):
            name = arm+'.pt'
            (self.root/name).write_bytes(b'synthetic-checksum-fixture-not-a-model')
            files[name] = run.digest(self.root/name)
            models.append(dict(arm=arm, coefficient=coefficient, file=name, sha256=files[name],
                settings=run.core.SETTINGS, examples_processed=16220160, optimizer_steps=31680,
                sampled_class_counts=[2027520]*8, eligible_contrastive_anchors=16220160,
                excluded_contrastive_anchors=0, initial_state_sha256='same', batch_order_sha256='same',
                class_weights=[1.]*8))
        names = run.ARMS+run.COMPARATORS
        for name in names:
            files[name+'-calibration.npy'] = run.previous.save_array(self.root, name+'-calibration.npy', pred)
        score = run.prior.old.prior.prior.scored(y, pred, pop)
        self.r = dict(status='complete', recipe=recipe, files=files, models=models,
            scores={n: copy.deepcopy(score) for n in names}, examples_processed=32440320, optimizer_steps=63360,
            continuation_checks={'tree': False, run.ARMS[0]: False}, continue_to_confirmation=False,
            test_opened=False, test_evaluated=False, audit_scored=False, independent_confirmation=False,
            deployment_authorized=False)
        run.frozen.save(self.root/'frozen-recipe.json', recipe)
        run.frozen.save(self.root/'models-frozen.json', models)
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        run.frozen.save(self.root/'receipt.json', self.r)

    def verify(self):
        with patch.object(run, 'identity', return_value=self.ident):
            return run.verify(self.root)

    def test_valid_receipt(self):
        self.assertEqual(self.verify(), self.r)

    def test_edited_prediction_even_with_new_checksum_is_rejected(self):
        path = self.root/(run.ARMS[1]+'-calibration.npy')
        a = np.load(path, allow_pickle=False)
        a[0] = 1
        np.save(path, a, allow_pickle=False)
        self.r['files'][path.name] = run.digest(path)
        self.save()
        with self.assertRaisesRegex(ValueError, 'Scores changed'):
            self.verify()

    def test_invented_promotion_is_rejected(self):
        self.r['continue_to_confirmation'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'Continuation decision changed'):
            self.verify()

    def test_wrong_sampling_control_is_rejected(self):
        self.r['models'][1]['batch_order_sha256'] = 'changed'
        run.frozen.save(self.root/'models-frozen.json', self.r['models'])
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unmatched control'):
            self.verify()

    def test_budget_or_test_access_is_rejected(self):
        for key, value in [('optimizer_steps', 63359), ('test_opened', True)]:
            previous = self.r[key]
            self.r[key] = value
            self.save()
            with self.assertRaisesRegex(ValueError, 'Invalid contrastive receipt'):
                self.verify()
            self.r[key] = previous

    def test_source_change_is_rejected(self):
        with patch.object(run, 'identity', return_value={'synthetic': False}):
            with self.assertRaisesRegex(ValueError, 'Invalid contrastive receipt'):
                run.verify(self.root)


if __name__ == '__main__':
    unittest.main()

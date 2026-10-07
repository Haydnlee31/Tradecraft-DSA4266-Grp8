"""Small deterministic fixtures for the observational client diagnostic."""
import copy
import unittest

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_packed import PackedData
from src.eval.official_client_diagnostic import branch, probe_rows, reduction_audit
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_streaming import train_epoch, evaluate
from src.utils.seed import set_seed, client_seed


class ClientDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(7)
        # In-memory tiny fixture using the real batching method; production uses
        # verified memory-mapped arrays, not an in-memory full dataset.
        self.data = object.__new__(PackedData)
        y = np.arange(96, dtype=np.int64) % 8
        self.data.arrays = {s: (rng.normal(size=(96, 39)).astype('float32'), y.copy())
                            for s in ('train', 'val')}
        self.data.manifest = {'train_class_counts': dict.fromkeys(CLASSES, 12)}
        self.parts = [np.arange(0, 32), np.arange(32, 96)]
        self.settings = {'seed': 7, 'batch_size': 8, 'loss': 'sqrt_weighted_ce',
                         'lr': .001, 'weight_decay': 1e-5}
        set_seed(7)
        self.model = MLPClassifier(39, 8, MLPConfig('probe', (8,), normalization='layer'))

    def test_single_class_weight_cancels_in_loss_and_gradient(self):
        logits = torch.randn(6, 8, requires_grad=True)
        labels = torch.full((6,), 3, dtype=torch.long)
        weights = torch.arange(1, 9, dtype=torch.float32)
        a = torch.nn.functional.cross_entropy(logits, labels)
        b = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(torch.autograd.grad(a, logits)[0],
                                   torch.autograd.grad(b, logits)[0])

    def test_probe_is_fixed_unique_and_class_stratified(self):
        rows = probe_rows(self.data.arrays['val'][1], 3)
        self.assertEqual(len(rows), 24)
        np.testing.assert_array_equal(rows, probe_rows(self.data.arrays['val'][1], 3))
        np.testing.assert_array_equal(np.bincount(self.data.arrays['val'][1][rows]), [3]*8)
        self.assertEqual(len(np.unique(rows)), 24)
        audit = reduction_audit([[8, 0], [0, 8]], [1, 4])
        self.assertEqual(audit['client_to_pooled_gradient_scale_proxy'], [2.5, .625])

    def test_cap_preserves_aggregation_weights_and_starting_model(self):
        original = copy.deepcopy(self.model.state_dict())
        rows = probe_rows(self.data.arrays['val'][1], 3)
        full = branch(self.data, self.model, self.parts, self.settings, rows, 21, 0)
        capped = branch(self.data, self.model, self.parts, self.settings, rows, 21, 2)
        repeated = branch(self.data, self.model, self.parts, self.settings, rows, 21, 2)
        self.assertEqual(capped, repeated)
        self.assertEqual(full['optimizer_steps'], 12)
        self.assertEqual(full['examples_processed'], 96)
        self.assertEqual(capped['optimizer_steps'], 4)
        self.assertEqual(capped['examples_processed'], 32)
        self.assertEqual(full['aggregation_weights'], [32, 64])
        self.assertEqual(capped['aggregation_weights'], [32, 64])
        for key in original:
            self.assertTrue(torch.equal(original[key], self.model.state_dict()[key]))

    def test_one_client_aggregation_matches_direct_local_training(self):
        from src.models.losses import build_criterion
        rows = np.arange(96)
        result = branch(self.data, self.model, [rows], self.settings, rows, 21, 0)
        seed = client_seed(7, 0, 21)
        set_seed(seed)
        direct = MLPClassifier(39, 8, self.model.config)
        direct.load_state_dict(self.model.state_dict())
        optimizer = torch.optim.Adam(direct.parameters(), lr=.001, weight_decay=1e-5)
        train_epoch(direct, optimizer,
                    build_criterion('sqrt_weighted_ce', self.data.manifest['train_class_counts']),
                    self.data.batches('train', 8, seed, rows), 'cpu')
        _, expected = evaluate(direct, self.data.batches('val', 8), 'cpu')
        self.assertEqual(result['aggregate_full_validation_metrics'], expected)


if __name__ == '__main__':
    unittest.main()

"""Explanation safety, sampling and numerical-contract checks."""

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import polars as pl
import torch

from src.data.label_map import CLASSES
from src.explain.research import load_model, logits, confusion, select_rows, attribution_summary
from src.models.dataset import to_arrays
from src.models.research import main as train
from tests.test_baselines import synthetic_splits


class ExplanationTests(unittest.TestCase):
    def test_sampling_shared_deterministic_and_missing_errors(self):
        y = np.repeat(np.arange(len(CLASSES)), 5)
        predictions = {'perfect': y.copy(), 'all_benign': np.zeros_like(y)}
        a = select_rows(y, predictions, 2, 4266)
        b = select_rows(y, predictions, 2, 4266)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(len(set(a[1])), len(a[1]))
        self.assertEqual(np.bincount(y[a[0]]).tolist(), [2] * len(CLASSES))
        self.assertIsNone(a[3]['perfect']['miss_Web-based']['selected_row'])
        self.assertEqual(a[3]['all_benign']['miss_Web-based']['eligible_validation_count'], 5)

    def test_both_normalizations_reproduce_predictions_and_reject_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_splits(root / 'splits')  # Deliberately no test split.
            for norm in ('batch', 'layer'):
                folder = root / norm
                train(['--lane', 'light', '--splits', str(root / 'splits'), '--output', str(folder),
                       '--normalization', norm, '--epochs', '1', '--batch-size', '32', '--threads', '1'])
                model, scaler, m, r = load_model(folder)
                x, y = to_arrays(pl.read_parquet(root / 'splits/val.parquet'), scaler, m['feature_columns'])
                self.assertEqual(confusion(y, logits(model, x).argmax(1)).tolist(), r['validation_metrics']['confusion_matrix'])
                # Manifests from either OS must retain the same strict hash checks.
                manifest = folder / 'environment.json'
                for separator in ('/', '\\'):
                    m['source_sha256'] = {name.replace('\\', '/').replace('/', separator): digest
                                          for name, digest in m['source_sha256'].items()}
                    manifest.write_text(json.dumps(m))
                    load_model(folder)
                original = dict(m['source_sha256'])
                m['source_sha256']['src/models/architectures.py'] = 'changed'
                manifest.write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError, 'Conflicting inference source hashes'):
                    load_model(folder)
                m['source_sha256'] = {name: digest for name, digest in original.items()
                                      if name != 'src\\models\\architectures.py'}
                manifest.write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError, 'Inference source changed'):
                    load_model(folder)
                m['source_sha256'] = original
                manifest.write_text(json.dumps(m))
                # A changed artifact must fail before deserialization is attempted.
                with (folder / 'best.pt').open('ab') as file:
                    file.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    load_model(folder)

    def test_expected_gradients_linear_contract(self):
        import shap
        torch.set_num_threads(1)
        model = torch.nn.Linear(3, len(CLASSES), bias=True).eval()
        x = torch.tensor([[1., 2., -1.], [2., 1., 3.]])
        background = torch.zeros(4, 3)
        v = np.asarray(shap.GradientExplainer(model, background).shap_values(x, nsamples=8, rseed=0))
        expected = x.numpy()[:, :, None] * model.weight.detach().numpy().T[None, :, :]
        np.testing.assert_allclose(v, expected, rtol=1e-5, atol=1e-6)
        out = model(x).detach().numpy()
        base = model(background).detach().numpy().mean(0)
        np.testing.assert_allclose(v.sum(1), out - base, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

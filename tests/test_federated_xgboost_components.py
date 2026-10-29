"""Synthetic checks for the federated XGBoost component study helpers (no xgboost needed)."""
import json
import unittest

import numpy as np

from src.eval import federated_xgboost_components as fx


class ComponentHelperTests(unittest.TestCase):
    def test_class_weight_powers(self):
        counts = np.array([100., 25., 4., 1., 1., 1., 1., 1.])
        np.testing.assert_allclose(fx.class_weights(counts, 'sqrt'), counts**-.5)
        np.testing.assert_allclose(fx.class_weights(counts, 'balanced'), 1/counts)
        np.testing.assert_allclose(fx.class_weights(counts, 'power0.75'), counts**-.75)
        np.testing.assert_allclose(fx.class_weights(counts, 'none'), np.ones(8))

    def test_row_weights_have_unit_mean(self):
        y = np.array([0]*90+[1]*9+[2])
        self.assertAlmostEqual(float(fx.weights(y, 'sqrt').mean()), 1., places=5)

    def test_plan_keeps_holdouts_closed(self):
        plan = json.loads(fx.PLAN.read_text())
        self.assertFalse(plan['confirmation_scoring_permitted'] or plan['test_scoring_permitted'])
        self.assertEqual(plan['stage1_central']['base']['max_depth'], 6)


if __name__ == '__main__':
    unittest.main()

"""Synthetic checks for the FedLC / local-prior offsets used with cyclic XGBoost (no xgboost needed)."""
import json
import unittest

import numpy as np

from src.eval import federated_xgboost_mechanisms as fm


class OffsetTests(unittest.TestCase):
    def test_fedlc_offsets_follow_count_power(self):
        y = np.array([0]*16+[1])
        np.testing.assert_allclose(fm.offsets('fedlc', y, None)[:3], [-16**-.25, -1., -1.])

    def test_local_prior_offset_is_zero_when_client_matches_global(self):
        y = np.repeat(np.arange(8), 10)
        prior = np.full(8, 1/8)
        np.testing.assert_allclose(fm.offsets('local_prior', y, prior), np.zeros(8), atol=1e-12)

    def test_reference_has_no_offset(self):
        np.testing.assert_array_equal(fm.offsets('none', np.array([0, 1]), None), np.zeros(8))

    def test_plan_keeps_holdouts_closed(self):
        plan = json.loads(fm.PLAN.read_text())
        self.assertFalse(plan['confirmation_scoring_permitted'] or plan['test_scoring_permitted'])
        self.assertEqual(plan['amendment_1']['arms_now'], ['reference', 'fedlc', 'local_prior'])


if __name__ == '__main__':
    unittest.main()

"""Synthetic PLE encoding checks; not CICIoT2023 quality experiments."""
import json
import unittest

import numpy as np
import torch

from src.eval.ple_shared_adam_screen import PLAN, PLE


class PLETests(unittest.TestCase):
    def test_cumulative_ramps_clip_tails_and_zero_constants(self):
        x = np.array([[0., 5.], [1., 5.], [2., 5.]], dtype=np.float32)
        ple = PLE(x, 2)
        self.assertEqual(ple.dim, 3)
        z = ple(torch.tensor([[1.5, 5.], [-3., 9.], [7., 5.]]))
        torch.testing.assert_close(z, torch.tensor([[1., .5, 0.], [0., 0., 0.], [1., 1., 0.]]))

    def test_plan_keeps_holdouts_closed_and_reuses_historical_rates(self):
        plan = json.loads(PLAN.read_text())
        self.assertEqual(plan['arms']['central_ple']['learning_rate'], .001)
        self.assertEqual(plan['arms']['compressed_ple']['learning_rate'], .003)
        self.assertFalse(plan['confirmation_scoring_permitted'] or plan['test_scoring_permitted'])


if __name__ == '__main__':
    unittest.main()

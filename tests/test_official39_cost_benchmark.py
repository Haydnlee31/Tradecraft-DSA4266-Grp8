"""Synthetic cost measurement checks without archives, heldout data or timing gates."""
import unittest
from unittest.mock import patch

import torch

from src.eval.official39_cost_benchmark import timing_summary, benchmark_cases, fingerprints, run
from src.eval.federated_class_safeguard import state_digest


class CostBenchmarkTests(unittest.TestCase):
    def test_units_and_repetition_not_tail_latency(self):
        r = timing_summary([2., 1., 3.], 10)
        self.assertEqual(r['median_mean_call_ms'], 2.)
        self.assertEqual(r['amortized_ms_per_flow'], .2)
        self.assertEqual(r['flows_per_second'], 5000.)
        self.assertFalse(r['tail_request_latency_measured'])
        for values in ([], [-1], [float('nan')]):
            with self.assertRaises(ValueError): timing_summary(values, 1)

    def test_no_grad_weights_stable_and_exact_work(self):
        models = {n: torch.nn.Linear(3, 8).eval() for n in ('a', 'b')}
        x = torch.ones(8, 3)
        states, predictions = {n: state_digest(m) for n, m in models.items()}, fingerprints(models, x)
        plan = {'batch_sizes': [1, 4], 'warmup_calls_per_case': 2, 'repeats': 3,
                'calls_per_repeat': {'1': 3, '4': 2}}
        hooks = [m.register_forward_pre_hook(lambda m, args: self.assertFalse(torch.is_grad_enabled())) for m in models.values()]
        with patch.object(torch.Tensor, 'backward', side_effect=AssertionError('No training')):
            r, w = benchmark_cases(models, x, plan)
        for h in hooks: h.remove()
        self.assertEqual(w['warmup_forward_examples'], 20)
        self.assertEqual(w['timed_forward_examples'], 66)
        self.assertEqual(w['timed_calls'], 30)
        self.assertEqual({n: state_digest(m) for n, m in models.items()}, states)
        self.assertEqual(fingerprints(models, x), predictions)
        for model in r.values():
            for v in model.values(): self.assertEqual(len(v['repeat_mean_call_ms']), 3)

    def test_training_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'eval-mode'):
            benchmark_cases({'a': torch.nn.Linear(3, 8)}, torch.ones(8, 3), {})

    def test_existing_output_never_overwritten(self):
        from pathlib import Path
        with self.assertRaisesRegex(ValueError, 'preserve'):
            run('missing', 'missing', Path(__file__).parent)


if __name__ == '__main__':
    unittest.main()

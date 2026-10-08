"""Guard the frozen replication matrix against accidental budget or scope changes."""
import json
from pathlib import Path
import unittest

from src.eval.official_scale_plan import budgets


class ConfirmationPlanTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.plan = json.loads((root / 'reports/full_data_extension/scaling-confirmation-plan.json').read_text())

    def test_scope_and_paired_seeds(self):
        p = self.plan
        self.assertEqual(p['training_seeds'], [7, 17, 27])
        self.assertEqual(p['new_seeds'], [17, 27])
        self.assertEqual(p['lanes'], ['light', 'iid'])
        self.assertEqual(len(p['new_seeds'])*len(p['lanes'])*len(p['cohorts']), p['new_runs'])
        self.assertEqual(p['settings']['partition_seed'], 7)
        self.assertFalse(p['test_evaluation'])
        self.assertFalse(p['automatic_extension'])
        self.assertEqual(p['settings']['local_max_batches'], 0)

    def test_endpoint_accounting(self):
        p = self.plan
        for cohort in p['cohorts'].values():
            for lane in p['lanes']:
                n = cohort['rows']
                sizes = [n] if lane == 'light' else [n//20]*20
                final = budgets(sizes, p['epochs'])
                self.assertEqual(final['examples_processed'], cohort['final_examples'])
                self.assertEqual(final['optimizer_steps'], cohort['final_updates'][lane])
                matched = budgets(sizes, cohort['matched_examples_step'])
                self.assertEqual(matched['examples_processed'], 10000000)
                if 'step5_updates' in cohort:
                    self.assertEqual(matched['optimizer_steps'], cohort['step5_updates'][lane])


if __name__ == '__main__':
    unittest.main()

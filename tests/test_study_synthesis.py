"""No private data or network needed for the class-failure/cost synthesis."""
import json
from pathlib import Path
import unittest

from scripts import official39_study_synthesis as synthesis
from src.data.label_map import CLASSES


class StudySynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = synthesis.build()

    def test_all_nine_primary_cases_and_exact_scope(self):
        self.assertEqual(len(self.r['records']), 9)
        for key in ('new_test_inference', 'dataset_arrays_opened', 'model_trained', 'cloud_used', 'deployment_authorized'):
            self.assertFalse(self.r[key])
        self.assertTrue(self.r['source_final_test_evaluated'])
        self.assertEqual(self.r['class_order'], CLASSES)
        for lane, g in self.r['groups'].items():
            self.assertEqual(set(g['per_class']), set(CLASSES))
            self.assertEqual(len(g['metrics']['macro_f1']['values']), 3)

    def test_all_errors_partition_support_and_rescaling(self):
        for r in self.r['records']:
            for name, e in r['errors'].items():
                if name == 'Benign':
                    self.assertEqual(e['correct']+e['false_alerts'], e['support'])
                    self.assertAlmostEqual(e['false_alerts_per_10000_benign'], 10000*e['false_alerts']/e['support'])
                else:
                    self.assertEqual(e['correct_category']+e['predicted_benign']+e['wrong_attack_category'], e['support'])
                    self.assertAlmostEqual(e['category_recall']+e['attack_to_benign_rate']+e['wrong_attack_category_rate'], 1.)
                    self.assertGreaterEqual(e['any_attack_alarm_recall'], e['category_recall']-1e-12)

    def test_budget_and_communication_are_not_measurements(self):
        for r in self.r['records']:
            costs = r['costs']
            self.assertEqual(costs['num_parameters'], 5096)
            self.assertEqual(costs['fp32_parameter_bytes'], 20384)
            self.assertEqual(costs['examples_processed'], 40000000)
            comm = costs['communication']
            self.assertEqual(comm['both_directions_bytes'], 0 if r['lane'] == 'light' else 16307200)
            self.assertFalse(comm['measured_network_traffic'])
            self.assertIsNone(comm['raw_data_collection_bytes'])
        s = self.r['training_only_sgd_diagnostic']
        self.assertEqual(s['scaffold']['communication']['both_directions_bytes'], 16307200)
        self.assertEqual(s['fedavg_sgd']['communication']['both_directions_bytes'], 8153600)
        self.assertEqual(s['scaffold']['work']['optimizer_steps'], 10000)

    def test_communication_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError): synthesis.communication(1, -1, 2)
        with self.assertRaises(ValueError): synthesis.communication(1., 1, 2)

    def test_frozen_training_diagnostic_not_used_as_final_test(self):
        for arm, r in self.r['training_only_sgd_diagnostic'].items():
            self.assertIn('TRAINING ONLY', r['scope'])
            self.assertEqual(set(r['argmax_training_metrics']['per_class']), set(CLASSES))
        self.assertGreater(self.r['paired_macro_f1_gaps']['iid_minus_dirichlet']['mean'], .20)
        self.assertLess(self.r['paired_macro_f1_gaps']['light_minus_iid']['mean'], .05)

    def test_tables_include_every_class_and_limitations(self):
        text = synthesis.tables(self.r)
        for name in CLASSES: self.assertIn('| '+name+' |', text)
        self.assertIn('NOT from model fitting', text)
        self.assertIn('NOT zero data-transfer cost', text)


if __name__ == '__main__':
    unittest.main()

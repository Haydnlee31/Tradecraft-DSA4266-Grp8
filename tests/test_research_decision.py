"""Frozen-study intake and unchanged decision-gate regression tests."""

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.eval.decision import Criteria, build_decision, load_reports, render_markdown
from src.eval.research_decision import normalize_confirmation, build_research_decision, validate_metrics, error_breakdown
from tests.test_decision import make_report


def fixture():
    common = {'loss': 'sqrt_weighted_ce', 'epochs': 60, 'batch_size': 512,
              'lr': .001, 'weight_decay': 1e-5, 'patience': 0, 'partition_seed': 0,
              'alpha': .5, 'clients': 20}
    plan = {'common': common, 'training_seeds': [0, 1, 2], 'new_training_seeds': [1, 2],
            'lanes': {}, 'limitations': ['Seed 0 is screening evidence.']}
    study = {'test_evaluated': False, 'lanes': {}}
    for lane in ('heavy', 'light', 'iid', 'dirichlet'):
        normalization = 'batch' if lane in ('heavy', 'light') else 'layer'
        dropout = .3 if lane == 'heavy' else .2
        plan['lanes'][lane] = {'normalization': normalization, 'dropout': dropout}
        runs = {}
        for seed in (0, 1, 2):
            score = {'heavy': 900, 'light': 890, 'iid': 950, 'dirichlet': 885}[lane]
            matrix = np.eye(len(CLASSES), dtype=int) * score
            for i in range(len(CLASSES)):
                matrix[i, (i + 1) % len(CLASSES)] = 1000 - score
            metrics = {'macro_f1': score / 1000, 'accuracy': score / 1000,
                       'benign_false_alert_rate': 1 - score / 1000,
                       'per_class_recall': {name: score / 1000 for name in CLASSES},
                       'confusion_matrix': matrix.tolist()}
            manifest = {'classes': CLASSES, 'settings': {**common, 'lane': lane, 'seed': seed,
                        'normalization': normalization, 'dropout': dropout},
                        'model_config': {'hidden_dims': [512] if lane == 'heavy' else [32],
                                         'dropout': dropout, 'normalization': normalization},
                        'source_sha256': {'source': 'same'}, 'split_sha256': {'train': 'train', 'val': 'val'},
                        'packages': {'test': '1'}, 'feature_columns': ['a'], 'scaler_sha256': 'same',
                        'train_class_counts': dict.fromkeys(CLASSES, 10), 'partition_sha256': lane,
                        'local_epochs': 1, 'fraction_train': 1.0}
            runs[str(seed)] = {'manifest': manifest, 'validation_metrics': metrics,
                              'num_parameters': 200000 if lane == 'heavy' else 5500,
                              'reused_screening_seed': seed == 0, 'path': f'{lane}/{seed}'}
        study['lanes'][lane] = {'runs': runs}
    return study, plan


class ResearchDecisionTests(unittest.TestCase):
    def test_legacy_validation_only_and_mixed_test_metrics(self):
        import json
        reports = []
        for lane in ('centralized-heavy', 'centralized-light', 'federated-light'):
            for seed in range(3):
                r = make_report(lane, seed, .8, .1, 5500)
                r['test_metrics'] = None
                reports.append(r)
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'run.json').write_text(json.dumps(reports[0]))
            loaded, warnings = load_reports(Path(d))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(warnings, [])
        decision = build_decision(reports, Criteria())
        self.assertEqual(decision['status'], 'ready')
        self.assertIn('N/A', render_markdown(decision))
        self.assertIsNone(decision['selected_candidates']['centralized-light']['test_macro_f1_mean'])
        reports[0]['test_metrics'] = make_report('centralized-heavy', 0, .8, .99, 5500)['test_metrics']
        mixed = build_decision(reports, Criteria())
        self.assertIsNone(mixed['selected_candidates']['centralized-heavy']['test_macro_f1_mean'])
        self.assertEqual(mixed['recommended_lane'], decision['recommended_lane'])

    def test_frozen_comparison_uses_same_gates_and_keeps_iid_diagnostic(self):
        study, plan = fixture()
        reports, diagnostics = normalize_confirmation(study, plan)
        payload = build_research_decision(study, plan, reports, diagnostics)
        self.assertEqual(payload['decision']['criteria'], asdict(Criteria()))
        self.assertEqual(payload['decision']['selected_candidates']['federated-light']['partitioner'], 'dirichlet')
        self.assertEqual(len(payload['decision']['all_candidates']), 4)
        self.assertFalse(payload['deployment_authorized'])
        self.assertEqual(payload['new_seed_sensitivity']['status'], 'incomplete')
        self.assertIsNone(payload['new_seed_sensitivity']['recommended_lane'])
        self.assertTrue(all(r['test_metrics'] is None for r in reports))

    def test_reject_missing_seeds_drift_and_nonfinite_metrics(self):
        original, plan = fixture()
        bad = copy.deepcopy(original)
        del bad['lanes']['heavy']['runs']['2']
        with self.assertRaisesRegex(ValueError, 'seed'):
            normalize_confirmation(bad, plan)
        bad = copy.deepcopy(original)
        bad['lanes']['iid']['runs']['1']['manifest']['settings']['lr'] = .01
        with self.assertRaisesRegex(ValueError, 'setting'):
            normalize_confirmation(bad, plan)
        bad = copy.deepcopy(original)
        bad['lanes']['light']['runs']['1']['validation_metrics']['macro_f1'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'macro_f1'):
            normalize_confirmation(bad, plan)
        bad = copy.deepcopy(original)
        bad['lanes']['dirichlet']['runs']['2']['manifest']['partition_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'partition'):
            normalize_confirmation(bad, plan)

    def test_attack_error_partition_distinguishes_benign_from_wrong_attack(self):
        matrix = np.eye(len(CLASSES), dtype=int) * 10
        i, benign, other = CLASSES.index('DoS'), CLASSES.index('Benign'), CLASSES.index('DDoS')
        matrix[i, benign], matrix[i, other] = 3, 7
        row = error_breakdown(matrix)['DoS']
        self.assertEqual(row['support'], 20)
        self.assertEqual(row['correct_category'], 10)
        self.assertEqual(row['predicted_benign'], 3)
        self.assertEqual(row['wrong_attack_category'], 7)
        self.assertEqual(row['attack_to_benign_rate'], .15)

    def test_counts_and_summary_must_agree(self):
        study, _ = fixture()
        m = study['lanes']['heavy']['runs']['0']['validation_metrics']
        m['per_class_recall']['Web-based'] = .99
        with self.assertRaisesRegex(ValueError, 'recall'):
            validate_metrics(m)

    def test_require_federated_does_not_fall_back_to_iid(self):
        study, plan = fixture()
        reports, diagnostics = normalize_confirmation(study, plan)
        for r in reports:
            if r.get('partitioner') == 'dirichlet':
                r['validation_metrics']['per_class_recall']['Web-based'] = 0
        payload = build_research_decision(study, plan, reports, diagnostics, require_federated=True)
        self.assertEqual(payload['decision']['status'], 'incomplete')
        self.assertIsNone(payload['decision']['recommended_lane'])


if __name__ == '__main__':
    unittest.main()

"""Portable integrity checks of the published counts and synthetic CPU gate."""
import hashlib
import json
from pathlib import Path
import unittest

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data import official_ablation_panel as panel
from src.models import official_streaming as streaming
from src.eval import official_ablation_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/full_data_extension'


def raw_digest(record, additions):
    raw = {k: v for k, v in record.items() if k not in additions}
    return hashlib.sha256((json.dumps(raw, indent=2)+'\n').encode()).hexdigest()


class AblationReceiptTests(unittest.TestCase):
    def test_panel_receipt_hashes_counts_and_nested_exclusions(self):
        r = json.loads((REPORTS/'ablation-panel-results.json').read_text())
        ref = json.loads((REPORTS/'explanation-cpu-plan.json').read_text())
        old = json.loads((REPORTS/'shortcut-audit-results.json').read_text())
        self.assertEqual(r['builder_sha256'], sha256(panel.__file__))
        self.assertEqual(r['plan_sha256'], sha256(REPORTS/'ablation-panel-plan.json'))
        self.assertEqual(r['source_receipt_sha256'], r['independent_replay_receipt_sha256'])
        self.assertEqual(r['source_receipt_sha256'], raw_digest(r, {'source_receipt_sha256', 'independent_replay_receipt_sha256'}))
        self.assertTrue(all(r[k] is False for k in panel.FLAGS))
        self.assertEqual(r['retained_rows']+r['excluded_rows'], r['source_validation_rows'])
        self.assertTrue(r['all_classes_retained'])
        self.assertEqual(r['retained_rows'], 2047805)
        self.assertEqual(r['excluded_rows'], 11479)
        for name in CLASSES:
            self.assertGreater(r['retained_class_counts'][name], 0)
            self.assertEqual(r['retained_class_counts'][name]+r['excluded_class_counts'][name], ref['validation_class_counts'][name])
        for split in ('train', 'val'):
            self.assertEqual(r['additional_overlap_rows_vs_full39'][split], 8)
            self.assertEqual(r['joint_projection'][f'cross_split_{split}_rows']-old['collisions']['full39'][f'cross_split_{split}_rows'], 8)
        j = r['joint_projection']
        self.assertEqual(j['unique_vectors'], j['train_unique_vectors']+j['val_unique_vectors']-j['cross_split_groups'])

    def test_archived_runner_gate_is_synthetic_complete_and_pinned(self):
        r = json.loads((REPORTS/'ablation-local-gate-results.json').read_text())
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['device'], 'cpu')
        self.assertIn('Synthetic', r['fixture_scope'])
        self.assertFalse(r['test_evaluated'])
        self.assertFalse(r['comparison_training_launched'])
        self.assertEqual(r['runner_sha256'], sha256(streaming.__file__))
        self.assertEqual(r['pilot_sha256'], sha256(recovery.__file__))
        for name, digest in r['fixture_sources_sha256'].items():
            self.assertEqual(digest, sha256(ROOT/name))
        for name, digest in r['supporting_sources_sha256'].items():
            self.assertEqual(digest, sha256(ROOT/'src'/name))
        self.assertEqual(r['source_receipt_sha256'], raw_digest(r, {'source_receipt_sha256', 'fixture_scope', 'reference_commit', 'fixture_sources_sha256', 'supporting_sources_sha256'}))
        self.assertEqual(r['reference_commit'], 'f6e0bc3')
        self.assertEqual(r['reference_runner_sha256'], '82eebe2e342c727fb79811278c9847e9debd0daeb9c2ff7debf7c29988a29646')
        self.assertEqual(r['total_examples_processed'], 6720)
        self.assertEqual(r['expected_total_examples_processed'], 6720)
        self.assertEqual(set(r['lanes']), {'light', 'iid', 'dirichlet'})
        for lane in r['lanes'].values():
            self.assertTrue(lane['default_path_bridge_exact'])
            self.assertEqual(lane['paired_initial_state']['num_parameters'], 5096)
            self.assertEqual(set(lane['arms']), {'full39', 'number_masked', 'number_total_masked'})
            self.assertTrue(all(a['recovery_exact'] for a in lane['arms'].values()))


if __name__ == '__main__':
    unittest.main()

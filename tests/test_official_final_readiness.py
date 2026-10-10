"""Audit published preparation receipts without arrays, checkpoints or Torch.

These checks keep the final-test boundary visible in CI. They do not replace
the original archive audit or the disk-backed checks on the private data.
"""
import hashlib
import unittest

from scripts import official39_closeout as closeout


class FinalReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = closeout.read(closeout.FINAL_PLAN)
        cls.cuda = closeout.read(closeout.REPORTS / 'final-validation-cuda-checks.json')
        cls.prepared = closeout.read(closeout.REPORTS / 'test-panel-preparation.json')

    def test_completed_cuda_gate_has_all_candidates_and_no_test_scores(self):
        r = self.cuda
        self.assertEqual(r['status'], 'complete')
        for key in ('candidates_verified', 'exact_same_runtime_references', 'exact_historical_endpoints'):
            self.assertEqual(r[key], 27)
        self.assertEqual(set(r['cases']), {f'case-{i:02d}.json' for i in range(27)})
        self.assertTrue(all(closeout.valid_digest(value) for value in r['cases'].values()))
        self.assertTrue(r['test_scoring_ready'])
        self.assertEqual(r['runtime']['device'], 'cuda')
        self.assertEqual(r['runtime']['torch'], '2.7.0+cu128')
        for key in ('test_opened', 'test_evaluated', 'model_trained'):
            self.assertIs(r[key], False)

    def test_receipts_bind_the_same_unchanged_implementation_and_protocol(self):
        self.assertEqual(self.cuda['source_sha256'], self.prepared['source_sha256'])
        for receipt in (self.cuda, self.prepared):
            self.assertEqual(receipt['plan_sha256'], closeout.digest(closeout.FINAL_PLAN))
        for name, expected in self.cuda['source_sha256'].items():
            path = (closeout.ROOT / name).resolve()
            self.assertTrue(path.is_relative_to(closeout.ROOT.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, name)
        self.assertEqual(closeout.audit_final_plan(self.plan)['status'], 'prepared_test_closed')

    def test_preparation_is_authorized_data_access_not_model_evaluation(self):
        r = self.prepared
        self.assertEqual(r['status'], 'prepared_not_scored')
        self.assertEqual(r['protocol'], 'official39-test-panel-v1')
        self.assertIs(r['explicit_test_preparation_opt_in'], True)
        self.assertIs(r['test_opened'], True)
        self.assertIs(r['test_evaluated'], False)
        self.assertIs(r['model_trained'], False)
        self.assertEqual(set(r['files']), {'x.npy', 'y.npy', 'retained.npy'})
        self.assertTrue(all(closeout.valid_digest(value) for value in r['files'].values()))
        identity = self.plan['data_identity']
        self.assertEqual(r['parent_manifest_sha256'], identity['parent_shards_manifest_sha256'])
        self.assertEqual(r['scaler_sha256'], identity['scaler_sha256'])
        self.assertEqual(r['features'], identity['features'])
        self.assertEqual(r['classes'], closeout.CLASSES)
        self.assertEqual(r['rows'], identity['test_rows_before_overlap_filter'])
        self.assertEqual(r['class_counts'], identity['test_class_counts_before_overlap_filter'])

    def test_shared_population_keeps_all_classes_and_counts_every_exclusion(self):
        r = self.prepared
        p = r['panel']
        self.assertIs(p['all_classes_retained'], True)
        self.assertIs(p['label_blind_membership'], True)
        self.assertEqual(p['retained_rows'] + p['excluded_rows'], r['rows'])
        self.assertEqual(sum(r['class_counts'].values()), r['rows'])
        for name in closeout.CLASSES:
            self.assertGreater(p['retained_class_counts'][name], 0)
            self.assertEqual(p['retained_class_counts'][name] + p['excluded_class_counts'][name], r['class_counts'][name])
        for population in ('retained', 'excluded'):
            self.assertEqual(sum(p[population + '_class_counts'].values()), p[population + '_rows'])
        self.assertEqual(set(p['projections']), set(closeout.ARMS))
        previous = 0
        for arm in closeout.ARMS:
            projection = p['projections'][arm]
            matched = projection['matched_rows']
            self.assertGreaterEqual(matched['either'], previous)
            self.assertGreaterEqual(matched['either'], max(matched['train'], matched['val']))
            self.assertLessEqual(matched['either'], matched['train'] + matched['val'])
            previous = matched['either']
            for source in ('train', 'val', 'either'):
                self.assertEqual(sum(projection['matched_class_counts'][source].values()), matched[source])
            self.assertEqual(projection['held_out_unique_projected_vectors'] + projection['duplicate_excess_rows'], r['rows'])
            self.assertLessEqual(projection['duplicate_groups'], projection['duplicate_excess_rows'])
        # Removing both columns contains the other two projection overlaps.
        self.assertEqual(previous, p['excluded_rows'])


if __name__ == '__main__':
    unittest.main()

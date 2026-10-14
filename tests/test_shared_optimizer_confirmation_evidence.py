"""The public readiness receipt contains no confirmation outcomes or row data."""
import hashlib
import json
import unittest

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import shared_optimizer_confirmation_prepare as prep


class ConfirmationEvidenceTests(unittest.TestCase):
    def evidence(self):
        path=prep.PLAN.parent/'confirmation-preparation-results.json'
        evidence=json.loads(path.read_text())
        return evidence['lock'],evidence['checks']

    def test_lock_digest_source_plan_and_parent_are_frozen(self):
        lock,checks=self.evidence(); plan,*_=prep.read_contract()
        self.assertEqual(checks['lock_sha256'],hashlib.sha256(json.dumps(lock,indent=2).encode()).hexdigest())
        self.assertEqual(lock['plan'],plan)
        self.assertEqual(lock['plan_sha256'],sha256(prep.PLAN))
        self.assertEqual(lock['parent_evidence_sha256'],sha256(prep.PARENT))
        self.assertEqual(lock['sources'],prep.sources())
        self.assertEqual(checks['source_sha256'],sha256(prep.__file__))
        self.assertEqual(checks['git_commit'],'adeddbb58dc4e7ddcf7dece2e2138f7689465bbf')

    def test_all_twelve_models_and_forty_eight_thresholds_derive_from_audited_evidence(self):
        lock,checks=self.evidence(); plan,_,_,_,parent=prep.read_contract()
        self.assertEqual(lock['models'],prep.registry(parent,plan))
        self.assertEqual(lock['development_split'],parent['summary']['split'])
        expected={(s,a) for s in plan['seeds'] for a in plan['arms']}
        self.assertEqual({(r['seed'],r['arm']) for r in checks['checks']},expected)
        self.assertEqual(len(checks['checks']),12)
        by_id={(r['seed'],r['arm']):r for r in lock['models']}
        for r in checks['checks']:
            frozen=by_id[r['seed'],r['arm']]
            self.assertEqual(r['checkpoint_sha256'],frozen['checkpoint_sha256'])
            self.assertEqual(r['model_sha256'],frozen['model_sha256'])
            self.assertTrue(r['checkpoint_inference_exact'])
            self.assertTrue(r['historical_endpoint_exact'])
            self.assertEqual(r['independent_frozen_operating_points'],4)
        self.assertEqual(sum(len(m['calibrations']) for m in lock['models']),48)

    def test_preparation_only_no_training_refitting_cloud_or_holdout_access(self):
        lock,checks=self.evidence()
        self.assertEqual(checks['status'],'ready_for_explicit_confirmation_authorization')
        for key in ('training_permitted','threshold_refitting_permitted','confirmation_scoring_authorized'):
            self.assertIs(lock[key],False)
        for key in ('thresholds_refitted','confirmation_opened','confirmation_scored','reserved_row_ids_opened',
                    'raw_shards_opened','original_validation_opened','test_opened','test_evaluated','cloud_used'):
            self.assertIs(checks[key],False)
        self.assertEqual(checks['new_model_fits'],0)
        self.assertEqual(checks['backward_examples'],0)
        # Only aggregate calibration evidence, identities and relative artifact
        # paths are published; no new confirmation scores or row-level arrays.
        self.assertEqual(set(lock),{'version','plan','plan_sha256','sources','parent_evidence_sha256',
            'data','models','development_split','runtime','confirmation_scoring_authorized',
            'training_permitted','threshold_refitting_permitted'})
        self.assertNotIn('confirmation_metrics',checks)

    def test_counts_and_provenance_match_existing_manifest_not_opened_arrays(self):
        lock,checks=self.evidence(); plan,base,_,performance,_=prep.read_contract()
        data=lock['data']; old=performance['summary']['data']
        self.assertEqual(data['confirmation_counts'],old['counts']['confirmation'])
        self.assertEqual(data['confirmation_rows'],sum(old['counts']['confirmation'].values()))
        self.assertEqual(data['confirmation_row_ids_sha256'],old['files']['confirmation_row_ids.npy'])
        self.assertEqual(data['scaler_sha256'],old['files']['scaler.json'])
        self.assertEqual(data['source_parent_manifest_sha256'],base['parent_manifest_sha256'])
        self.assertEqual(data['source_subset_manifest_sha256'],base['subset_manifest_sha256'])
        self.assertEqual(data['classes'],CLASSES)
        self.assertEqual(data['features'],old['features'])
        self.assertEqual(len(data['features']),39)
        for k in ('raw_shards_opened','reserved_row_ids_opened','unhashed_staging_cache_used'):
            self.assertFalse(data[k])
        self.assertEqual(checks['forward_examples'],12*plan['development_rows'])
        self.assertEqual(checks['planned_confirmation_forward_examples'],12*plan['confirmation_rows'])
        self.assertGreater(data['source_shard_count'],0)

    def test_recorded_timings_are_measured_inference_not_cloud_or_training(self):
        _,checks=self.evidence()
        self.assertEqual(checks['inference_seconds'],sum(c['inference_seconds'] for c in checks['checks']))
        self.assertGreater(checks['elapsed_seconds'],checks['inference_seconds'])
        self.assertGreater(checks['process_peak_rss_bytes'],0)
        self.assertEqual(checks['new_model_fits'],0)


if __name__=='__main__':
    unittest.main()

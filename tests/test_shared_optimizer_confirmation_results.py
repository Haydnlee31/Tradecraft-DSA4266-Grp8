"""Check the final published evidence without opening private data or models."""
import hashlib
import json
import unittest

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import shared_optimizer_confirmation as final


class ConfirmationResultsTests(unittest.TestCase):
    def evidence(self):
        return json.loads((final.PLAN.parent/'confirmation-results.json').read_text())

    def test_audit_summary_source_and_authorization_are_frozen(self):
        evidence=self.evidence(); s,a=evidence['summary'],evidence['audit']
        execution,lock,*_=final.read_contract()
        self.assertEqual(a['status'],'complete')
        self.assertEqual(a['summary_sha256'],hashlib.sha256(json.dumps(s,indent=2).encode()).hexdigest())
        self.assertEqual(s['execution_plan'],execution)
        self.assertEqual(s['contract'],final.contract())
        self.assertEqual(s['lock_sha256'],execution['lock_sha256'])
        self.assertTrue(execution['confirmation_scoring_authorized'])
        self.assertFalse(lock['confirmation_scoring_authorized'])  # Old preparation stays historical.

    def test_all_frozen_endpoints_and_thresholds_present_unchanged(self):
        s=self.evidence()['summary']; _,lock,*_=final.read_contract()
        self.assertEqual(len(s['results']),12)
        self.assertEqual({(r['seed'],r['arm']) for r in s['results']},
                         {(e['seed'],e['arm']) for e in lock['models']})
        for e in lock['models']:
            r=next(r for r in s['results'] if (r['seed'],r['arm'])==(e['seed'],e['arm']))
            for k in ('checkpoint_sha256','model_sha256'): self.assertEqual(r[k],e[k])
            self.assertEqual(r['work'],e['training_work'])
            self.assertEqual([p['calibration'] for p in r['operating_points']],e['calibrations'])
        self.assertEqual(sum(len(r['operating_points']) for r in s['results']),48)

    def test_confirmation_membership_groups_scaler_and_counts_verified(self):
        s=self.evidence()['summary']; d=s['data']; _,lock,*_=final.read_contract()
        self.assertEqual(s['data_receipt_sha256'],hashlib.sha256(json.dumps(d,indent=2).encode()).hexdigest())
        self.assertEqual(d['source'],lock['data'])
        self.assertEqual(d['rows'],398502); self.assertEqual(d['raw_rows_streamed'],2000000)
        self.assertEqual(d['fit_development_anchor_rows_verified'],899552)
        self.assertTrue(d['raw_vectors_unique']); self.assertTrue(d['all_frozen_memberships_exact'])
        self.assertEqual(d['unique_input_groups']['confirmation'],398281)
        self.assertFalse(any(d['cross_cohort_input_groups'].values()))
        self.assertEqual(d['counts']['Web-based'],500); self.assertEqual(d['counts']['Brute Force'],260)
        self.assertFalse(d['unhashed_staging_cache_used']); self.assertFalse(d['scaler_refitted'])
        self.assertEqual(d['builder_sha256'],sha256(final.cohort.__file__))

    def test_all_metrics_supports_default_and_four_points_are_consistent(self):
        s=self.evidence()['summary']; counts=s['data']['counts']
        for r in s['results']:
            for m in [r['default_metrics'],*[p['confirmation_metrics'] for p in r['operating_points']]]:
                self.assertEqual(sum(sum(row) for row in m['confusion_matrix']),398502)
                for i,c in enumerate(CLASSES):
                    self.assertEqual(sum(m['confusion_matrix'][i]),counts[c])
                    self.assertEqual(m['per_class'][c]['support'],counts[c])
                    self.assertEqual(m['per_class'][c]['recall'],m['confusion_matrix'][i][i]/counts[c])
                self.assertAlmostEqual(m['macro_f1'],sum(m['per_class'][c]['f1'] for c in CLASSES)/8,places=15)

    def test_assessment_all_three_pass_without_average_rescue(self):
        s=self.evidence()['summary']; _,lock,_,_,_,parent=final.read_contract()
        expected=final.summarize(s['results'],lock,parent)
        for k,v in expected.items(): self.assertEqual(s[k],v)
        self.assertTrue(s['assessment']['all_seeds_pass'])
        self.assertEqual([r['seed'] for r in s['assessment']['per_seed']],[7,17,27])
        for r in s['assessment']['per_seed']:
            self.assertTrue(r['passes']); self.assertEqual(len(r['checks']),5)
            self.assertTrue(all(r['checks'].values()))
        self.assertFalse(s['assessment']['previous_training_screen_overridden'])

    def test_primary_comparability_and_secondary_exceptions_not_hidden(self):
        s=self.evidence()['summary']
        primary=[r for r in s['comparisons'] if r['target_basis_points']==500]
        self.assertEqual(len(primary),9); self.assertTrue(all(r['actual_far_comparable'] for r in primary))
        exceptions={(r['seed'],r['target_basis_points'],r['reference']) for r in s['comparisons'] if not r['actual_far_comparable']}
        self.assertEqual(exceptions,{(17,2000,'shared_adam'),(17,2000,'fedavg_adam'),(27,2000,'shared_adam')})

    def test_report_headlines_rare_counts_and_tradeoffs_are_supported(self):
        s=self.evidence()['summary']
        d={r['arm']:r for r in s['descriptive'] if r['target_basis_points']==500}
        f={a:r['macro_f1']['mean'] for a,r in d.items()}
        self.assertAlmostEqual(f['block128'],.6325367323327683)
        self.assertAlmostEqual(100*(f['block128']-f['fedavg_adam']),11.386790626503739)
        self.assertAlmostEqual(100*(f['central_adam']-f['block128']),2.098448249528262)
        self.assertLess(d['block128']['per_class_recall']['Web-based']['mean'],.1)
        expected={7:(41,66),17:(42,73),27:(65,66)}
        for r in s['results']:
            if r['arm']!='block128': continue
            m=next(p['confirmation_metrics'] for p in r['operating_points'] if p['calibration']['target_basis_points']==500)
            self.assertEqual(tuple(m['confusion_matrix'][CLASSES.index(c)][CLASSES.index(c)]
                                  for c in ('Web-based','Brute Force')),expected[r['seed']])
            self.assertEqual(r['work']['estimated_payload_bytes'],240255840)
            self.assertEqual(r['work']['synchronization_rounds'],2160)

    def test_one_forward_pass_no_training_refit_cloud_or_original_test(self):
        s,a=[self.evidence()[k] for k in ('summary','audit')]
        self.assertEqual(s['forward_examples'],4782024)
        self.assertEqual(s['new_training_runs'],0); self.assertEqual(s['backward_examples'],0)
        self.assertTrue(s['confirmation_opened']); self.assertTrue(s['confirmation_scored'])
        for k in ('thresholds_refitted','scaler_refitted','original_validation_opened','test_opened',
                  'test_evaluated','cloud_used','network_measured'): self.assertFalse(s[k])
        self.assertEqual(a['independent_metric_endpoints'],60)
        self.assertEqual(a['additional_model_forward_examples'],0)
        self.assertEqual(len(a['checks']),12)


if __name__=='__main__': unittest.main()

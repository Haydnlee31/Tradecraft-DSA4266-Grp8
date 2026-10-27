"""Reporting must not turn missing seeds or inconsistent counts into evidence."""
import copy
import unittest

from scripts import methodology_report as report


class MethodologyReportTests(unittest.TestCase):
    def fixture(self):
        published = report.old.published_reference()['results']
        rows = []
        for seed in (7,17,27):
            for arm in report.study.read_plan()['arms']:
                ref = next(r for r in published if r['seed'] == seed and r['arm'] ==
                           (arm if arm in ('central_adam','shared_adam','fedavg_adam','block128') else 'fedavg_adam'))
                r = copy.deepcopy(ref)
                scores = {k: r[k] for k in ('default_metrics','operating_points')}
                r.update(arm=arm, recovery_exact=True, recovery_backward_examples=2, training_seconds=1.,
                         confirmation=scores, confirmation_logits_sha256=r['logits_sha256'],
                         development={'audit_default_metrics':scores['default_metrics'],
                             'full_development_default_metrics':scores['default_metrics'],
                             'operating_points':[{'calibration':p['calibration'],'audit_metrics':p['confirmation_metrics']}
                                                 for p in scores['operating_points']]})
                rows.append(r)
        return {'status':'complete','plan':report.study.read_plan(),'contract':report.study.contract(),'results':rows}

    def test_audit_all_endpoints_and_historical_references(self):
        analysis = report.summarize(self.fixture())
        self.assertEqual(analysis['confusion_endpoints_checked'],363)
        self.assertEqual(len(analysis['means']),11)
        self.assertEqual(len(analysis['paired_comparisons']),36)
        self.assertEqual(len(analysis['historical_bridges']),12)
        self.assertTrue(all(r['metrics_exact'] and r['model_exact'] and r['logits_exact']
                            for r in analysis['historical_bridges']))

    def test_missing_duplicate_or_failed_run_cannot_be_hidden(self):
        for case in ('missing','duplicate','recovery'):
            s=self.fixture()
            if case=='missing': s['results'].pop()
            if case=='duplicate': s['results'][-1]=s['results'][0]
            if case=='recovery': s['results'][0]['recovery_exact']=False
            with self.subTest(case=case), self.assertRaises(ValueError):
                report.summarize(s)

    def test_confusion_metrics_are_recomputed(self):
        s=self.fixture()
        s['results'][0]['confirmation']['default_metrics']['macro_f1'] += .01
        with self.assertRaisesRegex(ValueError,'Aggregate metric'):
            report.summarize(s)


if __name__=='__main__':
    unittest.main()

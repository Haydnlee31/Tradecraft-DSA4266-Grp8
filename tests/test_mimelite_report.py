"""Refuse incomplete, unequal-work or arithmetically inconsistent reports."""
import copy
import json
import unittest

from scripts import mimelite_report as report


class MimeLiteReportTests(unittest.TestCase):
    def fixture(self):
        p=report.study.read_plan()
        refs=json.loads((report.ROOT/'reports/methodology_comparison/results.json').read_text())['summary']['results']
        base=json.loads(report.study.old.training.PLAN.read_text())
        rows=[]
        for seed in p['seeds']:
            for arm,settings in p['arms'].items():
                ref=next(r for r in refs if r['seed']==seed and r['arm']=='fedavg_adam')
                rows.append({'arm':arm,'seed':seed,'recovery_exact':True,'recovery_backward_examples':2,
                    'initial_model_sha256':str(seed),
                    'job':{'contract':report.study.contract(),'seed':seed,'arm':arm,'settings':settings,
                           'schedule':report.study.schedule(base,arm,settings,p['backward_examples_per_run'])},
                    'historical_bridge':None if report.study.is_lite(arm) else
                        {k:True for k in ['model_exact','logits_exact','metrics_exact','work_exact','samples_exact']},
                    'work':report.expected_work(arm,settings,p['backward_examples_per_run']),
                    'development':copy.deepcopy(ref['development']),'training_seconds':1.})
        return {'status':'complete','plan':p,'contract':report.study.contract(),'results':rows,
                'confirmation_evaluated':False,'original_validation_evaluated':False,'test_evaluated':False}

    def test_complete_evidence(self):
        a=report.analyze(self.fixture())
        self.assertEqual(a['confusion_endpoints_checked'],126)
        self.assertEqual(a['historical_bridges_exact'],15)
        self.assertEqual(a['recovery_checks'],21)
        self.assertEqual(len(a['paired_comparisons']),36)

    def test_invalid_evidence_is_not_summarized(self):
        for case in ('missing','duplicate','recovery','bridge','work','metrics','test','settings','initialization'):
            s=self.fixture()
            if case=='missing':s['results'].pop()
            if case=='duplicate':s['results'][-1]=s['results'][0]
            if case=='recovery':s['results'][0]['recovery_exact']=False
            if case=='bridge':s['results'][0]['historical_bridge']['model_exact']=False
            if case=='work':s['results'][2]['work']['anchor_examples']-=1
            if case=='metrics':s['results'][0]['development']['audit_default_metrics']['macro_f1']+=.1
            if case=='test':s['test_evaluated']=True
            if case=='settings':s['results'][0]['job']['settings']={'lr':999.}
            if case=='initialization':s['results'][0]['initial_model_sha256']='changed'
            with self.subTest(case=case),self.assertRaises(ValueError):report.analyze(s)


if __name__=='__main__':unittest.main()

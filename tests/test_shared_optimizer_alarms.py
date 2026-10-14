"""Protect threshold fitting, duplicate groups and the no-retraining boundary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.federated_class_safeguard import state_digest
from src.eval.shared_optimizer_performance import initial_model
from src.eval.shared_optimizer_alarms import (PLAN, BENIGN, ATTACKS, calibrate, decisions, margins,
     metrics, evaluate_scores, grouped_split, load_development, run, replay, main)
from src.models.official_streaming import confusion_metrics
from src.models.research import atomic_json, atomic_save


class SharedOptimizerAlarmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_order_statistic_and_conservative_ties(self):
        r = calibrate(np.array([0.,1.,1.,2.,3.]),4000)
        self.assertEqual(r['threshold'],1.)
        self.assertEqual(r['observed_false_alerts'],2)
        self.assertEqual(r['boundary_ties'],2)
        tied = calibrate(np.array([0.,0.,0.,1.,1.]),2000)
        self.assertEqual(tied['allowed_false_alerts'],1)
        self.assertEqual(tied['observed_false_alerts'],0)
        self.assertEqual(calibrate(np.array([2.,3.]),0)['threshold'],3.)

    def test_monotonic_thresholds_exact_integer_caps(self):
        scores = np.random.default_rng(22).integers(-40,40,301).astype(float)
        points = [calibrate(scores,bps) for bps in [0,100,500,1000,2000,9999]]
        self.assertEqual([p['threshold'] for p in points],sorted([p['threshold'] for p in points],reverse=True))
        self.assertEqual([p['observed_false_alerts'] for p in points],sorted(p['observed_false_alerts'] for p in points))
        for p in points:
            self.assertLessEqual(p['observed_false_alerts'],301*p['target_basis_points']//10000)
            smaller = np.unique(scores[scores < p['threshold']])
            if len(smaller):
                self.assertGreater(int((scores > smaller[-1]).sum()),p['allowed_false_alerts'])

    def test_invalid_scores_and_budgets(self):
        for s in ([],[float('inf')],[float('nan')],[[1.,2.]]):
            with self.assertRaises(ValueError):
                calibrate(np.asarray(s),500)
        for b in (-1,10000,500.0,True):
            with self.assertRaises(ValueError):
                calibrate(np.arange(10.),b)
        with self.assertRaises(ValueError):
            margins(np.zeros((1,7)))

    def test_zero_margin_bridge_and_attack_tie_rule(self):
        logits = np.random.default_rng(4).normal(size=(100,8)).astype(np.float32)
        logits[:2] = 0.
        logits[1,ATTACKS[:2]] = 1.
        score, attack = margins(logits)
        np.testing.assert_array_equal(decisions(score,attack,0),logits.argmax(axis=1))
        self.assertEqual(attack[1],ATTACKS[0])
        for threshold in [-2.,0.,2.]:
            prediction = decisions(score,attack,threshold)
            self.assertTrue(((prediction == BENIGN) | (prediction == attack)).all())
        self.assertEqual(decisions(score,attack,1.)[1],BENIGN)

    def test_threshold_does_not_use_audit_labels_or_calibration_attacks(self):
        logits = np.random.default_rng(3).normal(size=(400,8)).astype(np.float32)
        y = np.arange(400,dtype=np.int64)%8
        calibration = np.arange(400) < 200
        a = evaluate_scores(logits,y,calibration,[100,500,2000])
        changed_y = y.copy()
        changed_y[~calibration] = 7-changed_y[~calibration]
        changed_logits = logits.copy()
        changed_logits[calibration & (y != BENIGN)] *= -100
        b = evaluate_scores(changed_logits,changed_y,calibration,[100,500,2000])
        self.assertEqual([p['calibration'] for p in a['operating_points']],
                         [p['calibration'] for p in b['operating_points']])

    def test_same_input_groups_and_signed_zeros_stay_together(self):
        x = np.arange(400,dtype=np.float32).reshape(100,4)
        x[1] = x[0]
        x[3] = 0.
        x[4] = -0.
        a,info = grouped_split(x,'unit-test:')
        self.assertEqual(a[0],a[1])
        self.assertEqual(a[3],a[4])
        self.assertEqual(info['cross_split_input_groups'],0)
        order = np.random.default_rng(2).permutation(len(x))
        b,_ = grouped_split(x[order],'unit-test:')
        np.testing.assert_array_equal(a[order],b)

    def test_detection_is_not_correct_attack_classification(self):
        y = np.arange(8,dtype=np.int64)
        prediction = np.array([BENIGN,*([int(ATTACKS[0])]*7)],dtype=np.int64)
        r = metrics(y,prediction)
        self.assertEqual(r['attack_detection_recall'],1.)
        self.assertAlmostEqual(r['macro_attack_class_recall'],1/7)
        self.assertEqual(r['per_class'][CLASSES[ATTACKS[-1]]]['recall'],0.)

    def test_calibration_cap_is_not_audit_guarantee(self):
        y = np.tile(np.arange(8,dtype=np.int64),2)
        calibration = np.arange(16) < 8
        logits = np.zeros((16,8),dtype=np.float32)
        logits[:,ATTACKS[0]] = 1.
        logits[8,ATTACKS[0]] = 3.
        p = evaluate_scores(logits,y,calibration,[500])['operating_points'][0]
        self.assertEqual(p['calibration']['observed_false_alerts'],0)
        self.assertEqual(p['audit_metrics']['benign_false_alert_rate'],1.)

    def test_end_to_end_saved_inference_and_independent_replay(self):
        plan = json.loads(PLAN.read_text())
        training_plan = json.loads((PLAN.parent/'performance-plan.json').read_text())
        rng = np.random.default_rng(191)
        x = rng.normal(size=(800,39)).astype(np.float32)
        y = np.arange(800,dtype=np.int64)%8
        plan['development_rows'] = len(y)
        model = initial_model(training_plan).eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x)).numpy()
        cm = np.bincount(y*8+logits.argmax(axis=1),minlength=64).reshape(8,8)
        contract = {'synthetic':True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data,training,output = root/'data',root/'training',root/'result'
            data.mkdir()
            files = {}
            for name,array in [('development_x.npy',x),('development_y.npy',y),
                               ('development_row_ids.npy',np.arange(800,dtype=np.int64))]:
                np.save(data/name,array,allow_pickle=False)
                files[name] = sha256(data/name)
            manifest = {'files':files,'features':list(range(39)),
                        'counts':{'development':dict(zip(CLASSES,[100]*8))}}
            atomic_json(manifest,data/'manifest.json')
            plan['data_manifest_sha256'] = sha256(data/'manifest.json')
            evidence = {'summary':{'data':manifest,'plan':training_plan,'results':[]},'audit':{'checks':[]}}
            for arm,lr in plan['selected_learning_rates'].items():
                destination = training/f'{arm}-lr{lr}'
                destination.mkdir(parents=True)
                atomic_save({'contract':contract,'state':{'model':model.state_dict()}},destination/'last.pt')
                evidence['summary']['results'].append({'arm':arm,'learning_rate':lr,'contract':contract,
                    'final_model_sha256':state_digest(model),'development_metrics':confusion_metrics(cm)})
                evidence['audit']['checks'].append({'arm':arm,'learning_rate':lr,'checkpoint_sha256':sha256(destination/'last.pt')})
            original_load = np.load
            opened = []
            def record_load(path,*args,**kwargs):
                opened.append(Path(path).name)
                return original_load(path,*args,**kwargs)
            with patch('src.eval.shared_optimizer_alarms.read_contract',return_value=(plan,evidence)), \
                 patch('torch.set_num_interop_threads'), patch('numpy.load',side_effect=record_load), \
                 patch('builtins.print'):
                result = run(data,training,output)
                check = replay(data,output,root/'replay.json')
            self.assertEqual(result['backward_examples'],0)
            self.assertEqual(result['new_model_fits'],0)
            self.assertEqual(result['forward_examples'],4000)
            self.assertTrue(check['replay_exact'])
            self.assertEqual(sum(c['operating_points_exact'] for c in check['checks']),20)
            self.assertFalse(result['confirmation_opened'])
            allowed = {'development_x.npy','development_y.npy','development_row_ids.npy','calibration_mask.npy'}
            allowed.update(f'{a}-logits.npy' for a in plan['selected_learning_rates'])
            self.assertLessEqual(set(opened),allowed)
            with self.assertRaisesRegex(ValueError,'Preserve'):
                with patch('src.eval.shared_optimizer_alarms.read_contract',return_value=(plan,evidence)):
                    run(data,training,output)
            # A modified input must be rejected, rather than silently reused.
            np.save(data/'development_y.npy',y[::-1],allow_pickle=False)
            with self.assertRaisesRegex(ValueError,'input differs'):
                load_development(data,plan,evidence)

    def test_cli_paths(self):
        response = {'status':'complete','backward_examples':0,'test_opened':False}
        with patch('sys.argv',['alarm','run','--data','d','--run-root','r','--output','o']), \
             patch('src.eval.shared_optimizer_alarms.run',return_value=response) as execute,patch('builtins.print'):
            main()
        execute.assert_called_once_with(Path('d'),Path('r'),Path('o'))
        with patch('sys.argv',['alarm','replay','--data','d','--output','o','--receipt','p']), \
             patch('src.eval.shared_optimizer_alarms.replay',return_value=response) as execute,patch('builtins.print'):
            main()
        execute.assert_called_once_with(Path('d'),Path('o'),Path('p'))


if __name__ == '__main__':
    unittest.main()

"""Small synthetic proofs before running the frozen real-data development screen."""
import copy
import io
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.shared_optimizer_data import PLAN, digest_array, split_rows, remove_precision_collisions, fit_scaler, owner_vector
from src.eval.shared_optimizer_performance import (work_plan, make_state, sampled_rows, advance,
                snapshot, restore, evaluate_development, screen)
from src.models.official_losses import build_official_criterion


class SharedOptimizerPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def plan(self):
        p = json.loads(PLAN.read_text())
        p.update(fit_rows=40, clients=2, batch_size=4, anchor_batch_size=9,
                 backward_examples_per_arm=144)
        p['mime'].update(local_steps=2, rounds=2)
        return p

    def fixture(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(40, 39)).astype(np.float32)
        y = np.arange(40) % len(CLASSES)
        data = SimpleNamespace(arrays={'fit': (x,y), 'development': (x[:16],y[:16])},
                               parts=[np.arange(13), np.arange(13,40)])
        data.batch = lambda rows: (torch.from_numpy(x[rows].copy()), torch.from_numpy(y[rows].copy()))
        criterion = build_official_criterion('sqrt_weighted_ce', dict(zip(CLASSES,[5]*8)), 'fixed_train_mean')
        return data, criterion

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a,b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (tuple,list)):
            self.assertEqual(len(a), len(b))
            for x,y in zip(a,b):
                self.assert_nested_equal(x,y)
        else:
            self.assertEqual(a,b)

    def test_frozen_exact_budgets(self):
        p = json.loads(PLAN.read_text())
        self.assertEqual({a: work_plan(p,a)[0] for a in p['arms']},
                         {'central_adam':43200, 'shared_adam':2160, 'fedavg_adam':108,
                          'central_mime_base':43200, 'mime':24})
        self.assertEqual(p['total_backward_examples'], 216000000)

    def test_serialized_recovery_every_arm_and_work(self):
        p, (data, criterion) = self.plan(), self.fixture()
        for arm in p['arms']:
            with self.subTest(arm=arm):
                state = make_state(p, arm, .001)
                units, per_unit = work_plan(p,arm)
                for _ in range(units//2):
                    advance(state,data,p,arm,.001,criterion)
                stream = io.BytesIO()
                torch.save(snapshot(state),stream)
                stream.seek(0)
                resumed = restore(torch.load(stream,weights_only=True),p,arm,.001)
                for _ in range(units-units//2):
                    advance(state,data,p,arm,.001,criterion)
                    advance(resumed,data,p,arm,.001,criterion)
                self.assert_nested_equal(snapshot(state),snapshot(resumed))
                self.assertEqual(state['work']['backward_examples'],144)
                self.assertEqual(state['unit']*per_unit,144)
                self.assertEqual(sum(map(sum,evaluate_development(state['model'],data)['confusion_matrix'])),16)
                if arm == 'mime':
                    self.assertEqual(state['work']['anchor_examples'],80)
                    self.assertEqual(state['work']['local_examples'],32)
                    self.assertEqual(state['work']['backward_batches'],26)
                    self.assertEqual(state['work']['logical_barriers'],4)

    def test_samples_stay_local_and_repeat(self):
        p, (data, _) = self.plan(), self.fixture()
        for client in range(2):
            a = sampled_rows(data,p,3,client,1)
            self.assertTrue(np.isin(a,data.parts[client]).all())
            np.testing.assert_array_equal(a,sampled_rows(data,p,3,client,1))

    def test_split_is_order_invariant_and_refuses_duplicates(self):
        d = digest_array(np.arange(4000,dtype=float).reshape(1000,4))
        split = split_rows(d,'unit-test:',200)
        shuffled = np.random.default_rng(2).permutation(len(d))
        np.testing.assert_array_equal(split[shuffled],split_rows(d[shuffled],'unit-test:',200))
        self.assertEqual(int((split==0).sum()),200)
        with self.assertRaisesRegex(ValueError,'not unique'):
            split_rows(np.concatenate([d,d[:1]]),'unit-test:',200)
        np.testing.assert_array_equal(digest_array([[0.,-0.]]),digest_array([[-0.,0.]]))

    def test_precision_collision_policy(self):
        h = digest_array([[1.],[1.],[2.],[2.],[3.],[1.]])
        s, removed = remove_precision_collisions(np.array([0,1,1,2,2,-1]),h)
        np.testing.assert_array_equal(s,[0,-1,1,-1,2,-1])
        self.assertEqual(removed,{'development':1,'confirmation':1})

    def test_scaler_ignores_reserved_values(self):
        x = np.array([[1.,2.],[3.,4.],[1e10,-1e10]])
        scaler = fit_scaler(x,np.array([0,1]),1)
        np.testing.assert_array_equal(scaler.mean_,[2.,3.])
        np.testing.assert_array_equal(scaler.var_,[1.,1.])
        self.assertEqual(scaler.n_samples_seen_,2)

    def test_ownership_rejects_missing_duplicate_and_out_of_range(self):
        np.testing.assert_array_equal(owner_vector([np.array([0,2]),np.array([1,3])],4),[0,1,0,1])
        for parts in ([np.array([0,0]),np.array([1,2,3])], [np.array([0,1]),np.array([1,2,3])],
                      [np.array([0]),np.array([1,2])], [np.array([0,4]),np.array([1,2,3])]):
            with self.assertRaises(ValueError):
                owner_vector(parts,4)

    def test_screen_does_not_hide_recall_or_alarm_regression(self):
        p = self.plan()
        m = {'macro_f1':.65,'benign_false_alert_rate':.15,
             'per_class':{c:{'recall':.5} for c in CLASSES}}
        selected = {a:{'development_metrics':copy.deepcopy(m)} for a in p['arms']}
        selected['fedavg_adam']['development_metrics']['macro_f1'] = .60
        self.assertTrue(screen(selected,p)['mime']['passes_development_screen'])
        selected['mime']['development_metrics']['per_class'][CLASSES[-1]]['recall'] = .4
        self.assertFalse(screen(selected,p)['mime']['passes_development_screen'])
        selected['shared_adam']['development_metrics']['benign_false_alert_rate'] = .16
        self.assertFalse(screen(selected,p)['shared_adam']['passes_development_screen'])

    def test_auditor_cli_argument_names(self):
        from pathlib import Path
        from src.eval.shared_optimizer_audit import main
        with patch('sys.argv',['audit','--data','data','--run-root','runs','--output','audit.json']), \
             patch('src.eval.shared_optimizer_audit.audit',return_value={'status':'complete'}) as execute, \
             patch('builtins.print'):
            main()
        execute.assert_called_once_with(data_root=Path('data'),run_root=Path('runs'),output=Path('audit.json'))


if __name__ == '__main__':
    unittest.main()

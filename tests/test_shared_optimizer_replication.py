"""Fixed seeds, permitted files, restart boundaries and paired comparisons."""
import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import shared_optimizer_replication as rep
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_alarms as alarms
from src.eval.federated_class_safeguard import state_digest
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_json, atomic_save


class SharedOptimizerReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        base = json.loads(training.PLAN.read_text())
        base.update(fit_rows=80, clients=2, batch_size=4, backward_examples_per_arm=64, seed=17)
        base['mime']['local_steps'] = 2
        x = np.random.default_rng(11).normal(size=(80,39)).astype(np.float32)
        y = np.arange(80,dtype=np.int64)%8
        np.save(root/'fit_x.npy',x,allow_pickle=False)
        np.save(root/'fit_y.npy',y,allow_pickle=False)
        np.savez(root/'clients.npz',client_0=np.arange(40),client_1=np.arange(40,80))
        m = {'status':'complete','features':list(range(39)),'classes':CLASSES,
             'files':{n:sha256(root/n) for n in ('fit_x.npy','fit_y.npy','clients.npz')},
             'counts':{'fit':dict(zip(CLASSES,[10]*8))},'fit_client_counts':[40,40]}
        atomic_json(m,root/'manifest.json')
        return base, m

    def test_frozen_contract_and_work(self):
        p,a,e,_ = rep.read_contract()
        self.assertEqual(p['seeds'],[17,27])
        self.assertEqual(p['new_backward_examples'],129600000)
        for arm,units in zip(p['arms'],[43200,2160,108]):
            self.assertEqual(training.work_plan(e['summary']['plan'],arm)[0],units)
        self.assertEqual(p['primary_target_basis_points'],500)
        self.assertFalse(p['test_scoring_permitted'])

    def test_loader_only_opens_fit_whitelist(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); base,m = self.fixture(root)
            original = np.load
            seen=[]
            def checked(path,*args,**kwargs):
                seen.append(Path(path).name)
                self.assertIn(Path(path).name,('fit_x.npy','fit_y.npy','clients.npz'))
                return original(path,*args,**kwargs)
            with patch.object(rep.np,'load',side_effect=checked):
                data = rep.FitData(root,base,sha256(root/'manifest.json'))
            self.assertEqual(set(seen),{'fit_x.npy','fit_y.npy','clients.npz'})
            self.assertEqual(tuple(data.batch(np.array([1,3]))[0].shape),(2,39))
            with self.assertRaisesRegex(ValueError,'manifest differs'):
                rep.FitData(root,base,'wrong')

    def test_each_arm_resumes_exactly_and_rejects_other_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); base,m = self.fixture(root)
            data=rep.FitData(root,base,sha256(root/'manifest.json'))
            criterion=build_official_criterion('sqrt_weighted_ce',m['counts']['fit'],'fixed_train_mean')
            for arm in ('central_adam','shared_adam','fedavg_adam'):
                lr=.003
                full=training.make_state(base,arm,lr)
                units,_=training.work_plan(base,arm)
                for _ in range(units):
                    training.advance(full,data,base,arm,lr,criterion)
                paused=training.make_state(base,arm,lr)
                for _ in range(units//2):
                    training.advance(paused,data,base,arm,lr,criterion)
                destination=root/arm; destination.mkdir()
                identity={'contract':{'test':True},'seed':17,'arm':arm,'learning_rate':lr}
                atomic_save({'identity':identity,'state':training.snapshot(paused),'training_seconds':0.},destination/'last.pt')
                resumed,_=rep.train_candidate(data,base,arm,lr,destination,{'test':True},criterion)
                self.assertEqual(state_digest(full['model']),state_digest(resumed['model']))
                self.assertEqual(full['work'],resumed['work'])
                self.assertEqual(full['sample_chain_sha256'],resumed['sample_chain_sha256'])
                with self.assertRaisesRegex(ValueError,'identity differs'):
                    rep.train_candidate(data,{**base,'seed':27},arm,lr,destination,{'test':True},criterion)

    def test_initialization_pairing_and_sampling(self):
        base=json.loads(training.PLAN.read_text())
        for seed in (17,27):
            recipe={**base,'seed':seed}
            hashes={state_digest(training.make_state(recipe,a,.003)['model'])
                    for a in ('central_adam','shared_adam','fedavg_adam')}
            self.assertEqual(len(hashes),1)
        self.assertNotEqual(state_digest(training.initial_model({**base,'seed':17})),
                            state_digest(training.initial_model({**base,'seed':27})))

    def test_federated_arms_have_identical_sample_multisets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); base,m=self.fixture(root)
            data=rep.FitData(root,base,sha256(root/'manifest.json'))
            c,k=base['clients'],base['mime']['local_steps']
            # Shared consumes step-then-client; FedAvg consumes client-then-step.
            shared=[training.sampled_rows(data,base,u//k,client=i,step=u%k).tobytes()
                    for u in range(training.work_plan(base,'shared_adam')[0]) for i in range(c)]
            fed=[training.sampled_rows(data,base,r,client=i,step=s).tobytes()
                 for r in range(training.work_plan(base,'fedavg_adam')[0]) for i in range(c) for s in range(k)]
            self.assertEqual(Counter(shared),Counter(fed))
            self.assertNotEqual(shared,fed)

    def test_all_work_counters_and_twentyfold_payload(self):
        base=json.loads(training.PLAN.read_text())
        central=rep.expected_work(base,'central_adam',20384)
        shared=rep.expected_work(base,'shared_adam',20384)
        fed=rep.expected_work(base,'fedavg_adam',20384)
        self.assertEqual([central['optimizer_steps'],shared['optimizer_steps'],fed['optimizer_steps']],
                         [43200,2160,43200])
        self.assertEqual(shared['estimated_payload_bytes'],1761177600)
        self.assertEqual(fed['estimated_payload_bytes'],88058880)
        self.assertEqual(shared['estimated_payload_bytes'],20*fed['estimated_payload_bytes'])

    def test_descriptive_statistics_are_sample_not_population_sd(self):
        self.assertEqual(rep.statistics([1.,2.,3.]),{'mean':2.,'sample_sd':1.,'min':1.,'max':3.,'n':3})
        with self.assertRaises(ValueError):
            rep.statistics([1.])

    def test_complete_toy_run_and_separate_audit(self):
        p,alarm_plan,performance,old=rep.read_contract()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); base,m=self.fixture(root)
            base['seed']=7
            performance=copy.deepcopy(performance)
            performance['summary']['plan']=base
            p=copy.deepcopy(p); p['data_manifest_sha256']=sha256(root/'manifest.json')
            p['backward_examples_per_run']=64; p['new_backward_examples']=384
            x=np.random.default_rng(55).normal(size=(800,39)).astype(np.float32)
            y=np.arange(800,dtype=np.int64)%8
            mask,groups=alarms.grouped_split(x,alarm_plan['split_domain'])
            np.save(root/'mask.npy',mask,allow_pickle=False)
            p['calibration_mask_sha256']=sha256(root/'mask.npy')
            old=copy.deepcopy(old); old['summary']['split']=groups
            contract_return=(p,alarm_plan,performance,old)
            output=root/'run'
            with patch.object(rep,'read_contract',return_value=contract_return), \
                 patch.object(alarms,'load_development',return_value=(x,y,np.arange(len(y)))), \
                 patch.object(rep.subprocess,'check_output',return_value='synthetic-test-commit\n'):
                # A source archive has no .git directory. Runtime provenance is
                # not under test here; the real-run receipt pins its actual commit.
                summary=rep.run(root,output)
                self.assertEqual(summary['new_training_runs'],6)
                self.assertEqual(summary['new_backward_examples'],384)
                audit=rep.audit(root,output,root/'audit.json')
                self.assertEqual(len(audit['checks']),6)
                self.assertFalse(audit['test_opened'])
                with self.assertRaisesRegex(ValueError,'Completed summary'):
                    rep.run(root,output)
                with self.assertRaisesRegex(ValueError,'Preserve existing audit'):
                    rep.audit(root,output,root/'audit.json')
                # A modified saved checkpoint must not be accepted by replay.
                saved=output/'seed17-shared_adam'/'last.pt'
                checkpoint=torch.load(saved,weights_only=True)
                checkpoint['state']['unit']+=1
                atomic_save(checkpoint,saved)
                with self.assertRaisesRegex(ValueError,'Saved artifacts differ'):
                    rep.audit(root,output,root/'tamper-audit.json')
            with self.assertRaisesRegex(ValueError,'Missing/duplicate'):
                rep.paired_comparisons(summary['results'][:-1],p)


if __name__ == '__main__':
    unittest.main()

"""Receipt tampering checks using synthetic arrays, without model fitting."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import official39_hard_example as run


class HardReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ident = {'fixture': True}
        y = np.arange(409918,dtype=np.int64)%8
        pred = np.zeros(len(y),np.uint8)
        pop = [1000]*8
        files = {}
        for n,a in [('fit-rows.npy',np.arange(16)),('calibration-rows.npy',np.arange(len(y))+16)]:
            files[n] = run.previous.previous.save_array(self.root,n,a)
        run.frozen.save(self.root/'edges.json',[[0,1]])
        files['edges.json'] = run.digest(self.root/'edges.json')
        recipe = dict(identity=self.ident,files=copy.deepcopy(files),fit_population_counts=pop)
        files['calibration-labels.npy'] = run.previous.previous.save_array(self.root,'calibration-labels.npy',y)
        models = []
        for arm,forced in zip(run.ARMS,[0,7434240]):
            n = arm+'.pt'
            (self.root/n).write_bytes(b'checksum-only synthetic model fixture')
            files[n] = run.digest(self.root/n)
            models.append(dict(arm=arm,file=n,sha256=files[n],settings=run.core.SETTINGS,
                completed_epochs=60,examples_processed=16220160,optimizer_steps=31680,
                mining_examples_scored=16219560,sampled_class_counts=[2027520]*8,
                forced_hard_positions=forced,mining=[dict(epoch=i) for i in range(1,61)],
                initial_state_sha256='initial',final_state_sha256='final',class_weights=[1.]*8,
                parameters=18584,deployment_parameters=17000,observations=[dict(epoch=1),dict(epoch=5)]))
        score = run.prior.old.prior.prior.scored(y,pred,pop)
        scores = {}
        for n in run.ARMS+run.COMPARATORS:
            files[n+'-calibration.npy'] = run.previous.previous.save_array(self.root,n+'-calibration.npy',pred)
            scores[n] = copy.deepcopy(score)
        self.r = dict(status='complete',recipe=recipe,files=files,models=models,scores=scores,
            control_historical_bridge_exact=True,examples_processed=32440320,optimizer_steps=63360,
            mining_examples_scored=32439120,test_opened=False,test_evaluated=False,audit_scored=False,
            cloud_used=False,independent_confirmation=False,deployment_authorized=False,
            continuation_checks={'tree':False,run.ARMS[0]:False},continue_to_confirmation=False)
        self.reference = self.root/'historical.json'
        run.frozen.save(self.reference,dict(models=[dict(final_state_sha256='final')],
            scores={n:copy.deepcopy(score) for n in ['tree','ple','balanced_ce_control']}))
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        run.frozen.save(self.root/'frozen-recipe.json',self.r['recipe'])
        run.frozen.save(self.root/'models-frozen.json',self.r['models'])
        run.frozen.save(self.root/'receipt.json',self.r)

    def verify(self):
        with patch.object(run,'identity',return_value=self.ident),patch.object(run,'REFERENCE',self.reference):
            return run.verify(self.root)

    def test_valid(self):
        self.assertEqual(self.verify(),self.r)

    def test_changed_prediction_with_updated_hash_fails(self):
        n=run.ARMS[1]+'-calibration.npy'
        p=np.load(self.root/n,allow_pickle=False); p[0]=1
        np.save(self.root/n,p,allow_pickle=False)
        self.r['files'][n]=run.digest(self.root/n)
        self.save()
        with self.assertRaisesRegex(ValueError,'predictions/scores'):
            self.verify()

    def test_changed_budget_fails(self):
        self.r['models'][1]['forced_hard_positions']+=1
        self.save()
        with self.assertRaisesRegex(ValueError,'Model/work'):
            self.verify()

    def test_changed_bridge_fails(self):
        self.r['models'][0]['final_state_sha256']='wrong'
        self.save()
        with self.assertRaisesRegex(ValueError,'Historical bridge'):
            self.verify()

    def test_fabricated_continuation_fails(self):
        self.r['continue_to_confirmation']=True
        self.save()
        with self.assertRaisesRegex(ValueError,'Continuation'):
            self.verify()

    def test_no_overwrite_or_self_replay(self):
        with self.assertRaises(ValueError):
            run.run(self.root)
        with self.assertRaises(ValueError):
            run.publish(self.root,self.root,self.root/'new.json')


if __name__=='__main__':
    unittest.main()

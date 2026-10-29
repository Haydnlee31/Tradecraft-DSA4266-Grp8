"""Native tree checks in a subprocess; never mix torch and XGBoost runtimes."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


class TreeRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'optional XGBoost wheel absent')
    def test_intercept_bridge_and_fedlc_changes_only_training_offsets(self):
        code = '''
import numpy as np
import xgboost as xgb
from src.eval import federated_xgboost_mechanisms as fm
rng=np.random.default_rng(18)
x=rng.normal(size=(240,4)).astype('float32')
y=np.concatenate([rng.choice(8,80,p=rng.dirichlet(np.ones(8))) for _ in range(3)])
a={'fit_x':x,'fit_y':y}
parts=list(np.split(np.arange(len(y)),3))
config={'objective':'multi:softprob','num_class':8,'tree_method':'hist','max_depth':2,'eta':.05,'lambda':1.,'max_bin':32,'nthread':2,'rounds':30,'weights':'sqrt','multi_strategy':'one_output_per_tree'}
models=[fm.cyclic(a,parts,config,kind,7,xgb)[0] for kind in ('none','zero_offset_bridge','fedlc')]
pred=[m.predict(xgb.DMatrix(x),output_margin=True) for m in models]
assert np.array_equal(pred[0],pred[1]), 'Explicit zero offsets must preserve the fitted intercept'
assert not np.array_equal(pred[0],pred[2]), 'FedLC treatment must change fitted trees'
assert all(m.num_boosted_rounds()==30 for m in models)
print('PASS: native intercept bridge and FedLC tree treatment')
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':
    unittest.main()

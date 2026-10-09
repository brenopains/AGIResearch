import os
import platform
import sys
if sys.platform=="win32":
    platform.machine=platform._get_machine_win32
    platform.system=lambda:"Windows"
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="2"
import unittest
import numpy as np
from transporte_compacto import unique_numeric_rows,CompactTransport
from transporte_funcoes import FunctionTransport


class CompactEquivalence(unittest.TestCase):
    def test_sort_and_dedup_for_signed_numbers(self):
        x=np.array([[0.,-2,1],[-0.,-2,1],[0.,-2,-1],[-3.,0.,0.],[4.,1.,0.],[4.,1.,0.]])
        self.assertTrue(np.array_equal(unique_numeric_rows(x),np.unique(x,axis=0)))

    def test_float_scales_and_static_prefix(self):
        rng=np.random.default_rng(733)
        for dtype in (np.float32,np.float64):
            x=np.zeros((90,500),dtype)
            x[:,492:]=rng.normal(size=(90,8)).astype(dtype)
            x[-20:]=x[:20]
            self.assertTrue(np.array_equal(unique_numeric_rows(x),np.unique(x,axis=0)))

    def test_numeric_extremes(self):
        x=np.array([[-1e300,0],[-1e-300,1],[1e-300,-1],[1e300,1],[-0.,0],[0.,0]])
        self.assertTrue(np.array_equal(unique_numeric_rows(x),np.unique(x,axis=0)))

    def test_dictionary_and_programs_original_and_shuffled(self):
        rng=np.random.default_rng(109)
        source=rng.normal(size=(120,100))
        after=.7*source+rng.normal(size=source.shape)
        actions=np.arange(120)%4
        reward=source[:,1]-source[:,4]
        continuation=(np.arange(120)%9!=0).astype(float)
        for shuffled in (False,True):
            old=FunctionTransport(supports=12).fit(source,after,actions,reward,continuation,shuffled)
            new=CompactTransport(supports=12).fit(source,after,actions,reward,continuation,shuffled)
            self.assertTrue(np.array_equal(old.support,new.support))
            self.assertEqual(old.scale,new.scale)
            self.assertTrue(np.array_equal(old.reward,new.reward))
            self.assertTrue(np.array_equal(old.transport,new.transport))
            words=[[0,1,3,2]*4]*10
            self.assertTrue(np.array_equal(old.predict(source[:10],words),new.predict(source[:10],words)))

    def test_all_identical_rows(self):
        x=np.ones((20,100))
        self.assertTrue(np.array_equal(unique_numeric_rows(x),np.unique(x,axis=0)))

    def test_invalid_values_rejected(self):
        for x in (np.empty((0,3)),np.array([[float("nan")]]),np.ones((2,2),int)):
            with self.assertRaises(ValueError):
                unique_numeric_rows(x)


if __name__=="__main__":
    unittest.main()

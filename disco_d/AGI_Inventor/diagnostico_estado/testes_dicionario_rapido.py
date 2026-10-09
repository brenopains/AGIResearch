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
from dicionario_rapido import exact_unique_sorted,FastFunctionTransport
from transporte_funcoes import FunctionTransport


class DictionaryEquivalence(unittest.TestCase):
    def test_numeric_order_with_duplicates_and_zeros(self):
        x=np.array([[0.,-2,1],[-0.,-2,1],[0.,-2,-1],[-3.,0.,0.],[-3.,0.,0.],[4.,1.,0.]])
        self.assertTrue(np.array_equal(np.unique(x,axis=0),exact_unique_sorted(x)))

    def test_late_columns_and_static_prefix(self):
        x=np.zeros((40,1000))
        x[:,997:]=np.random.default_rng(42).integers(-2,3,(40,3))
        self.assertTrue(np.array_equal(np.unique(x,axis=0),exact_unique_sorted(x)))

    def test_all_duplicate(self):
        x=np.ones((30,100))
        self.assertTrue(np.array_equal(np.unique(x,axis=0),exact_unique_sorted(x)))

    def test_original_model_dictionary_and_predictions(self):
        rng=np.random.default_rng(98)
        source=rng.normal(size=(100,80))
        after=.7*source+rng.normal(size=source.shape)
        actions=np.arange(100)%4
        reward=source[:,2]-source[:,5]
        continuation=(np.arange(100)%9!=0).astype(float)
        slow=FunctionTransport(supports=12).fit(source,after,actions,reward,continuation)
        fast=FastFunctionTransport(supports=12).fit(source,after,actions,reward,continuation)
        self.assertTrue(np.array_equal(slow.support,fast.support))
        self.assertEqual(slow.scale,fast.scale)
        self.assertTrue(np.array_equal(slow.reward,fast.reward))
        self.assertTrue(np.array_equal(slow.transport,fast.transport))
        words=[[0,1,3,2]]*10
        self.assertTrue(np.array_equal(slow.predict(source[:10],words),fast.predict(source[:10],words)))


if __name__=="__main__":
    unittest.main()

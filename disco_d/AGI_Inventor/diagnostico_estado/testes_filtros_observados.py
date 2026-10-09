import os
import platform
import sys
if sys.platform=="win32":
    platform.machine=platform._get_machine_win32
    platform.system=lambda:"Windows"
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="2"
import tempfile
import unittest
from pathlib import Path
import numpy as np
from filtros_observados import ObservedFilters


class ObservedFunctionTests(unittest.TestCase):
    def test_learns_from_changes_not_labels(self):
        x=np.random.default_rng(43).normal(size=(300,100))
        x[:,4]+=np.random.default_rng(44).normal(0,20,300)
        body=ObservedFilters.fit(x)
        self.assertEqual(np.argmax(np.abs(body.weights[0])),4)
        self.assertTrue(np.allclose(body.weights.sum(1),0))

    def test_constant_patch_does_not_create_contrast(self):
        y=ObservedFilters.random().encode(np.full((2,210,160),90,np.uint8))
        self.assertTrue(np.allclose(y[:,1:],0,atol=1e-7))

    def test_serialized_body_preserved(self):
        body=ObservedFilters.fit(np.random.default_rng(55).normal(size=(150,100)))
        frames=np.random.default_rng(56).integers(0,256,(3,210,160),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"functions.npz"
            body.save(path,{"test_only":True})
            self.assertTrue(np.array_equal(body.encode(frames),ObservedFilters.load(path).encode(frames)))
            with self.assertRaises(RuntimeError):
                body.save(path,{})


if __name__=="__main__":
    unittest.main()

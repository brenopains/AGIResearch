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
from sensor_resolucao import coarse,moments,windows,encoded_windows
from coleta import small


class SensorContracts(unittest.TestCase):
    def test_original_compression_matches_bitwise(self):
        frames=np.random.default_rng(123).integers(0,256,(20,210,160),dtype=np.uint8)
        self.assertTrue(np.array_equal(coarse(frames),np.stack([small(f,21).reshape(336) for f in frames])))

    def test_subcell_translation_disappears_in_average_not_moment(self):
        frames=np.zeros((2,210,160),np.uint8)
        frames[0,5,2]=255
        frames[1,5,7]=255
        self.assertTrue(np.array_equal(coarse(frames[:1]),coarse(frames[1:])))
        self.assertFalse(np.array_equal(moments(frames[:1]),moments(frames[1:])))
        # This proves a codec distinction on fabricated inputs, not Atari learning.

    def test_uniform_patch_has_zero_first_moments(self):
        z=moments(np.full((1,210,160),127,np.uint8))
        self.assertTrue(np.allclose(z[:,1:3],0,atol=1e-7))

    def test_windows_terminal_padding_is_exact(self):
        frames=np.random.default_rng(99).integers(0,256,(24,210,160),dtype=np.uint8)
        ends=np.array([16,24,24])
        full=windows(frames,ends)
        self.assertTrue(np.array_equal(full[1],full[2]))
        self.assertTrue(np.array_equal(encoded_windows(frames,ends,"coarse"),np.stack([coarse(w) for w in full])))

    def test_invalid_endpoints_and_inputs(self):
        frames=np.zeros((16,210,160),np.uint8)
        for ends in ([15],[17],[16,15],[16.0]):
            with self.assertRaises(ValueError):
                windows(frames,np.array(ends))
        with self.assertRaises(ValueError):
            coarse(frames.astype(float))


if __name__=="__main__":
    unittest.main()

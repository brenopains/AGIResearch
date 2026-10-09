"""Composição não comutativa, término e recuperação do corpo exportado."""

import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import tempfile
import unittest
from pathlib import Path
import numpy as np
from transporte_funcoes import FunctionTransport


class Contracts(unittest.TestCase):
    def model(self):
        m = FunctionTransport(actions=2, supports=1)
        m.support = np.zeros((1, 1))
        m.scale = 1.
        m.reward = np.asarray([[0., 1.], [2., -1.]])
        m.transport = np.asarray([[[1., 1.], [0., 2.]], [[1., 0.], [1., 1.]]])
        m.initial_prior = np.ones(2)
        m.action_counts, m.local_residual_mse = [4, 4], [0., 0.]
        m.mode = "transport"
        return m

    def test_matrix_composition_equals_function_execution(self):
        m = self.model()
        for word in ([], [0], [1], [0, 1], [1, 0], [0, 1, 0]):
            state = np.concatenate(([0.], m.features([[.3]])[0]))
            result = state @ m.augmented(word)
            self.assertAlmostEqual(result[0], float(m.predict([[.3]], [word])[0]), places=12)
            for k in range(len(word) + 1):
                self.assertTrue(np.allclose(m.augmented(word[:k]) @ m.augmented(word[k:]), m.augmented(word)))
        self.assertFalse(np.allclose(m.augmented([0, 1]), m.augmented([1, 0])))

    def test_terminal_reward_then_absorbing_zero(self):
        m = self.model()
        m.transport[0] = 0
        self.assertAlmostEqual(float(m.predict([[0.]], [[0, 1, 1]])[0]), float(m.predict([[0.]], [[0]])[0]))

    def test_save_load_and_preserve(self):
        m = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body.npz"
            m.save(path)
            loaded = FunctionTransport.load(path)
            self.assertTrue(np.allclose(m.predict([[.2]], [[0, 1]]), loaded.predict([[.2]], [[0, 1]])))
            with self.assertRaises(RuntimeError):
                m.save(path)

    def test_invalid_data_and_missing_action(self):
        m = FunctionTransport(actions=2, supports=2)
        with self.assertRaises(ValueError):
            m.fit([[0.], [1.]], [[1.], [2.]], [0, 0], [0., 1.], [1., 1.])
        with self.assertRaises(ValueError):
            m.fit([[0.], [1.]], [[1.], [2.]], [0, 1], [0., 1.], [1., .3])
        with self.assertRaises(ValueError):
            self.model().word_value([7])

    def test_no_context_and_no_transport_controls(self):
        m = self.model()
        m.mode = "no_context"
        self.assertAlmostEqual(float(m.predict([[0.]], [[0, 1]])[0]), float(m.predict([[9.]], [[0, 1]])[0]))
        m.mode = "no_transport"
        self.assertTrue(np.allclose(m.word_value([0, 1]), m.word_value([1, 0])))


if __name__ == "__main__":
    unittest.main()

"""Corpos não lineares, produtos sem pais promovidos e controles de informação."""

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
from funcoes_contraexemplos import PairFunctions, GeneratedTransport, propose, fit_features
from transporte_funcoes import FunctionTransport


class FunctionTests(unittest.TestCase):
    def test_pair_anchors_and_nonlinear_product(self):
        functions = PairFunctions([[2., 0.], [0., 2.]], [0., 0.],
            [{"op": "unit", "units": [0]}, {"op": "unit", "units": [1]}, {"op": "product", "units": [0, 1]}], [])
        raw = np.asarray([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
        values = functions.apply(raw)
        self.assertTrue(np.array_equal(np.sign(values[:, 2]), [1, -1, -1, 1]))
        self.assertTrue(np.allclose(values[:, 2], values[:, 0] * values[:, 1]))
        # Product adds a distinction outside the affine span of its two parents.
        affine = np.column_stack((np.ones(4), values[:, :2]))
        projected = affine @ np.linalg.lstsq(affine, values[:, 2], rcond=None)[0]
        self.assertGreater(np.mean((projected - values[:, 2]) ** 2), .1)
        self.assertTrue(np.array_equal(functions.apply(raw, [None, 2])[:, 0], np.zeros(4)))

    def test_generation_uses_independent_episode_pairs(self):
        raw = np.asarray([[0.], [.2], [.8], [1.]])
        actions, reward, episodes = np.zeros(4, int), np.asarray([0., 0., 1., 1.]), np.asarray([0, 1, 0, 1])
        functions, stats = propose(raw, actions, reward, episodes, np.column_stack((np.ones(4), raw)), "counterexample_pairs")
        self.assertGreater(stats["candidate_functions"], 0)
        for origin in functions.origins:
            self.assertNotEqual(*origin["episodes"])
            high, low = origin["rows"]
            self.assertGreater(functions.apply(raw[[high, low]])[0, functions.origins.index(origin)], 0)
        self.assertEqual(stats["validation_targets_used"], 0)

    def test_no_contrast_does_not_create_fake_witness(self):
        raw = np.asarray([[0.], [.2], [.8], [1.]])
        functions, stats = propose(raw, np.zeros(4, int), np.zeros(4), np.asarray([0, 1, 0, 1]), np.column_stack((np.ones(4), raw)))
        self.assertEqual(stats["candidate_functions"], 0)
        self.assertEqual(functions.apply(raw).shape, (4, 0))

    def test_interpreter_rejects_unknown_operations(self):
        with self.assertRaises(ValueError):
            PairFunctions([[1.]], [0.], [{"op": "python", "units": [0]}], [])

    def test_full_export_reexecution(self):
        source = np.asarray([[0.], [.3], [.6], [1.]])
        target = source[::-1].copy()
        actions, reward = np.asarray([0, 1, 0, 1]), np.asarray([0., 1., 0., 1.])
        alive = np.ones(4)
        base = FunctionTransport(2, 2).fit(source, target, actions, reward, alive)
        functions = PairFunctions([[2.]], [1.], [{"op": "unit", "units": [0]}], [])
        new = GeneratedTransport(base, functions, [0])
        body = fit_features(base, functions, [0], new.features(source), new.features(target), reward, alive, actions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body.npz"
            body.save(path)
            restored = GeneratedTransport.load(path)
            self.assertTrue(np.array_equal(body.comparison(source, [[0, 1]] * 4, [[1, 0]] * 4),
                                           restored.comparison(source, [[0, 1]] * 4, [[1, 0]] * 4)))
            with self.assertRaises(RuntimeError):
                body.save(path)


if __name__ == "__main__":
    unittest.main()

"""Testes de tempo da observação, conservação, término e persistência."""
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
from leitor_eventos import EventReader, PotentialPlanner, design, potential_constraints


class EventTests(unittest.TestCase):
    def test_before_reader_never_uses_future(self):
        p = np.column_stack((np.ones(8), np.arange(8)))
        actions = np.tile(np.arange(4), 2)
        body = EventReader("before").fit(p, p + [0, 2], actions, np.arange(8))
        np.testing.assert_array_equal(body.predict(p, p + [0, 10], actions), body.predict(p, None, actions))

    def test_after_reader_requires_future_observation(self):
        with self.assertRaises(ValueError):
            design(np.ones((2, 3)), None, "transition")

    def test_learned_field_telescopes(self):
        p = np.column_stack((np.ones(8), np.arange(8)))
        q = np.column_stack((np.ones(8), np.arange(8) + np.tile([1., 2.], 4)))
        body = EventReader("potential").fit(p, q, np.tile(np.arange(4), 2), (q - p)[:, 1])
        a, b, c = p[:1], q[:1], q[:1] + [0, 2]
        self.assertAlmostEqual(float(body.predict(a, b, [0])[0] + body.predict(b, c, [1])[0]),
            float(body.predict(a, c, [2])[0]), places=12)

    def test_nonconservative_cycle_rejected_and_consistent_cycle_kept(self):
        self.assertTrue(potential_constraints(["a", "b", "c"], ["b", "c", "a"], [1, 2, -3])["compatible_on_observed_graph"])
        self.assertFalse(potential_constraints(["a", "b", "c"], ["b", "c", "a"], [1, 2, -2])["compatible_on_observed_graph"])
        self.assertFalse(potential_constraints(["a"], ["a"], [1])["compatible_on_observed_graph"])
        with self.assertRaises(ValueError):
            potential_constraints(["a"], ["b"], [.1])

    def test_terminal_state_keeps_final_value(self):
        class Body:
            actions = 1
            initial_prior = np.ones(1)
            transport = np.zeros((1, 1, 1))
            def word_value(self, word):
                if any(a != 0 for a in word):
                    raise ValueError()
        planner = PotentialPlanner(Body(), [3], np.asarray([[[2.]]]))
        self.assertEqual(planner.value(np.asarray([1.]), [0]), 3)
        self.assertEqual(planner.value(np.asarray([1.]), [0, 0]), 3)

    def test_save_load_preserves_field_and_refuses_overwrite(self):
        p = np.column_stack((np.ones(8), np.arange(8)))
        q, actions = p + [0, 1], np.tile(np.arange(4), 2)
        body = EventReader("potential").fit(p, q, actions, np.ones(8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "body.npz"
            body.save(path, {"path": "frozen.npz", "sha256": "test"}, {})
            restored = EventReader.load(path)
            np.testing.assert_array_equal(body.predict(p, q, actions), restored.predict(p, q, actions))
            np.testing.assert_array_equal(body.state_value(p), restored.state_value(p))
            with self.assertRaises(RuntimeError):
                body.save(path, {}, {})


if __name__ == "__main__":
    unittest.main()

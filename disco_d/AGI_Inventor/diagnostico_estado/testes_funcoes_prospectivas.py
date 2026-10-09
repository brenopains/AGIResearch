"""Contratos do gerador prospectivo; não são resultados de inteligência."""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import unittest
import numpy as np
from funcoes_prospectivas import crossfit_future, propose_future


class FutureTests(unittest.TestCase):
    def test_future_difference_without_immediate_reward(self):
        source = np.asarray([[-2.], [-1.], [1.], [2.]])
        phi = np.ones((4, 1))
        successor_phi = np.asarray([[-1.], [-1.], [1.], [1.]])
        functions, report = propose_future(source, np.zeros(4, int), successor_phi, np.ones(4), [0, 1, 0, 1], phi)
        self.assertGreater(report["atoms"], 0)
        self.assertEqual(report["validation_targets_used"], 0)
        for origin in functions.origins:
            self.assertNotEqual(*origin["episodes"])
            self.assertEqual(origin["question"], "continuation_times_observed_successor_functions")

    def test_constant_future_is_not_invented_signal(self):
        functions, report = propose_future(np.arange(4)[:, None], np.zeros(4, int), np.ones((4, 1)),
            np.ones(4), [0, 1, 0, 1], np.ones((4, 1)))
        self.assertEqual(functions.expressions, [])
        self.assertEqual(report["status"], "no_varying_future_question")

    def test_raw_alias_rejects_separation_from_missing_information(self):
        functions, report = propose_future(np.zeros((4, 1)), np.zeros(4, int), np.arange(4)[:, None],
            np.ones(4), [0, 1, 0, 1], np.ones((4, 1)))
        self.assertEqual(functions.expressions, [])
        self.assertGreater(report["raw_aliases_encountered"], 0)

    def test_crossfit_does_not_fit_held_episode_target(self):
        phi = np.ones((4, 1))
        target = np.asarray([[1.], [2.], [3.], [4.]])
        actions, episodes = np.zeros(4, int), np.asarray([0, 0, 1, 1])
        before, _ = crossfit_future(phi, actions, target, episodes)
        target[:2] += 100
        after, _ = crossfit_future(phi, actions, target, episodes)
        np.testing.assert_array_equal(before[:2], after[:2])
        self.assertFalse(np.array_equal(before[2:], after[2:]))

    def test_action_without_other_episode_fails(self):
        with self.assertRaises(ValueError):
            crossfit_future(np.ones((2, 1)), np.asarray([0, 1]), np.ones((2, 1)), np.asarray([0, 1]))


if __name__ == "__main__":
    unittest.main()

"""Identidade em ordem não comutativa, padding e cancelamentos reais de erro."""

import unittest
import numpy as np
from decompor_erros_transporte import trajectory, summarize
import testes_transporte_funcoes as fixtures


class Attribution(unittest.TestCase):
    def test_telescope_with_arbitrary_true_states(self):
        body = fixtures.Contracts().model()
        phi = np.asarray([[1., .3], [1., .7], [1., .2], [1., .9]])
        for mode in ("transport", "no_context", "no_transport"):
            body.mode = mode
            result = trajectory(body, phi, np.asarray([0, 1, 0]), np.asarray([4., -1., 2.]), np.ones(4, bool))
            self.assertLess(result["identity_error"], 1e-12)
            self.assertAlmostEqual(result["predicted"] - result["actual"],
                                   result["reward_error"] + result["transport_error"] + result["boundary_error"])

    def test_terminal_before_suffix_still_attributes_predicted_hallucination(self):
        body = fixtures.Contracts().model()
        phi = np.asarray([[1., .3], [1., .7], [1., .7], [1., .7]])
        result = trajectory(body, phi, np.asarray([0, 1, 0]), np.asarray([4., 0., 0.]), np.asarray([True, False, False, False]))
        self.assertEqual(len(result["steps"]), 1)
        self.assertLess(result["identity_error"], 1e-12)

    def test_dead_root_has_separate_boundary_error(self):
        body = fixtures.Contracts().model()
        result = trajectory(body, np.asarray([[1., .3], [1., .3]]), np.asarray([0]), np.asarray([0.]), np.zeros(2, bool))
        self.assertEqual(result["predicted"], result["boundary_error"])
        self.assertEqual(result["reward_error"], result["transport_error"])
        self.assertEqual(result["steps"], [])

    def test_invalid_terminal_padding_is_rejected(self):
        body = fixtures.Contracts().model()
        with self.assertRaises(RuntimeError):
            trajectory(body, np.ones((3, 2)), np.asarray([0, 1]), np.asarray([0., 1.]), np.asarray([True, False, False]))

    def test_cross_terms_can_cancel_completely(self):
        row = {"actual_delta": 1., "predictions": {"composed": 1., "real_histories_learned_reward_oracle": 4.,
               "reward_error_removed_oracle": -2., "initial_history_real_termination_oracle": 4., "zero": 0.},
               "reward_error": 3., "transport_error": -3., "boundary_error": 0., "identity_error": 0.}
        result = summarize([row])
        self.assertEqual(sum(result["mse_decomposition"].values()), 0.)
        self.assertEqual(result["mse_decomposition"]["reward_transport_cross"], -18.)


if __name__ == "__main__":
    unittest.main()

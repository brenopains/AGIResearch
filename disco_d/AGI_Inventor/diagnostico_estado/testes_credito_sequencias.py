"""Linearidade útil, ordem de ações, rejeição de contradições e exportação."""

import unittest
import numpy as np
import testes_transporte_funcoes as fixtures
from credito_sequencias_transportadas import credits, fit_credit


class Credit(unittest.TestCase):
    def test_credit_reproduces_full_body_and_preserves_order(self):
        body = fixtures.Contracts().model()
        h = np.asarray([[0.], [.4], [.8], [.2]])
        words = [[0], [1, 0], [0, 1], [1, 0, 1]]
        features = credits(body, h, words)
        self.assertTrue(np.allclose(features @ body.reward.reshape(-1), body.predict(h, words)))
        self.assertFalse(np.allclose(credits(body, [[.4]], [[1, 0]]), credits(body, [[.4]], [[0, 1]])))
        for mode in ("no_context", "no_transport"):
            body.mode = mode
            self.assertTrue(np.allclose(credits(body, h, words) @ body.reward.reshape(-1), body.predict(h, words)))

    def test_fit_changes_only_linear_half(self):
        body = fixtures.Contracts().model()
        h = np.asarray([[0.], [.4], [.8], [.2]])
        left, right = [[0], [1, 0], [0, 1], [1, 0, 1]], [[1], [0, 1], [1, 0], [0, 1, 0]]
        x = credits(body, h, left) - credits(body, h, right)
        target = x @ np.asarray([1., 2., -1., .5])
        model, report, _, _ = fit_credit(body, h, left, right, target, [1, 2, 2, 3], alpha=1e-6)
        self.assertTrue(np.array_equal(model.transport, body.transport))
        self.assertTrue(np.array_equal(model.support, body.support))
        self.assertLess(np.mean((model.comparison(h, left, right) - target) ** 2), 1e-8)
        self.assertEqual(report["solved_parameters"], 4)

    def test_identical_words_cannot_get_nonzero_target(self):
        body = fixtures.Contracts().model()
        with self.assertRaises(ValueError):
            fit_credit(body, [[0.]], [[0]], [[0]], [1.], [1])


if __name__ == "__main__":
    unittest.main()

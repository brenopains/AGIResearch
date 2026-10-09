"""Testes de cobertura do pool; oráculo não é solução executável."""
import unittest
from diagnosticar_pool_funcoes import opportunity


class PoolTests(unittest.TestCase):
    def test_no_margin_is_not_failure_of_generator(self):
        report = opportunity([0, 0], [1, -1], [[2, 3], [-2, -3]])
        self.assertEqual(report["baseline_error_positions"], [])
        self.assertEqual(report["maximum_roots_improved_by_one_candidate"], 0)

    def test_complementary_candidates_require_oracle_for_per_case_choice(self):
        report = opportunity([1, 1], [-1, -1], [[1, -1], [-1, 1]])
        self.assertEqual(report["maximum_roots_improved_by_one_candidate"], 1)
        self.assertEqual(report["root_improved_by_any_candidate"], [0, 1])
        self.assertEqual(report["answer_conditioned_pool_oracle_fair_regret"], 0)
        self.assertTrue(report["oracle_is_not_an_executable_decision_rule"])

    def test_unresolved_root_and_fair_tie(self):
        report = opportunity([2, -2], [0, 1], [[1, 1]])
        self.assertEqual(report["baseline_fair_regret"], 1.5)
        self.assertEqual(report["root_not_improved_by_any_candidate"], [1])
        self.assertEqual(report["answer_conditioned_pool_oracle_fair_regret"], 1)

    def test_invalid_predictions_rejected(self):
        with self.assertRaises(RuntimeError):
            opportunity([1], [1], [[float("nan")]])


if __name__ == "__main__":
    unittest.main()

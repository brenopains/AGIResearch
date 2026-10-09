"""Contratos relevantes: sintaxe, identidade física, defeitos e controle trivial."""

import copy
import unittest
import numpy as np
from diagnosticar_semantica_prefixo import common_prefix, metrics, physical_case
from auditar_semantica_prefixo import scores, summary


class Contracts(unittest.TestCase):
    def bank(self):
        plans = np.zeros((1, 2, 16), dtype=int)
        plans[0, 1, 2:] = 1
        r = np.zeros((1, 2, 16))
        r[0, :, 0] = 7
        r[0, 0, 3], r[0, 1, 4] = 4, 1
        return {"actions": plans, "histories": np.zeros((1, 2, 17, 2, 1)),
                "valid": np.ones((1, 2, 17), dtype=bool), "rewards": r,
                "roots": np.asarray([8]), "episode": np.asarray([1])}

    def test_longest_prefix_and_cancel(self):
        self.assertEqual(common_prefix([0, 1, 2], [0, 1, 3]), 2)
        self.assertEqual(common_prefix([0], [1]), 0)
        self.assertEqual(common_prefix([0, 1], [0, 1]), 2)
        case = physical_case(self.bank(), 0)
        self.assertEqual(case["actual_delta"], 3)
        self.assertEqual(case["actual_suffix_delta"], 3)
        self.assertEqual(case["prefix_length"], 2)

    def test_prefix_mismatch_is_instrument_error(self):
        for field, index in (("histories", (0, 1, 1, 0, 0)), ("rewards", (0, 1, 0)), ("valid", (0, 1, 1))):
            bank = copy.deepcopy(self.bank())
            bank[field][index] = not bank[field][index] if field == "valid" else 2
            with self.assertRaises(RuntimeError):
                physical_case(bank, 0)

    def test_termination_padding_cannot_reward(self):
        bank = self.bank()
        bank["valid"][0, 0, 5:] = False
        bank["rewards"][0, 0, 7] = 1
        with self.assertRaises(RuntimeError):
            physical_case(bank, 0)

    def test_empty_suffix_rejected(self):
        bank = self.bank()
        bank["actions"][0, 1] = bank["actions"][0, 0]
        with self.assertRaises(RuntimeError):
            physical_case(bank, 0)

    def test_consistent_zero_is_not_capability(self):
        targets = [2., -6.]
        predicted = [0., 0.]
        result = metrics(targets, predicted)
        self.assertEqual(result, scores(targets, predicted))
        self.assertEqual(result["fair_tie_regret"], 2.)
        self.assertEqual(result["mse"], 20.)
        rows = [{"actual_delta": a, "predictions": {q: 0. for q in
                ("full_initial", "suffix_initial", "suffix_actual_midpoint_oracle")}} for a in targets]
        self.assertEqual(summary(rows)["maximum_absolute_consistency_defect"], 0.)
        self.assertEqual(summary(rows)["oracle_fair_tie_gain_over_full"], 0.)

    def test_wrong_sign_and_real_ties(self):
        targets, predicted = [2., -6., 0., 1.], [-1., 2., 8., 1.]
        self.assertEqual(metrics(targets, predicted), scores(targets, predicted))
        self.assertEqual(metrics(targets, predicted)["fair_tie_regret"], 2.)
        self.assertEqual(metrics(targets, predicted)["directional_accuracy_on_real_nonzero"], 1 / 3)


if __name__ == "__main__":
    unittest.main()

import unittest
from diagnosticar_contexto_resolucao_v2 import cost,summarize


def rows(actual,predicted):
    return [{"bank_row":i,"left":0,"right":1,"actual_delta":d,"predicted_delta":p} for i,(d,p) in enumerate(zip(actual,predicted))]


class ContextTests(unittest.TestCase):
    def test_fair_ties_and_directions(self):
        self.assertEqual(cost(-2,0),1)
        self.assertEqual(cost(-2,-1),0)
        self.assertEqual(cost(-2,1),2)
        self.assertEqual(cost(0,1),0)

    def test_context_can_beat_constant_oracle(self):
        value=summarize({"no_context":rows([1,-1],[1,1]),"learned":rows([1,-1],[1,-1])})
        self.assertEqual(value["retrospective_constant_per_pair_floor"],.5)
        result=value["arms"]["learned"]
        self.assertEqual(result["fair_tie_regret"],0)
        self.assertEqual(result["beneficial_changed_choices"],1)
        self.assertEqual(result["gain_over_retrospectively_best_constant_per_pair"],.5)

    def test_variable_context_can_harm(self):
        result=summarize({"no_context":rows([2,-1],[1,1]),"learned":rows([2,-1],[-1,1])})
        self.assertEqual(result["retrospective_constant_per_pair_floor"],.5)
        self.assertEqual(result["arms"]["learned"]["harmful_changed_choices"],1)
        self.assertEqual(result["arms"]["learned"]["gain_over_learned_body_without_context"],-1)

    def test_reference_is_primary_ablation_for_other_arms(self):
        value=summarize({"no_context":rows([1,-1],[1,1]),"random":rows([1,-1],[1,-1])})
        self.assertEqual(value["same_body_comparison_valid_only_for"],"learned")
        self.assertNotIn("gain_over_same_body_without_context",value["arms"]["random"])
        self.assertEqual(value["arms"]["random"]["gain_over_learned_body_without_context"],.5)

    def test_rejects_duplicated_or_contradictory_cases(self):
        with self.assertRaises(RuntimeError):
            summarize({"no_context":rows([1],[0])*2})
        with self.assertRaises(RuntimeError):
            summarize({"no_context":rows([1],[0]),"learned":rows([-1],[0])})


if __name__=="__main__":
    unittest.main()

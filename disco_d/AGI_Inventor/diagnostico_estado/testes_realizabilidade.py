import unittest
from diagnosticar_realizabilidade import np,bounds,rollout


class PositiveBoundsTests(unittest.TestCase):
    def test_valid_subprobability_and_absorption(self):
        self.assertFalse(bounds([.5,.1,.5],1.)["outside_necessary_bounds"])
        self.assertFalse(bounds([0.,0.,0.],.5)["outside_necessary_bounds"])

    def test_positive_features_cannot_exceed_survival(self):
        self.assertTrue(bounds([.4,.6,.1],1.)["outside_necessary_bounds"])
        self.assertTrue(bounds([.4,-.1,.1],1.)["outside_necessary_bounds"])

    def test_survival_cannot_revive_or_exceed_one(self):
        self.assertTrue(bounds([.8,.1],.5)["outside_necessary_bounds"])
        self.assertTrue(bounds([1.1,.1])["outside_necessary_bounds"])

    def test_valid_mass_is_not_forced_to_one(self):
        class Body:
            mode="transport"
            transport=np.array([[[.5,0.],[0.,.5]]])
        rows=rollout(Body(),np.array([1.,.8]),[0,0])
        self.assertEqual([r["mass"] for r in rows],[1.,.5,.25])
        self.assertFalse(any(r["outside_necessary_bounds"] for r in rows))

    def test_nonfinite_is_rejected(self):
        with self.assertRaises(RuntimeError):
            bounds([1.,float("nan")])


if __name__=="__main__":
    unittest.main()

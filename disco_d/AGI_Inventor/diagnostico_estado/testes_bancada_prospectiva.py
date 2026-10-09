"""Validação estrutural da bancada; nenhum teste afirma nível humano."""
import unittest
from piloto_prospectivo import np, coverage, transitions, view, witness_indices


class BankTests(unittest.TestCase):
    def bank(self):
        return {"histories": np.zeros((32, 8, 17, 16, 336), np.float32), "rewards": np.zeros((32, 8, 16)),
            "valid": np.ones((32, 8, 17), bool), "episode": np.repeat(np.arange(8), 4),
            "programs": np.arange(128).reshape(8, 16) % 4}

    def test_transition_alignment_and_dead_padding(self):
        bank = self.bank()
        bank["valid"][0, 0, 1:] = False
        bank["rewards"][0, 0, 0] = 7
        data = transitions(bank)
        self.assertEqual(len(data["reward"]), 32 * 8 * 16 - 15)
        self.assertEqual(data["reward"][0], 7)
        self.assertEqual(data["continuation"][0], 0)
        self.assertEqual(data["actions"][0], bank["programs"][0, 0])

    def test_witnesses_ignore_rewards_and_cover_groups(self):
        actions = np.tile(np.arange(4), 1024)
        episodes = np.repeat(np.arange(8), 512)
        ids = witness_indices(actions, episodes)
        self.assertEqual(len(ids), 512)
        for e in range(8):
            for a in range(4):
                self.assertEqual(np.sum((episodes[ids] == e) & (actions[ids] == a)), 16)

    def test_view_uses_repeated_known_or_new_programs(self):
        bank = self.bank()
        bank["rewards"][:, 4, 0] = 2
        bank["rewards"][:, 5, 0] = 1
        cases = view(bank, 4)
        self.assertEqual(len(cases["actual"]), 64)
        self.assertEqual(cases["actual"][0], 1)
        self.assertEqual(cases["actual"][1], 0)
        self.assertEqual(cases["bank_cases"][0], (0, 4, 5))

    def test_coverage_needs_both_preferences_in_multiple_episodes(self):
        episodes = np.repeat(np.arange(8), 8)
        self.assertFalse(coverage(np.zeros(64), episodes)["sufficient"])
        values = np.tile([1, -1, -1, 1], 16)
        self.assertTrue(coverage(values, episodes)["sufficient"])
        values[episodes > 0] = 0
        self.assertFalse(coverage(values, episodes)["sufficient"])


if __name__ == "__main__":
    unittest.main()

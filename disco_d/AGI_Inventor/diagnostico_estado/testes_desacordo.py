"""Fumaça da seleção por desacordo e do verificador ALE com rollback."""

from pathlib import Path

import numpy as np

import desacordo_ativo as D


class FakeModel:
    def __init__(self, votes):
        self.votes = votes

    def predict(self, _x):
        row = np.zeros((1, 4))
        row[0, self.votes] = 1
        return row


def test_votes():
    observation = {"r0_21x16_2": np.zeros(672),
                   "hist_21x16_16": np.zeros(5376)}
    unanimous = [("r0_21x16_2", FakeModel(2)) for _ in range(12)]
    split = [("r0_21x16_2", FakeModel(i % 2)) for i in range(12)]
    assert D.vote_score(unanimous, observation)[0] == 0
    assert D.vote_score(split, observation)[0] == 0.5


def test_ale():
    D.EPISODES, D.POOL, D.PROBES = 1, 4, 1
    output = Path(__file__).resolve().parent / "resultados" / "fumaca_desacordo.json"
    result = D.run("breakout", output, seed=17)
    assert result["complete_episodes"] == 1
    assert result["replay_checked"]
    assert result["total_steps"] <= 250000


if __name__ == "__main__":
    test_votes()
    test_ale()
    print("fumaça_desacordo_ok")

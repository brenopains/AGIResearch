"""Trocar de ação, término e concordância com o corpo já confirmado."""

import itertools

import numpy as np

from futuros_distribuidos import value_table, weights
from planejador_distribuido import ExperiencePlanner


def main():
    states = np.asarray([[0.], [1.]])
    edges = [{"source": states, "successor": np.ones((2, 1)), "reward": [1, 0], "continuation": [1, 1]},
             {"source": states, "successor": np.zeros((2, 1)), "reward": [0, 5], "continuation": [1, 1]}]
    planner = ExperiencePlanner(edges, neighbors=1)
    plans = list(itertools.product(range(2), repeat=2))
    values = planner.plan_values([[0.]], plans)[0]
    assert np.array_equal(values, [1, 6, 1, 0])
    search = planner.search([[0.]], horizon=2, width=4)
    assert search["plans"] == [[0, 1]] and search["values"] == [6.]
    assert search["expansions_per_root"] == [6]
    # Uma busca larga coincide com enumeração exata no mundo finito.
    plans = list(itertools.product(range(2), repeat=4))
    values = planner.plan_values([[0.], [1.]], plans)
    search = planner.search([[0.], [1.]], horizon=4, width=16)
    assert np.allclose(search["values"], values.max(1))
    for action in range(2):
        e = planner.edges[action]
        old = weights([[0.]], e["source"], 1) @ value_table(
            e["source"], e["successor"], e["reward"], e["continuation"], stages=4, neighbors=1)
        new = planner.plan_values([[0.]], [[action] * 4])
        assert np.allclose(new[:, 0], old[:, -1])
    terminal = [{"source": [[0]], "successor": [[0]], "reward": [3], "continuation": [0]}]
    assert ExperiencePlanner(terminal, 1).search([[0]], 5, 1)["values"] == [3.]
    for invalid in ([[2]], [[]]):
        try:
            planner.plan_values([[0]], invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("plano inválido não rejeitado")
    print("planejador_distribuido_ok: troca_de_acao, enumeracao_exata, corpo_confirmado, absorcao")


if __name__ == "__main__":
    main()

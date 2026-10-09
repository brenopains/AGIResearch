"""Controle positivo: conservar a distribuição evita estado médio fictício."""

import numpy as np

from futuros_distribuidos import mean_rollout, value_table, weights


def main():
    support = np.asarray([[-1.0], [1.0], [0.0]])
    probability = weights([[0]], support)
    assert np.all(probability >= 0)
    assert np.allclose(probability.sum(1), 1)
    # As duas realidades têm recompensa 1; seu ponto médio tem 0.
    # Um único vizinho torna os extremos absorventes: expectativa correta
    # é 2 ao compor dois passos, enquanto a leitura do meio é 0.
    r = np.asarray([1., 1., 0.])
    mixture = np.asarray([.5, .5, 0.])
    value = value_table(support, support, r, np.ones(3), stages=2, neighbors=1)
    assert float(mixture @ value[:, -1]) == 2
    mean = (mixture @ support)[None]
    collapsed_mixture = mean_rollout(mean, support, support, r, np.ones(3),
                                    stages=2, neighbors=1)
    assert collapsed_mixture[0, -1] == 0
    terminal = value_table([[0]], [[999]], [3.], [0.], stages=5)
    assert np.all(terminal == 3)
    persistent = value_table([[0]], [[0]], [3.], [1.], stages=5)
    assert np.all(persistent == np.arange(1, 6) * 3)
    collapsed = mean_rollout([[0]], np.asarray([[0.]]), np.asarray([[0.]]),
                             np.asarray([3.]), np.asarray([1.]), stages=5)
    assert np.array_equal(collapsed, persistent)
    print("futuros_distribuidos_ok: pesos, bellman, absorcao, soma, mistura")


if __name__ == "__main__":
    main()

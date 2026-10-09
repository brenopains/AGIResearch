"""Controle XOR: peças sem ganho isolado precisam poder formar candidata."""

import numpy as np

from conjuntos_sondas import select_bundle
from selecionar_sondas_decisao import preferences


def main():
    pattern = np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    train = np.tile(pattern, (4, 1))
    val = np.tile(pattern, (2, 1))
    def reward(x):
        odd = np.prod(x, axis=1) < 0
        return np.column_stack((~odd, odd)).astype(float)
    chosen, trace = select_bundle(np.zeros((16, 1)), np.zeros((8, 1)), train, val,
                                  preferences(reward(train)), reward(val),
                                  np.repeat([4, 5], 4), [(0,), (1,), (0, 1)])
    assert chosen == [0, 1]
    assert not trace[0]["eligible"] and not trace[1]["eligible"]
    assert trace[2]["eligible"]
    print("conjuntos_ok: xor_sem_ganho_isolado, composicao_admitida")


if __name__ == "__main__":
    main()

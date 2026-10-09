"""Ganho conjunto com dois episódios já perfeitos não deve ser impossível."""

import numpy as np

from avaliar_leituras_variaveis import choose, eligible
from selecionar_sondas_decisao import preferences


def main():
    assert eligible([0.5, 0.5, 0, 0])
    assert not eligible([0.5, 0, 0, 0])
    assert not eligible([1, 1, -0.5, 0])
    assert not eligible([0, 0, 0, 0])
    signal = np.asarray([-1, 1] * 4)
    reward = np.column_stack((signal < 0, signal > 0)).astype(float)
    test_signal = np.asarray([-1, 1, -1, 1, -1, -1, -1, -1])
    test_reward = np.column_stack((test_signal < 0, test_signal > 0)).astype(float)
    chosen, columns, trace = choose(np.zeros((8, 1)), np.zeros((8, 1)),
                                     signal[:, None], test_signal[:, None], preferences(reward),
                                     test_reward, np.repeat(np.arange(4), 2), [(0,)], channels=1)
    assert chosen == [0] and columns == [0]
    assert trace[0]["gain_each_episode"] == [0.5, 0.5, 0, 0]
    print("portao_leituras_ok: margem_parcial, melhora_repetida, nenhuma_piora")


if __name__ == "__main__":
    main()

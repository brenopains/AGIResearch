"""A seleção exige utilidade em ambos os episódios; separação não basta."""

import numpy as np

from selecionar_sondas_decisao import preferences, regrets, select


def main():
    signal = np.asarray([-1, 1] * 4)
    reward = np.column_stack((signal < 0, signal > 0)).astype(float)
    validation_signal = np.asarray([-1, 1, -1, 1])
    validation_reward = np.column_stack((validation_signal < 0, validation_signal > 0)).astype(float)
    base_fit, base_val = np.zeros((8, 2)), np.zeros((4, 2))
    extra_fit = np.column_stack((signal, np.ones(8)))
    extra_val = np.column_stack((validation_signal, np.ones(4)))
    groups = np.asarray([4, 4, 5, 5])
    selected, trace = select(base_fit, base_val, extra_fit, extra_val,
                              preferences(reward), validation_reward, groups)
    assert selected == [0]
    wrong = extra_val.copy()
    wrong[2:, 0] *= -1
    selected, _ = select(base_fit, base_val, extra_fit, wrong,
                        preferences(reward), validation_reward, groups)
    assert selected == []
    assert np.all(preferences(np.ones((3, 2))) == 0)
    assert np.all(regrets(validation_reward, validation_reward) == 0)
    print("sondas_decisao_ok: ganho_em_dois_episodios, rejeicao_de_ganho_local, empates")


if __name__ == "__main__":
    main()

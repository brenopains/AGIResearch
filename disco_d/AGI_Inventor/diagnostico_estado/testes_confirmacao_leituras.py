"""Controles do comparador pareado, incluindo um ganho raro inconclusivo."""

import numpy as np

from confirmar_leituras_variaveis import comparison


def main():
    episode = np.repeat(np.arange(8), 8)
    proposed = np.zeros(64)
    strong = comparison(proposed, np.ones(64), episode)
    assert strong["mean_positive"] and strong["lower_above_zero"]
    assert strong["positive_episodes"] == 8 and strong["regressed_episodes"] == 0
    tied = comparison(proposed, proposed, episode)
    assert not tied["lower_above_zero"] and not tied["mean_positive"]
    rare = np.zeros(64)
    rare[0] = 1
    scarce = comparison(proposed, rare, episode)
    assert scarce["mean_positive"] and not scarce["lower_above_zero"]
    negative = comparison(np.ones(64), proposed, episode)
    assert negative["regressed_episodes"] == 8 and not negative["mean_positive"]
    print("confirmacao_leituras_ok: pareamento, ganho_raro, empate, regressao")


if __name__ == "__main__":
    main()

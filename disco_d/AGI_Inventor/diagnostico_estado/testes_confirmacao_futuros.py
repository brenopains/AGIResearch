"""Placebos fazem parte do veredito, ganho raro não é confirmação."""

import numpy as np

from confirmar_futuros_distribuidos import CONTROLS, gate


def main():
    episode = np.repeat(np.arange(8), 8)
    errors = {name: np.ones(64) for name in CONTROLS}
    errors["distributed"] = np.zeros(64)
    assert gate(errors, episode)[0]
    errors["successor_shuffled"] = np.zeros(64)
    assert not gate(errors, episode)[0]
    errors["successor_shuffled"] = np.ones(64)
    rare = np.zeros(64)
    rare[0] = 1
    errors["constant"] = rare
    assert not gate(errors, episode)[0]
    errors["constant"] = np.ones(64)
    errors["direct_preferences"] = np.zeros(64)
    assert not gate(errors, episode)[0]
    print("confirmacao_futuros_ok: ganho, placebo_no_portao, ganho_raro, base_direta")


if __name__ == "__main__":
    main()

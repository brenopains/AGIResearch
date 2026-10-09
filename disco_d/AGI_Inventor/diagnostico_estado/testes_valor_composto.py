"""Crédito observado é aditivo; terminação impede recompensa posterior."""

import numpy as np

from valor_composto import expected_return, increments


def main():
    cumulative = np.asarray([[0, 1, 1, -1], [2, 2, 3, 4]], dtype=float)
    delta = increments(cumulative)
    assert np.array_equal(delta.cumsum(-1), cumulative)
    assert np.all(expected_return(delta, np.ones_like(delta)) == cumulative[:, -1])
    continuation = np.ones_like(delta)
    continuation[:, 1] = 0
    assert np.all(expected_return(delta, continuation) == cumulative[:, 1])
    assert np.all(expected_return(np.zeros_like(delta), continuation) == 0)
    print("valor_composto_ok: telescopio, soma, credito_da_ultima_acao, absorcao")


if __name__ == "__main__":
    main()

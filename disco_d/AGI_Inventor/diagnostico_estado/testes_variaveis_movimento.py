"""Controles mínimos da extração temporal a partir de pixels."""

import numpy as np

from variaveis_movimento import batch, describe


def moving(base, level):
    frames = np.full((16, 64, 64), base, np.uint8)
    for t in range(16):
        frames[t, 20:23, 4 + t:7 + t] = level
    return frames


if __name__ == "__main__":
    empty = np.zeros((16, 64, 64), np.uint8)
    assert describe(empty).shape == (120,)
    assert not np.any(describe(empty))
    bright = describe(moving(0, 255))
    dark = describe(moving(255, 0))
    assert bright[8] == 1 and bright[2] > 0
    assert dark[68] == 1 and dark[62] > 0
    assert np.array_equal(batch(np.stack((empty, moving(0, 255)))),
                          np.stack((describe(empty), bright)))
    print("movimento, polaridade e entrada vazia: OK")

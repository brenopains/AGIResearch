"""Contratos de posição variável em duas formas de observação, sem rótulos."""

import numpy as np

from leituras_variaveis import execute, make_readout


def world(shape, dy=0, dx=0):
    image = np.zeros(shape)
    image[2 + dy:4 + dy, 2 + dx:4 + dx] = [[1, 2], [0.5, 1]]
    image[6 + dy:8 + dy, 5 + dx:7 + dx] = -np.asarray([[1, 0.5], [2, 1]])
    return image


def main():
    for shape in ((12, 12), (21, 16)):
        original = world(shape)
        program = make_readout(original, 7)
        source = execute(program, original[None])
        moved = execute(program, world(shape, 2, 3)[None])
        assert np.allclose(source, moved, atol=1e-12)
        # Distâncias mudam quando só um fragmento se move.
        altered = original.copy()
        altered[6:8, 5:7] = 0
        altered[7:9, 5:7] = original[6:8, 5:7]
        result = execute(program, altered[None])
        assert not np.allclose(source[:, 3], result[:, 3])
        assert np.all(execute(program, np.zeros((1, *shape))) == 0)
        assert source.shape == (1, 4)
    assert make_readout(np.zeros((3, 4)), 7)["positive"] is None
    print("leituras_variaveis_ok: translacao, relacao_sensivel, ausencia, duas_geometrias")


if __name__ == "__main__":
    main()

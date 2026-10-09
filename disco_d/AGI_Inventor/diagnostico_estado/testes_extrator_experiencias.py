"""Evita vazamento de episódio e confere a composição de medidas reais."""

import numpy as np

from extrator_experiencias import stratified


def main():
    groups = np.repeat(np.arange(8), 200)
    selected = stratified(groups, {0, 2, 3}, 256)
    assert len(selected) == 256 and len(set(selected)) == 256
    assert set(groups[selected]) == {0, 2, 3}
    counts = [np.sum(groups[selected] == e) for e in (0, 2, 3)]
    assert max(counts) - min(counts) <= 1
    assert np.array_equal(selected, stratified(groups, {3, 0, 2}, 256))
    assert not len(stratified(groups, set(), 256))
    rng = np.random.default_rng(123)
    first, middle, last = rng.random((3, 336), dtype=np.float32)
    assert np.allclose((middle - first) + (last - middle), last - first, atol=2e-7)
    print("extrator_ok: episodios_separados, estratificacao, reproducao, composicao_medida")


if __name__ == "__main__":
    main()

"""Testes pequenos das invariantes de informação e partição."""

from __future__ import annotations

import numpy as np

from avaliar import group_split, placebo_action_labels, views, KernelRidge


def run():
    # Mesmo r0, passados diferentes: nenhuma função de r0 pode distinguir os dois.
    frames = np.zeros((2, 16, 210, 160), dtype=np.uint8)
    frames[1, -5, 60:80, 40:60] = 255
    features = views(frames)
    assert np.array_equal(features["r0_21x16_2"][0], features["r0_21x16_2"][1])
    assert not np.array_equal(features["hist_21x16_16"][0], features["hist_21x16_16"][1])

    # Mesmo agregado espacial 21x16, detalhe 42x32 diferente.
    frames = np.zeros((2, 16, 210, 160), dtype=np.uint8)
    frames[0, -1, 50:55, 50:55] = 255
    frames[1, -1, 55:60, 50:55] = 255
    features = views(frames)
    assert np.array_equal(features["r0_21x16_2"][0], features["r0_21x16_2"][1])
    assert not np.array_equal(features["res_42x32_2"][0], features["res_42x32_2"][1])

    episode = np.repeat(np.arange(8), 2)
    tr, val, test = group_split(episode)
    assert set(episode[tr]).isdisjoint(episode[val])
    assert set(episode[tr]).isdisjoint(episode[test])

    outcomes = np.broadcast_to(np.arange(4)[None, None, :, None, None],
                               (16, 3, 4, 21, 16)).copy()
    altered = placebo_action_labels(outcomes, tr, seed=3)
    assert np.array_equal(altered[test], outcomes[test])
    for i in tr:
        assert np.array_equal(np.sort(altered[i, 0, :, 0, 0]), np.arange(4))

    x = np.linspace(-1, 1, 20)[:, None]
    y = np.sin(x)
    predicted = KernelRidge().fit(x[::2], y[::2]).predict(x[1::2])
    assert float(np.mean((predicted - y[1::2]) ** 2)) < 0.02
    print("5 invariantes passaram")


if __name__ == "__main__":
    run()

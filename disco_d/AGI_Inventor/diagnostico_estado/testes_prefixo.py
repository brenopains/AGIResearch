"""Controle positivo da escolha de prefixo dependente de memória."""

import numpy as np

from avaliar import group_split, views
from prefixo import choose, evaluate


def run():
    n = 64
    episode = np.repeat(np.arange(8), 8)
    hidden = np.tile(np.array([-1.0, 1.0] * 4), 8)
    h = np.full((n, 16, 21, 16), 0.5, np.float32)
    h[:, -5, 6:9, 8:12] = 0.5 + hidden[:, None, None] * 0.4
    native = (np.repeat(np.repeat(h, 10, axis=2), 10, axis=3) * 255).astype(np.uint8)
    values = np.stack([hidden, -hidden, np.zeros(n), np.zeros(n)], axis=1)
    returns = np.repeat(values[:, :, None], 4, axis=2).reshape(n, 16)
    tr, val, test = group_split(episode)
    base = views(native)["r0_21x16_2"]
    selected = choose(base, h, values, episode, tr, val)
    outcome = evaluate({"histories": native, "returns": returns},
                       h, selected, tr, val, test)
    assert outcome["regret_gain"] > 0.4, outcome
    print("controle positivo de valor passou", outcome["program"], outcome["regret_gain"])


if __name__ == "__main__":
    run()

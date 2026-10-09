"""Controle positivo e controle de presente para o compilador."""

import numpy as np

from avaliar import group_split, views
from contraexemplos import assess, synthesize


def run():
    n = 64
    episode = np.repeat(np.arange(8), 8)
    hidden = np.tile(np.array([-1.0, 1.0] * 4), 8)
    history = np.full((n, 16, 21, 16), 0.5, np.float32)
    history[:, -5, 6:9, 8:12] = 0.5 + 0.4 * hidden[:, None, None]
    native = (np.repeat(np.repeat(history, 10, axis=2), 10, axis=3) * 255).astype(np.uint8)
    values = np.stack([hidden, -hidden, np.zeros(n), np.zeros(n)], axis=1)
    tr, val, test = group_split(episode)
    base = views(native)["r0_21x16_2"]
    chosen = synthesize(base, history, values, episode, tr, val, "history")
    outcome = assess(base, history, values, chosen, tr, val, test)
    present = synthesize(base, history, values, episode, tr, val, "current")
    present_outcome = assess(base, history, values, present, tr, val, test)
    assert outcome["train_counterexamples"] >= 4
    assert outcome["regret_gain"] > 0.4, outcome
    assert present_outcome.get("regret_gain", 0) < 0.1, present_outcome
    print("controle positivo passou; presente sozinho não recuperou a variável")


if __name__ == "__main__":
    run()

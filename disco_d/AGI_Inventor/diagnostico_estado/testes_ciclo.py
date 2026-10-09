"""Controle positivo: uma consequência depende de informação só presente no passado."""

import numpy as np

from avaliar import group_split, views
from ciclo import choose, evaluate_choice


def run():
    count = 64
    episode = np.repeat(np.arange(8), 8)
    hidden = np.tile(np.array([-1.0, 1.0] * 4), 8)
    history = np.full((count, 16, 21, 16), 0.5, np.float32)
    history[:, -5, 6:9, 8:12] = 0.5 + 0.4 * hidden[:, None, None]
    native = (np.repeat(np.repeat(history, 10, axis=2), 10, axis=3) * 255).astype(np.uint8)
    images = np.zeros((count, 16, 21, 16), np.float32)
    images[:, 5, 7, 9] = hidden
    returns = np.zeros((count, 16), np.float32)
    returns[:, 5] = hidden
    tr, val, test = group_split(episode)
    base = views(native)["r0_21x16_2"]
    chosen = choose(base, history, images, episode, tr, val)
    assert chosen is not None and chosen["question_id"] == 5, chosen
    result = evaluate_choice({"histories": native, "images": images,
                              "returns": returns}, chosen, history, tr, val, test)
    assert result["test_relative_gain"] > 0.5, result
    assert result["reward_informative_roots"] == len(test), result
    print("controle positivo passou", result["program"], result["test_relative_gain"])


if __name__ == "__main__":
    run()

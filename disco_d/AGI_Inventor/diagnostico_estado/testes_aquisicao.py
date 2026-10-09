"""Invariantes de replay e busca da aquisição de contraexemplos."""

import numpy as np

from aquisicao import frozen_model, make_ale, mutate, new_macro, prepare_episode, probe


def run():
    rng = np.random.default_rng(43)
    seen = set()
    first = new_macro(rng, seen)
    seen.add(first)
    second = mutate(rng, first, seen)
    assert first != second
    assert sum(a != b for a, b in zip(first, second)) == 1
    model, _digest = frozen_model("breakout")
    ale = make_ale("breakout", 43)
    actions = list(ale.getMinimalActionSet()[:4])
    initial, history = prepare_episode(ale, actions, rng, [32])
    one = probe(ale, actions, initial, history, first, model, [5000])
    two = probe(ale, actions, initial, history, first, model, [5000])
    assert one["terminal"] == two["terminal"]
    assert one["counterexample"] == two["counterexample"]
    assert one["steps"] == two["steps"]
    if not one["terminal"]:
        assert one["value_by_prefix"] == two["value_by_prefix"]
    print("macro, orçamento e replay passaram")


if __name__ == "__main__":
    run()

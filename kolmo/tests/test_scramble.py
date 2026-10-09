"""Contract tests for the structure-destroying controls.

These decide whether the cross-domain transfer result means anything, so
they have to actually destroy the input->output relation while changing
nothing else. A scrambler that leaves a learnable rule behind would make
the control library look artificially good and hide an optimization prior.
"""
import torch

from kolmo.grids import (NUM_COLORS, grid_to_pair, scramble_random_targets,
                         scramble_recolor_targets)


def _pairs(n: int = 3):
    out = []
    for i in range(n):
        inp = [[(r + c + i) % 4 for c in range(6)] for r in range(6)]
        outp = [[(v + 1) % 4 for v in row] for row in inp]  # a real, consistent rule
        out.append(grid_to_pair(inp, outp, device="cpu"))
    return out


def test_scramblers_preserve_inputs_shapes_and_masks():
    orig = _pairs()
    for scramble in (scramble_random_targets, scramble_recolor_targets):
        for a, b in zip(orig, scramble(orig, seed=0)):
            assert torch.equal(a.x, b.x), "the input must be untouched"
            assert torch.equal(a.mask, b.mask), "the geometry must be untouched"
            assert a.y_idx.shape == b.y_idx.shape


def test_scramblers_are_deterministic_in_the_seed():
    orig = _pairs()
    for scramble in (scramble_random_targets, scramble_recolor_targets):
        a, b = scramble(orig, seed=7), scramble(orig, seed=7)
        for x, y in zip(a, b):
            assert torch.equal(x.y_idx, y.y_idx)
        c = scramble(orig, seed=8)
        assert any(not torch.equal(x.y_idx, y.y_idx) for x, y in zip(a, c)), \
            "different seeds must give different scrambles"


def test_scramblers_actually_change_the_targets():
    orig = _pairs()
    for scramble in (scramble_random_targets, scramble_recolor_targets):
        changed = sum(1 for a, b in zip(orig, scramble(orig, seed=0))
                      if not torch.equal(a.y_idx, b.y_idx))
        assert changed == len(orig), f"{scramble.__name__} left targets untouched"


def test_recolor_uses_a_different_permutation_per_pair():
    """This is what removes the rule: one shared permutation would just be
    a relabeled version of the original task, still perfectly learnable."""
    scrambled = scramble_recolor_targets(_pairs(4), seed=0)
    maps = []
    for a, b in zip(_pairs(4), scrambled):
        m = {}
        for old, new in zip(a.y_idx[a.mask].tolist(), b.y_idx[b.mask].tolist()):
            m[old] = new
        maps.append(tuple(sorted(m.items())))
    assert len(set(maps)) > 1, "every pair got the same permutation -- the rule survives"


def test_recolor_is_a_bijection_within_each_pair():
    for a, b in zip(_pairs(), scramble_recolor_targets(_pairs(), seed=0)):
        pairs_seen = set(zip(a.y_idx.reshape(-1).tolist(), b.y_idx.reshape(-1).tolist()))
        srcs = [s for s, _ in pairs_seen]
        assert len(srcs) == len(set(srcs)), "a colour mapped to two different colours"


def test_random_targets_stay_in_range_and_are_masked():
    for b in scramble_random_targets(_pairs(), seed=0):
        assert int(b.y_idx.max()) < NUM_COLORS and int(b.y_idx.min()) >= 0
        assert int(b.y_idx[~b.mask].abs().sum()) == 0, "cells outside the grid must stay 0"

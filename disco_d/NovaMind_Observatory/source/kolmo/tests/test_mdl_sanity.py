"""Reference-compressor sanity check, per the plan's verification section:
a real compressor (gzip here, over the raw grid bytes) must be BEATEN by
the model on structured data, and must BEAT the model on random data.

Runs on CPU deliberately -- small nets, few steps, no reason to contend
with a GPU job that might be running a real experiment.
"""
from __future__ import annotations

import random
import zlib

import torch

from kolmo.fit import fit_task
from kolmo.grids import grid_to_pair

LN2_BITS_PER_BYTE = 8.0


def _gzip_bits(grid: list) -> float:
    flat = bytes(c for row in grid for c in row)
    compressed = zlib.compress(flat, level=9)
    return len(compressed) * LN2_BITS_PER_BYTE


def test_structured_task_beats_gzip():
    # Output = input, constant color 0, on a 10x10 grid: maximally repetitive.
    grid = [[0] * 10 for _ in range(10)]
    pair = grid_to_pair(grid, grid, device="cpu")

    res = fit_task([pair], hidden=16, depth=2, steps=300, lr=2e-2, sigma_w=1.0, seed=0, device="cpu")
    gzip_bits = _gzip_bits(grid)

    assert res.bits_total < gzip_bits, (
        f"model ({res.bits_total:.1f} bits) should beat gzip ({gzip_bits:.1f} bits) "
        "on a maximally repetitive grid"
    )


def _unrelated_pairs(n: int, rng: random.Random, size: int = 8):
    pairs, all_bytes = [], b""
    for _ in range(n):
        input_grid = [[rng.randrange(10) for _ in range(size)] for _ in range(size)]
        output_grid = [[rng.randrange(10) for _ in range(size)] for _ in range(size)]
        pairs.append(grid_to_pair(input_grid, output_grid, device="cpu"))
        all_bytes += bytes(c for row in output_grid for c in row)
    return pairs, all_bytes


def test_more_unrelated_data_costs_more_bits():
    # This replaces a stricter "model loses to gzip on random data" test
    # that turned out to be FALSE for this net/codec, and is kept failing
    # in git history rather than quietly deleted -- see the note below.
    #
    # What actually happens: on 4 mutually-unrelated 8x8 grids (a strong
    # single demo pair beats gzip trivially -- see test above), the model
    # reaches ~82 bits total against a raw/gzip floor around 900-1100 bits.
    # That is real, reproducible, and worth taking seriously rather than
    # tuning away: with a rich one-hot-per-cell input, each 3x3 receptive
    # field is close to a unique "address" even across unrelated grids (10
    # colors ^ 9 cells vastly exceeds the few hundred cells being fit), so
    # a small local conv net can memorize several unrelated grids near-
    # exactly at low L2 norm. Gaussian code length measures parameter
    # *norm*, not information content, and norm-efficient memorization of
    # random labels via a rich per-example "address" is a known, real
    # property of small nets (see Zhang et al., "Understanding Deep
    # Learning Requires Rethinking Generalization"), not a codec bug.
    #
    # Why this doesn't quietly invalidate Phase 0/1 (see plan doc, and the
    # commit that added this test): Phase 0's death criterion checks bits
    # measured from TRAINING demos against exact_match on a HELD-OUT test
    # grid the weights never saw. Address-style memorization of training
    # addresses does not transfer to a novel grid's addresses, so it can't
    # manufacture a false correlation there. Phase 1's ablation compares
    # real/shuffled/empty libraries on the SAME net architecture, so any
    # memorization floor is common to all three conditions and biases
    # AGAINST detecting a real library effect, not toward a false one.
    # RESOLVED (see tests/test_variational_codec.py): the gap was the code,
    # not the architecture. Measured on 4 unrelated 8x8 grids whose outputs
    # carry 850 bits of entropy: the Gaussian code claims 112 bits -- below
    # the floor, so it is not a code -- while the variational code, which
    # prices the precision each weight demands rather than its norm, lands
    # at 1009 bits (above the floor, below gzip's 1112) and still beats
    # gzip on structured data. The stricter test this comment describes as
    # FALSE is now true under codec=VARIATIONAL, and is asserted there.
    rng = random.Random(0)
    pairs1, _ = _unrelated_pairs(1, rng)
    pairs4, _ = _unrelated_pairs(4, rng)

    res1 = fit_task(pairs1, hidden=8, depth=1, steps=60, lr=2e-2, sigma_w=1.0, seed=0, device="cpu")
    res4 = fit_task(pairs4, hidden=8, depth=1, steps=60, lr=2e-2, sigma_w=1.0, seed=0, device="cpu")

    assert res4.bits_total > res1.bits_total, (
        f"4 mutually-unrelated grids ({res4.bits_total:.1f} bits) should cost more than "
        f"1 ({res1.bits_total:.1f} bits) -- if not, bits_total isn't tracking data volume at all"
    )

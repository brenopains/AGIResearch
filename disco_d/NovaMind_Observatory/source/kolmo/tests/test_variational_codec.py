"""The incompressibility floor: no valid code can describe random data in
fewer bits than its entropy.

This is the two-sided sanity check the plan's verification section asked
for and that test_mdl_sanity.py had to abandon -- its long comment records
a stricter "model must lose to gzip on random data" test that turned out
to be FALSE under the Gaussian weight code, and was kept failing in git
history rather than deleted. This file is the resolution: the Gaussian
code fails the floor because it measures parameter NORM, and a small conv
net can memorize unrelated grids at low norm; the variational code prices
the PRECISION each weight demands, and memorization demands precision
everywhere, so it lands above the floor where a real code must.

Consequence worth stating plainly: bits reported under GAUSSIAN in Phases
0-2 are a relative ranking, not code lengths. Those phases compare matched
conditions (same net, same prior, same task) where the bias is common to
every arm, so their comparisons stand -- but no absolute claim may be made
from them.

CPU only, small nets, same policy as test_mdl_sanity.py.
"""
from __future__ import annotations

import math
import random
import zlib

from kolmo.fit import GAUSSIAN, VARIATIONAL, fit_task
from kolmo.grids import grid_to_pair

SIZE, N_PAIRS, N_COLORS = 8, 4, 10
FIT_KW = dict(hidden=8, depth=1, steps=300, lr=2e-2, seed=0, device="cpu")


def _entropy_bits(n_grids: int = N_PAIRS) -> float:
    """Exact entropy of the output grids: uniform over 10 colors per cell."""
    return n_grids * SIZE * SIZE * math.log2(N_COLORS)


def _gzip_bits(grids: list) -> float:
    return len(zlib.compress(bytes(c for g in grids for row in g for c in row), 9)) * 8.0


def _random_pairs(seed: int = 0):
    rng = random.Random(seed)
    grid = lambda: [[rng.randrange(N_COLORS) for _ in range(SIZE)] for _ in range(SIZE)]
    ins = [grid() for _ in range(N_PAIRS)]
    outs = [grid() for _ in range(N_PAIRS)]
    return [grid_to_pair(i, o, device="cpu") for i, o in zip(ins, outs)], outs


def _structured_pairs():
    grids = [[[(r + c + i) % 3 for c in range(SIZE)] for r in range(SIZE)] for i in range(N_PAIRS)]
    return [grid_to_pair(g, g, device="cpu") for g in grids], grids


def test_variational_respects_the_incompressibility_floor():
    pairs, outs = _random_pairs()
    bits = fit_task(pairs, codec=VARIATIONAL, **FIT_KW).bits_total
    floor = _entropy_bits()
    assert bits > floor, (
        f"variational code claimed {bits:.0f} bits for data with {floor:.0f} bits of "
        "entropy -- that is not a code, it is a leak"
    )


def test_gaussian_violates_the_floor_which_is_why_variational_exists():
    """Characterization test: asserts the known defect, so it cannot be
    quietly 'fixed' or forgotten. If this ever fails, the Gaussian code
    changed and every phase that used it must be re-read."""
    pairs, _ = _random_pairs()
    bits = fit_task(pairs, codec=GAUSSIAN, **FIT_KW).bits_total
    assert bits < _entropy_bits(), (
        "the Gaussian weight code no longer under-prices memorized random data; "
        "update the docs in codec.py and test_mdl_sanity.py that describe it as such"
    )


def test_variational_still_beats_gzip_on_structured_data():
    """Respecting the floor is worthless if the codec also stops
    compressing real structure -- that would be a codec that just charges
    a lot for everything."""
    pairs, grids = _structured_pairs()
    bits = fit_task(pairs, codec=VARIATIONAL, **FIT_KW).bits_total
    gzip_bits = _gzip_bits(grids)
    assert bits < gzip_bits, (
        f"variational model ({bits:.0f} bits) should beat gzip ({gzip_bits:.0f} bits) "
        "on a highly regular grid"
    )


def test_variational_separates_structure_from_noise_by_a_wide_margin():
    struct, _ = _structured_pairs()
    rand, _ = _random_pairs()
    b_struct = fit_task(struct, codec=VARIATIONAL, **FIT_KW).bits_total
    b_rand = fit_task(rand, codec=VARIATIONAL, **FIT_KW).bits_total
    assert b_rand > 3 * b_struct, (
        f"noise ({b_rand:.0f} bits) must cost far more than structure "
        f"({b_struct:.0f} bits) under a code that prices precision"
    )

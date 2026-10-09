"""Tests for the Phase 1.5 library parametrization W = mu + B a + r.

The first test is the point of the whole change: a library built from
fits of (near-)identical tasks must make re-describing such a task nearly
free, and massively cheaper than the empty control -- that is what "the
mean is library content" buys. Runs on CPU with a tiny net, same policy
as test_mdl_sanity.py."""
import torch

from kolmo.fit import fit_task
from kolmo.grids import grid_to_pair
from kolmo.library import Library


def _trivial_task_pairs():
    # identity task on a structured grid: output = input
    grid = [[(r + c) % 3 for c in range(8)] for r in range(8)]
    return [grid_to_pair(grid, grid, device="cpu")]


def test_library_of_identical_fits_makes_task_nearly_free():
    pairs = _trivial_task_pairs()
    base = fit_task(pairs, hidden=8, depth=1, steps=200, lr=2e-2, seed=0, device="cpu")

    noise = torch.Generator().manual_seed(0)
    thetas = torch.stack([
        base.theta + 0.01 * torch.randn(base.theta.shape, generator=noise)
        for _ in range(4)
    ])
    lib = Library.from_weight_vectors(thetas, k=2)

    real = fit_task(pairs, hidden=8, depth=1, steps=200, lr=2e-2, seed=0,
                    device="cpu", library=lib)
    empty = fit_task(pairs, hidden=8, depth=1, steps=200, lr=2e-2, seed=0,
                     device="cpu", library=lib.empty())

    # warm start at mu ~= converged weights: weight bits should be tiny
    assert real.bits_weights < 0.1 * base.bits_weights, (
        f"re-describing a known task should be nearly free in weight bits: "
        f"got {real.bits_weights:.1f} vs {base.bits_weights:.1f} from scratch"
    )
    assert real.bits_total < 0.5 * empty.bits_total, (
        f"real library ({real.bits_total:.1f} bits) should beat empty "
        f"({empty.bits_total:.1f} bits) decisively on a task it was built from"
    )


def test_shuffled_preserves_values_and_destroys_structure():
    g = torch.Generator().manual_seed(0)
    thetas = torch.randn(5, 300, generator=g)
    lib = Library.from_weight_vectors(thetas, k=3)
    shuf = lib.shuffled(seed=0)

    assert torch.allclose(lib.mu.sort().values, shuf.mu.sort().values)
    for j in range(lib.B.shape[1]):
        assert torch.allclose(lib.B[:, j].sort().values, shuf.B[:, j].sort().values)
    assert not torch.allclose(lib.mu, shuf.mu), "shuffle must actually move entries"
    assert not torch.allclose(lib.B, shuf.B)
    assert shuf.sigma_a == lib.sigma_a and shuf.sigma_r == lib.sigma_r


def test_empty_zeroes_mu_and_basis():
    thetas = torch.randn(4, 100)
    lib = Library.from_weight_vectors(thetas, k=2)
    empty = lib.empty()
    assert empty.mu.abs().sum().item() == 0.0
    assert empty.B.abs().sum().item() == 0.0
    assert empty.mu.shape == lib.mu.shape and empty.B.shape == lib.B.shape

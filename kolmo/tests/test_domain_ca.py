"""Contract tests for the CA domain.

The point of a second domain is that it is genuinely different data, so
these check the generator is a real CA (Conway's Life reproduced exactly),
that tasks are deterministic and non-degenerate, and -- most important for
the transfer claim -- that the held-out pair is not derivable from the
demonstrations.
"""
import torch

from kolmo.domains.ca import load_tasks, make_task, step
from kolmo.grids import IN_CHANNELS, MAX_SIZE, NUM_COLORS


def test_step_reproduces_conways_life_blinker():
    """B3/S23 turns a vertical blinker into a horizontal one and back."""
    vertical = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    horizontal = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    assert step(vertical, birth={3}, survive={2, 3}) == horizontal
    assert step(horizontal, birth={3}, survive={2, 3}) == vertical


def test_step_reproduces_conways_life_block_still_life():
    block = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    assert step(block, birth={3}, survive={2, 3}) == block


def test_tasks_are_deterministic_in_the_seed():
    a = make_task(7, device="cpu")
    b = make_task(7, device="cpu")
    assert (a is None) == (b is None)
    if a is not None:
        for pa, pb in zip(a.train + a.test, b.train + b.test):
            assert torch.equal(pa.x, pb.x) and torch.equal(pa.y_idx, pb.y_idx)


def test_pairs_have_the_grid_contract_shapes():
    tasks = load_tasks(3, device="cpu")
    for task in tasks.values():
        for p in task.train + task.test:
            assert p.x.shape == (IN_CHANNELS, MAX_SIZE, MAX_SIZE)
            assert p.y_idx.shape == (MAX_SIZE, MAX_SIZE)
            assert p.mask.shape == (MAX_SIZE, MAX_SIZE)
            assert int(p.y_idx.max()) < NUM_COLORS


def test_no_pair_is_degenerate():
    """Every pair must change something and leave something alive --
    otherwise 'predict the output' is solved by copying or by blanking."""
    for task in load_tasks(6, device="cpu").values():
        for p in task.train + task.test:
            inp = p.x[:NUM_COLORS].argmax(dim=0)
            assert not torch.equal(inp[p.mask], p.y_idx[p.mask]), "output equals input"
            assert int((p.y_idx[p.mask] != 0).sum()) > 0, "output is empty"


def test_test_pair_is_not_a_demo():
    """The held-out pair must be new data, not a repeat of a demo -- the
    transfer measurement is worthless if the answer is already in the fit."""
    for task in load_tasks(6, device="cpu").values():
        for t in task.test:
            for d in task.train:
                assert not torch.equal(t.x, d.x), "test input duplicates a demo input"


def test_task_ids_are_stable_across_calls():
    assert list(load_tasks(5, device="cpu")) == list(load_tasks(5, device="cpu"))

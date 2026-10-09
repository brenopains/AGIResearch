"""Life-like cellular automata as a task domain.

One task = one rule. A rule is a pair (birth counts, survival counts)
over the 8-neighbourhood, exactly the B/S notation Conway's Life writes
as B3/S23. Demonstrations are pairs (state at t, state at t+1) started
from independent random configurations; the held-out test pair is another
such pair the fit never sees.

Why this domain, for the cross-domain transfer question:

- Its generative process shares nothing with ARC. ARC tasks are
  human-authored visual puzzles; these are deterministic local update
  rules with no author and no intent. If a library built only from ARC
  lowers the bits of these, that is not "puzzles resemble puzzles".
- It is grid-to-grid and shape-preserving by construction, so it needs no
  encoding trick to fit the fixed architecture. That removes the one
  design risk that could have forced a redesign (see domains/__init__).
- Difficulty is tunable and unbounded: the rule space is 2^9 x 2^9, and
  nothing in the fitted net has access to the rule.

Deliberately NOT done: no rule is filtered for being "interesting" by any
criterion that looks at how well a model does on it. Rules are rejected
only for being degenerate as DATA (a step that changes nothing, or that
empties the grid), which is a property of the rule alone.
"""
from __future__ import annotations

import random

import torch

from ..grids import MAX_SIZE, Task, grid_to_pair

NEIGHBOUR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def step(grid: list, birth: set, survive: set) -> list:
    """One synchronous CA update. Cells outside the grid count as dead."""
    h, w = len(grid), len(grid[0])
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            n = 0
            for dr, dc in NEIGHBOUR_OFFSETS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr][cc]:
                    n += 1
            alive = grid[r][c] != 0
            out[r][c] = 1 if ((alive and n in survive) or (not alive and n in birth)) else 0
    return out


def _random_state(h: int, w: int, density: float, rng: random.Random) -> list:
    return [[1 if rng.random() < density else 0 for _ in range(w)] for _ in range(h)]


def _colorize(grid: list, color: int) -> list:
    return [[color if v else 0 for v in row] for row in grid]


def _usable(before: list, after: list) -> bool:
    """Reject pairs that carry no signal: nothing changed, or all died."""
    live_after = sum(sum(row) for row in after)
    return before != after and live_after > 0


def make_task(seed: int, device: str, n_demos: int = 3, n_test: int = 1,
              max_size: int = MAX_SIZE) -> Task | None:
    """One CA task, or None if this rule/geometry produced no usable pairs.

    Returning None instead of resampling internally keeps the caller's
    seed-to-task mapping honest: a rejected seed stays rejected.
    """
    rng = random.Random(seed)
    birth = set(rng.sample(range(9), rng.randint(1, 3)))
    survive = set(rng.sample(range(9), rng.randint(1, 4)))
    h = rng.randint(8, max_size)
    w = rng.randint(8, max_size)
    color = rng.randint(1, 9)
    density = rng.uniform(0.2, 0.45)

    pairs = []
    for _ in range((n_demos + n_test) * 6):  # bounded attempts, no infinite loop
        before = _random_state(h, w, density, rng)
        after = step(before, birth, survive)
        if _usable(before, after):
            pairs.append(grid_to_pair(_colorize(before, color), _colorize(after, color),
                                      device, max_size))
        if len(pairs) == n_demos + n_test:
            return Task(train=pairs[:n_demos], test=pairs[n_demos:])
    return None


def load_tasks(n: int, device: str, seed0: int = 0, max_size: int = MAX_SIZE) -> dict:
    """{task_id: Task} for n usable CA tasks, ids stable given seed0."""
    out, seed = {}, seed0
    while len(out) < n:
        task = make_task(seed, device, max_size=max_size)
        if task is not None:
            out[f"ca{seed:04d}"] = task
        seed += 1
    return out

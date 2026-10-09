"""ARC grid <-> tensor conversion and task loading.

Scope, disclosed: only tasks where every (input, output) pair has matching
shape are used. That is a real restriction (not every ARC task preserves
shape), kept because it lets a single fixed-size conv net do cell-to-cell
prediction without a separate shape-prediction head. It removes ~35% of
ARC-AGI-1's training tasks; lifting it is future work, not needed to test
the falsifiable claim this codebase exists to test (see plan doc).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch

NUM_COLORS = 10
MAX_SIZE = 15
# Channel layout for the network's input tensor: 10 color channels + 1
# "outside the grid" channel, so padding is distinguishable from color 0.
IN_CHANNELS = NUM_COLORS + 1


@dataclass
class GridPair:
    x: torch.Tensor      # (IN_CHANNELS, MAX_SIZE, MAX_SIZE) float
    y_idx: torch.Tensor  # (MAX_SIZE, MAX_SIZE) long, color index per cell
    mask: torch.Tensor   # (MAX_SIZE, MAX_SIZE) bool, True where a real cell exists


@dataclass
class Task:
    train: list  # list[GridPair]
    test: list   # list[GridPair]


def _shape_ok(task_json: dict, max_size: int) -> bool:
    pairs = task_json["train"] + task_json["test"]
    for p in pairs:
        if len(p["input"]) != len(p["output"]) or len(p["input"][0]) != len(p["output"][0]):
            return False
    mx = max(
        max(len(p["input"]), len(p["input"][0]), len(p["output"]), len(p["output"][0]))
        for p in pairs
    )
    return mx <= max_size


def grid_to_pair(input_grid: list, output_grid: list, device: str, max_size: int = MAX_SIZE) -> GridPair:
    h, w = len(input_grid), len(input_grid[0])
    x = torch.zeros(IN_CHANNELS, max_size, max_size, device=device)
    x[NUM_COLORS, :, :] = 1.0  # "outside" channel defaults to 1 everywhere
    y_idx = torch.zeros(max_size, max_size, dtype=torch.long, device=device)
    mask = torch.zeros(max_size, max_size, dtype=torch.bool, device=device)
    for r in range(h):
        for c in range(w):
            color = input_grid[r][c]
            x[color, r, c] = 1.0
            x[NUM_COLORS, r, c] = 0.0  # inside the real grid
            y_idx[r, c] = output_grid[r][c]
            mask[r, c] = True
    return GridPair(x=x, y_idx=y_idx, mask=mask)


def d4_pairs(pairs: list) -> list:
    """All 8 D4 transforms (4 rotations x optional horizontal flip) of each
    pair, identity included once. Applying the same element to x, y and mask
    is only guaranteed-valid supervision when the task's true transformation
    commutes with that element -- which is why callers must treat this as a
    *hypothesis priced in bits* (phase0 --variants), never a default."""
    out = []
    for p in pairs:
        for flip in (False, True):
            x = torch.flip(p.x, dims=[2]) if flip else p.x
            y = torch.flip(p.y_idx, dims=[1]) if flip else p.y_idx
            m = torch.flip(p.mask, dims=[1]) if flip else p.mask
            for k in range(4):
                out.append(GridPair(
                    x=torch.rot90(x, k, dims=(1, 2)),
                    y_idx=torch.rot90(y, k, dims=(0, 1)),
                    mask=torch.rot90(m, k, dims=(0, 1)),
                ))
    return out


def color_canon_map(pairs: list) -> torch.Tensor:
    """(NUM_COLORS,) long, old color -> canonical index, by descending
    frequency over the given pairs' inputs and outputs; ties and unseen
    colors ordered by color index (a bijection always)."""
    freq = torch.zeros(NUM_COLORS)
    for p in pairs:
        freq += p.x[:NUM_COLORS].sum(dim=(1, 2)).cpu()
        freq += torch.bincount(p.y_idx[p.mask].reshape(-1).cpu(), minlength=NUM_COLORS).float()
    order = sorted(range(NUM_COLORS), key=lambda c: (-freq[c].item(), c))  # canonical -> old
    mapping = torch.empty(NUM_COLORS, dtype=torch.long)
    for new, old in enumerate(order):
        mapping[old] = new
    return mapping


def apply_color_map(pair: GridPair, mapping: torch.Tensor) -> GridPair:
    """Relabel colors of a pair by `mapping` (old -> new). The 'outside'
    channel is untouched. Bijective, so exact match in canonical space is
    equivalent to exact match in the original space."""
    inv = torch.argsort(mapping).to(pair.x.device)  # new -> old
    x = pair.x.clone()
    x[:NUM_COLORS] = pair.x[:NUM_COLORS][inv]
    y = mapping.to(pair.y_idx.device)[pair.y_idx]
    return GridPair(x=x, y_idx=y, mask=pair.mask)


def scramble_random_targets(pairs: list, seed: int) -> list:
    """Replace every output with uniform random colors, keeping shape/mask.

    The strongest form of "no learnable structure": there is no function
    from these inputs to these outputs, so a fit can only memorize. Used
    to ask whether cross-domain transfer needs the source tasks to have
    been LEARNABLE, or merely to have been OPTIMIZED to convergence.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for i, p in enumerate(pairs):
        rand = torch.randint(NUM_COLORS, p.y_idx.shape, generator=g).to(p.y_idx.device)
        out.append(GridPair(x=p.x, y_idx=torch.where(p.mask, rand, torch.zeros_like(rand)),
                            mask=p.mask))
    return out


def scramble_recolor_targets(pairs: list, seed: int) -> list:
    """Recolor each output by a DIFFERENT random color permutation.

    Milder than random targets and, for that reason, the more informative
    of the two: outputs keep the shape statistics and the spatial layout
    of real ARC outputs, but because the permutation differs per pair
    there is no consistent input->output rule left to learn. Bracketing
    the question with both controls avoids blaming a result on the
    pathology of fitting pure noise.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for p in pairs:
        perm = torch.randperm(NUM_COLORS, generator=g).to(p.y_idx.device)
        out.append(GridPair(x=p.x, y_idx=perm[p.y_idx], mask=p.mask))
    return out


def load_tasks(split: str, root: str, device: str, max_size: int = MAX_SIZE, limit: int | None = None) -> dict:
    """Load shape-preserving ARC tasks from `root`/data/{split}/*.json.

    Returns {task_id: Task}. `root` is the legacy novamind ARC data dir,
    e.g. legacy/novamind/data/ARC.
    """
    d = os.path.join(root, "data", split)
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    out = {}
    for f in files:
        task_json = json.load(open(f, encoding="utf-8"))
        if not _shape_ok(task_json, max_size):
            continue
        task_id = os.path.splitext(os.path.basename(f))[0]
        train = [grid_to_pair(p["input"], p["output"], device, max_size) for p in task_json["train"]]
        test = [grid_to_pair(p["input"], p["output"], device, max_size) for p in task_json["test"]]
        out[task_id] = Task(train=train, test=test)
        if limit is not None and len(out) >= limit:
            break
    return out

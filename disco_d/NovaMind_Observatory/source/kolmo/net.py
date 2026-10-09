"""Per-task grid-to-grid network.

Deliberately plain: a small conv resnet, translation-local by construction
(that's what convolution gives for free -- no extra machinery needed since
input/output grids are cell-aligned by construction, see grids.py's shape
restriction). No D4 augmentation, no explicit color-permutation equivariance.

That is a disclosed simplification, not an oversight. D4 augmentation is
only valid to add per-pair when the task's true transformation commutes
with the augmenting symmetry, which is not true for every ARC task (e.g.
"rotate the grid" tasks involve reflections that don't commute with all
D4 elements) -- doing it naively risks silently corrupting supervision for
a subset of tasks. Adding unverified "clever" structure like that is
exactly the failure pattern this codebase exists to avoid (see the
audit in the plan doc, section on Clifford bivectors that are written and
never read).

The verified equivariant architecture that note called future work now
exists here as `symmetric=True` (see D4Conv2d): it is opt-in, never a
default, and it is verified rather than asserted --
tests/test_equivariance.py checks net(g.x) == g.net(x) for all eight
elements of D4 before anything is allowed to use it. Crucially it is not
augmentation: it constrains kernels to the invariant subspace, which
REMOVES parameters, so the symmetry hypothesis can finally win an MDL
selection instead of being charged for (see experiments/phase0.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .grids import IN_CHANNELS, NUM_COLORS


def _d4_orbit_basis() -> torch.Tensor:
    """(3, 3, 3): the three D4-invariant patterns a 3x3 kernel decomposes into.

    Under the dihedral group of the square, the nine cells of a 3x3 kernel
    fall into exactly three orbits -- center, edge midpoints, corners. A
    kernel is D4-invariant iff it is constant on each orbit, so the
    invariant subspace is 3-dimensional, not 9. That is the whole point:
    the symmetry hypothesis becomes a REAL parameter saving (3 numbers per
    channel pair instead of 9), which the weight code charges less for.

    This is the ledger column the Phase 0 --variants run put the hypothesis
    in wrongly. There, D4 entered as demo augmentation, which fits 8x the
    pairs and therefore costs MORE weight bits -- it lost the MDL selection
    on every task, correctly, while still being the only variant that
    solved anything. Here the same hypothesis is charged the way MDL can
    actually reward: fewer parameters to transmit.
    """
    basis = torch.zeros(3, 3, 3)
    basis[0, 1, 1] = 1.0                                      # center
    basis[1, 0, 1] = basis[1, 1, 0] = basis[1, 1, 2] = basis[1, 2, 1] = 1.0  # edges
    basis[2, 0, 0] = basis[2, 0, 2] = basis[2, 2, 0] = basis[2, 2, 2] = 1.0  # corners
    return basis


class D4Conv2d(nn.Module):
    """3x3 conv whose kernel is constrained to the D4-invariant subspace.

    Stores 3 coefficients per (out, in) pair and expands them through the
    orbit basis at forward time, so the parameter vector fit.py transmits
    is genuinely 3/9 the size of the unconstrained kernel's.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.coef = nn.Parameter(torch.empty(out_ch, in_ch, 3))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        # match the fan-in scale of an unconstrained 3x3 kernel: each
        # coefficient is shared by up to 4 cells, so divide by the orbit sizes
        bound = (1.0 / (in_ch * 9)) ** 0.5
        with torch.no_grad():
            self.coef.uniform_(-bound, bound)
        self.register_buffer("basis", _d4_orbit_basis(), persistent=False)

    def forward(self, x):
        weight = torch.einsum("oik,khw->oihw", self.coef, self.basis)
        return torch.nn.functional.conv2d(x, weight, self.bias, padding=1)


def _conv3x3(in_ch: int, out_ch: int, symmetric: bool) -> nn.Module:
    return D4Conv2d(in_ch, out_ch) if symmetric else nn.Conv2d(in_ch, out_ch, 3, padding=1)


class ResBlock(nn.Module):
    def __init__(self, ch: int, symmetric: bool = False):
        super().__init__()
        self.conv1 = _conv3x3(ch, ch, symmetric)
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = _conv3x3(ch, ch, symmetric)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):
        h = torch.nn.functional.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return torch.nn.functional.silu(x + h)


class TinyGridNet(nn.Module):
    """input: (IN_CHANNELS, H, W) one-hot+outside grid -> output: (NUM_COLORS, H, W) logits."""

    def __init__(self, hidden: int = 32, depth: int = 3, symmetric: bool = False):
        super().__init__()
        self.stem = _conv3x3(IN_CHANNELS, hidden, symmetric)
        self.blocks = nn.ModuleList([ResBlock(hidden, symmetric) for _ in range(depth)])
        self.head = nn.Conv2d(hidden, NUM_COLORS, 1)  # 1x1: already D4-equivariant

    def forward(self, x):
        # x: (IN_CHANNELS, H, W) or batched (N, IN_CHANNELS, H, W)
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        out = self.head(h)  # (N, NUM_COLORS, H, W)
        return out.squeeze(0) if single else out


def param_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())

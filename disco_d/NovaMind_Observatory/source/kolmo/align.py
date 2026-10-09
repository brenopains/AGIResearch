"""Neuron-permutation alignment (weight matching) for TinyGridNet fits.

Why: PCA over the weight vectors of independently fitted nets is nearly
blind to shared structure, because the same function appears at permuted
points of weight space (hidden channels are interchangeable). Aligning
every fit to a common reference before the SVD concentrates shared
structure into few components -- without it, the "real" basis degenerates
toward generic extra capacity, which is exactly what the shuffled control
also provides (the observed Phase 1 tie). This is the weight-matching
variant of "git re-basin" (Ainsworth et al.), specialized to TinyGridNet.

Only function-preserving permutations are used, and TinyGridNet's
GroupNorm makes that non-trivial: channels are normalized within groups,
so an arbitrary channel permutation changes the normalization statistics.
The permutations that commute with GroupNorm are exactly those that map
groups onto groups (whole-group swaps composed with within-group
shuffles). The assignment is therefore solved hierarchically: a Hungarian
assignment within every (ref group, cand group) pair, then a group-level
Hungarian over the resulting scores.

TinyGridNet admits two independent permutation families:
- one stream permutation P shared by the stem output, every block's
  conv1 input, conv2 output, norm2, and the head input -- forced to be a
  single permutation by the residual additions ``x + h``;
- one free intermediate permutation Q_i per block (conv1 output / norm1 /
  conv2 input).

The contract for this module is
test_align.py::test_permutation_preserves_function: a permuted net must
produce bit-identical output on the same input. If that fails, nothing
returned here may be used.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment

from .fit import _split_theta


@dataclass
class NetPerm:
    """Index tensors with the convention aligned[i] = original[perm[i]]."""
    stream: torch.Tensor        # (H,) long
    inter: list                 # depth x (H,) long


def _layout(names: list, shapes: list) -> tuple:
    hidden = shapes[names.index("stem.weight")][0]
    depth = sum(1 for n in names if n.endswith("conv1.weight"))
    num_groups = min(8, hidden)  # must mirror TinyGridNet's GroupNorm(min(8, ch), ch)
    return hidden, depth, num_groups


def apply_perm(pd: dict, perm: NetPerm) -> dict:
    """Apply a function-preserving permutation to a TinyGridNet param dict."""
    P = perm.stream
    out = {
        "stem.weight": pd["stem.weight"][P],
        "stem.bias": pd["stem.bias"][P],
        "head.weight": pd["head.weight"][:, P],
        "head.bias": pd["head.bias"],
    }
    for i, Q in enumerate(perm.inter):
        pre = f"blocks.{i}."
        out[pre + "conv1.weight"] = pd[pre + "conv1.weight"][Q][:, P]
        out[pre + "conv1.bias"] = pd[pre + "conv1.bias"][Q]
        out[pre + "norm1.weight"] = pd[pre + "norm1.weight"][Q]
        out[pre + "norm1.bias"] = pd[pre + "norm1.bias"][Q]
        out[pre + "conv2.weight"] = pd[pre + "conv2.weight"][P][:, Q]
        out[pre + "conv2.bias"] = pd[pre + "conv2.bias"][P]
        out[pre + "norm2.weight"] = pd[pre + "norm2.weight"][P]
        out[pre + "norm2.bias"] = pd[pre + "norm2.bias"][P]
    return out


def _group_assign(S: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Max-score assignment restricted to GroupNorm-compatible permutations.

    S[i, j] = score of matching ref channel i to candidate channel j.
    Returns perm with aligned[i] = candidate[perm[i]].
    """
    h = S.shape[0]
    gs = h // num_groups
    S = S.detach().cpu()
    group_scores = torch.zeros(num_groups, num_groups)
    within = {}
    for gr in range(num_groups):
        for gc in range(num_groups):
            sub = S[gr * gs:(gr + 1) * gs, gc * gs:(gc + 1) * gs]
            ri, ci = linear_sum_assignment(-sub.numpy())
            group_scores[gr, gc] = float(sub[ri, ci].sum())
            within[(gr, gc)] = ci
    grs, gcs = linear_sum_assignment(-group_scores.numpy())
    perm = torch.empty(h, dtype=torch.long)
    for gr, gc in zip(grs, gcs):
        perm[gr * gs:(gr + 1) * gs] = torch.as_tensor(within[(gr, gc)], dtype=torch.long) + gc * gs
    return perm


def _cat_feats(pairs: list) -> torch.Tensor:
    """pairs: list of (ref_feat, cand_feat), each (H, ...). Returns S = R @ C.T."""
    r = torch.cat([a.reshape(a.shape[0], -1) for a, _ in pairs], dim=1)
    c = torch.cat([b.reshape(b.shape[0], -1) for _, b in pairs], dim=1)
    return r @ c.T


def _stream_similarity(ref: dict, cand: dict, inter: list, depth: int) -> torch.Tensor:
    pairs = [
        (ref["stem.weight"], cand["stem.weight"]),
        (ref["stem.bias"].unsqueeze(1), cand["stem.bias"].unsqueeze(1)),
        (ref["head.weight"].transpose(0, 1), cand["head.weight"].transpose(0, 1)),
    ]
    for i in range(depth):
        pre = f"blocks.{i}."
        Q = inter[i]
        # conv1: rows = intermediate (already matched via Q), cols = stream
        pairs.append((ref[pre + "conv1.weight"].transpose(0, 1),
                      cand[pre + "conv1.weight"][Q].transpose(0, 1)))
        # conv2: rows = stream, cols = intermediate
        pairs.append((ref[pre + "conv2.weight"], cand[pre + "conv2.weight"][:, Q]))
        pairs.append((ref[pre + "conv2.bias"].unsqueeze(1), cand[pre + "conv2.bias"].unsqueeze(1)))
        pairs.append((ref[pre + "norm2.weight"].unsqueeze(1), cand[pre + "norm2.weight"].unsqueeze(1)))
        pairs.append((ref[pre + "norm2.bias"].unsqueeze(1), cand[pre + "norm2.bias"].unsqueeze(1)))
    return _cat_feats(pairs)


def _inter_similarity(ref: dict, cand: dict, i: int, stream: torch.Tensor) -> torch.Tensor:
    pre = f"blocks.{i}."
    pairs = [
        # conv1: rows = intermediate, cols = stream (already matched via P)
        (ref[pre + "conv1.weight"], cand[pre + "conv1.weight"][:, stream]),
        (ref[pre + "conv1.bias"].unsqueeze(1), cand[pre + "conv1.bias"].unsqueeze(1)),
        (ref[pre + "norm1.weight"].unsqueeze(1), cand[pre + "norm1.weight"].unsqueeze(1)),
        (ref[pre + "norm1.bias"].unsqueeze(1), cand[pre + "norm1.bias"].unsqueeze(1)),
        # conv2: rows = stream (candidate matched to ref order via P), cols = intermediate
        (ref[pre + "conv2.weight"].transpose(0, 1),
         cand[pre + "conv2.weight"][stream].transpose(0, 1)),
    ]
    return _cat_feats(pairs)


def find_perm(theta_ref: torch.Tensor, theta: torch.Tensor, names: list, shapes: list,
              iters: int = 3) -> NetPerm:
    """Coordinate descent over (P, Q_0..Q_d) maximizing <theta_ref, perm(theta)>."""
    hidden, depth, num_groups = _layout(names, shapes)
    ref = _split_theta(theta_ref, names, shapes)
    cand = _split_theta(theta, names, shapes)
    perm = NetPerm(
        stream=torch.arange(hidden),
        inter=[torch.arange(hidden) for _ in range(depth)],
    )
    for _ in range(iters):
        perm.stream = _group_assign(
            _stream_similarity(ref, cand, perm.inter, depth), num_groups)
        for i in range(depth):
            perm.inter[i] = _group_assign(
                _inter_similarity(ref, cand, i, perm.stream), num_groups)
    return perm


def permute_theta(theta: torch.Tensor, names: list, shapes: list, perm: NetPerm) -> torch.Tensor:
    pd = apply_perm(_split_theta(theta, names, shapes), perm)
    return torch.cat([pd[n].reshape(-1) for n in names])


def align_thetas(thetas: torch.Tensor, names: list, shapes: list, iters: int = 3) -> torch.Tensor:
    """Align every row of thetas (N, P) to thetas[0]. Row 0 is returned as-is."""
    out = [thetas[0]]
    for i in range(1, thetas.shape[0]):
        perm = find_perm(thetas[0], thetas[i], names, shapes, iters=iters)
        out.append(permute_theta(thetas[i], names, shapes, perm))
    return torch.stack(out)

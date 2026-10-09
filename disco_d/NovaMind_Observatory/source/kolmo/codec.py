"""Bits accounting. This IS the loss -- nothing else may enter fit.py's loss.

Two terms, both closed-form and auditable:

- bits_crossentropy: cost of describing the known (demonstration) output
  grids given the network's predictions. Standard cross-entropy converted
  from nats to bits.
- bits_gaussian: two-part MDL code length of a real-valued parameter
  vector under a fixed zero-mean Gaussian reference distribution N(0, sigma^2).
  This is the closed form for "how many bits to transmit theta if both
  sender and receiver share the prior N(0, sigma^2)" -- algebraically it is
  L2 weight decay with a specific, non-arbitrary coefficient (1 / (2 sigma^2 ln 2)),
  plus a constant entropy term that does not depend on theta. CompressARC
  calls the same quantity "KL for weights (approximating L2 regularization)";
  this is the two-part-code version of that idea (no reparameterization /
  noise sampling), which is simpler and sufficient for what phases 0-1 test.
"""
from __future__ import annotations

import math

import torch

LN2 = math.log(2.0)


def bits_crossentropy(logits: torch.Tensor, target_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits: (C, H, W) or (N, C, H, W). target_idx: (H, W) or (N, H, W) long.
    mask: same shape as target_idx, bool. Batched and unbatched forms compute
    the identical sum -- batching exists only so fit.py can do one forward
    pass per step instead of one per pair."""
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
        target_idx = target_idx.unsqueeze(0)
        mask = mask.unsqueeze(0)
    log_probs = torch.log_softmax(logits, dim=1)  # (N, C, H, W), natural log
    picked = log_probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)  # (N, H, W), nats
    nats = -(picked * mask).sum()
    return nats / LN2


def bits_gaussian(params: torch.Tensor, sigma: float) -> torch.Tensor:
    """L2-equivalent weight code length, in bits: ||params||^2 / (2 sigma^2 ln2).

    This is the *quadratic* term of -log2 N(params; 0, sigma^2) only. The
    full two-part MDL codelength also has an additive differential-entropy
    term 0.5*n*log2(2*pi*sigma^2) -- deliberately dropped here, for two
    reasons: (1) differential entropy of a continuous density isn't itself
    a valid bit count without pairing it with a stated quantization step,
    which nothing in this codebase specifies; (2) empirically, for the
    parameter counts used here (n ~ 1e4-1e5), that term is a five-figure
    constant identical across every condition being compared (same n, same
    sigma), so including it drowns the actual signal in the reported
    number without changing any gradient (it doesn't depend on params) or
    any ranking between conditions. Bits reported by this codebase are
    therefore only ever meaningful as *relative* comparisons at matched n
    and sigma (exactly how they're used: real vs. shuffled vs. empty
    library, same net, same seed) -- not as an absolute code length.
    """
    return (params.pow(2).sum()) / (2.0 * sigma * sigma * LN2)


def bits_kl_gaussian(mean: torch.Tensor, log_sigma_q: torch.Tensor, sigma_p: float) -> torch.Tensor:
    """KL(N(mean, sigma_q^2) || N(0, sigma_p^2)) in bits, diagonal covariance.

    Why this exists: bits_gaussian above measures the *norm* of a parameter
    vector, not its information content. A small net can memorize unrelated
    grids at low norm by using each cell's one-hot neighborhood as a de
    facto address (documented, with the failing test kept, in
    tests/test_mdl_sanity.py) -- so norm-based bits under-charge exactly
    the behavior MDL is supposed to penalize. Phase 0's --variants run made
    the consequence concrete: the d4 variant solved two held-out tasks
    exactly and reconstructed the demos in fewer bits, but lost the MDL
    selection because fitting more data cost more norm.

    This is the variational (bits-back) code length instead: the sender
    transmits a posterior N(mean, sigma_q^2) rather than a point. sigma_q
    is the *imprecision the function tolerates*, and it is learned. A
    weight that must be exact to reproduce the data forces sigma_q small,
    and the log(sigma_p/sigma_q) term charges for every bit of precision
    demanded. A weight the function is robust to lets sigma_q rise to
    sigma_p and costs ~0. Memorization needs precision everywhere, so it
    becomes expensive; a robust rule does not, so it stays cheap. This is
    the same quantity CompressARC calls "KL for weights".

    Only meaningful when the forward pass uses weights SAMPLED from q (see
    fit.py's reparameterization): without sampling, nothing opposes
    sigma_q growing to sigma_p and the code collapses to zero cost for
    free. Like bits_gaussian, this is a relative measure -- valid to
    compare at matched parameter count and prior.
    """
    var_q = torch.exp(2.0 * log_sigma_q)
    nats = (
        math.log(sigma_p) - log_sigma_q
        + (var_q + mean.pow(2)) / (2.0 * sigma_p * sigma_p)
        - 0.5
    ).sum()
    return nats / LN2

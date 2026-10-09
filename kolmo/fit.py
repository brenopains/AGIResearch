"""MDL fit of a single task via gradient descent. No search, no branching.

Structural rule (see plan): this function receives only tensors -- no
task_id, no label, nothing that could let a caller special-case an answer.
That is what makes the "the test produces its own answer" failure mode
found throughout the audited codebase structurally unreachable here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.func import functional_call

from .codec import bits_crossentropy, bits_gaussian, bits_kl_gaussian
from .net import TinyGridNet, param_count

GAUSSIAN = "gaussian"          # point code: bits ~ ||theta||^2 (Phases 0-2)
VARIATIONAL = "variational"    # posterior code: bits = KL(q||p), prices precision


@dataclass
class FitResult:
    bits_total: float
    bits_recon: float
    bits_weights: float
    steps: int
    seconds: float
    peak_vram_mb: float
    theta: torch.Tensor   # detached flat parameter vector (mu + B @ a + r if library given)
    param_names: list
    param_shapes: list
    net: TinyGridNet       # loaded with the final theta, ready for eval
    # variational only: per-weight posterior log-std in weight space, so a
    # caller can estimate E_q[CE] on data the fit did not train on. Exact
    # when library is None; a diagonal approximation otherwise (it drops
    # the off-diagonal covariance that B a induces).
    theta_log_sigma: torch.Tensor = None


def _param_layout(net: torch.nn.Module):
    names, shapes = [], []
    for n, p in net.named_parameters():
        names.append(n)
        shapes.append(p.shape)
    return names, shapes


def _split_theta(theta: torch.Tensor, names: list, shapes: list) -> dict:
    out = {}
    i = 0
    for n, s in zip(names, shapes):
        k = 1
        for d in s:
            k *= d
        out[n] = theta[i:i + k].view(s)
        i += k
    return out


def fit_task(
    train_pairs: list,
    hidden: int = 32,
    depth: int = 3,
    steps: int = 2000,
    lr: float = 1e-2,
    sigma_w: float = 1.0,
    seed: int = 0,
    device: str = "cuda",
    library=None,  # kolmo.library.Library or None
    symmetric: bool = False,   # D4-equivariant kernels (fewer params to transmit)
    codec: str = GAUSSIAN,
    init_sigma_q_frac: float = 0.1,  # variational only: sigma_q starts at this * sigma_prior
    mc_samples: int = 16,            # variational only: samples for the final recon estimate
) -> FitResult:
    torch.manual_seed(seed)
    net = TinyGridNet(hidden=hidden, depth=depth, symmetric=symmetric).to(device)
    names, shapes = _param_layout(net)
    p = param_count(net)

    # "code variables": the things actually transmitted. Under GAUSSIAN each
    # is a point; under VARIATIONAL each is a posterior mean paired with a
    # learned log sigma. Everything below is written once against this list
    # so the two codecs cannot drift apart.
    if library is None:
        means = [torch.nn.utils.parameters_to_vector(net.parameters()).detach().clone()]
        priors = [sigma_w]
    else:
        assert library.B.shape[0] == p, (
            f"library param dim {library.B.shape[0]} != net param dim {p}; "
            "hidden/depth must match the config the library was built with"
        )
        k = library.B.shape[1]
        # a=0, r=0 => the fit starts at theta=mu, the library's shared prior
        # weight vector (the old theta=0 start left the whole net dead at
        # step 0 -- stem and blocks all-zero -- unlike the Kaiming start of
        # library=None fits, which made conditions incomparable to Phase 0).
        means = [torch.zeros(k, device=device), torch.zeros(p, device=device)]
        priors = [library.sigma_a, library.sigma_r]
    for m in means:
        m.requires_grad_(True)
    opt_params = list(means)

    variational = codec == VARIATIONAL
    if variational:
        log_sigmas = [
            torch.full_like(m, float(torch.log(torch.tensor(init_sigma_q_frac * s))), requires_grad=True)
            for m, s in zip(means, priors)
        ]
        opt_params += log_sigmas
    else:
        log_sigmas = [None] * len(means)

    def assemble(vals: list) -> torch.Tensor:
        return vals[0] if library is None else library.mu + library.B @ vals[0] + vals[1]

    def draw() -> list:
        if not variational:
            return means
        return [m + torch.exp(ls) * torch.randn_like(m) for m, ls in zip(means, log_sigmas)]

    def weight_bits() -> torch.Tensor:
        if variational:
            return sum(bits_kl_gaussian(m, ls, s) for m, ls, s in zip(means, log_sigmas, priors))
        return sum(bits_gaussian(m, s) for m, s in zip(means, priors))

    opt = torch.optim.Adam(opt_params, lr=lr)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # one batched forward per step instead of one per pair -- identical sum
    # of per-pair losses (see bits_crossentropy), just vectorized
    xs = torch.stack([p.x for p in train_pairs])
    ys = torch.stack([p.y_idx for p in train_pairs])
    ms = torch.stack([p.mask for p in train_pairs])

    def recon_bits_of(vals: list) -> torch.Tensor:
        logits = functional_call(net, _split_theta(assemble(vals), names, shapes), (xs,))
        return bits_crossentropy(logits, ys, ms)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        (recon_bits_of(draw()) + weight_bits()).backward()
        opt.step()

    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else 0.0

    with torch.no_grad():
        w_bits_final = weight_bits().item()
        th_final = assemble(means).detach()
        if variational:
            # the transmitted code pays E_q[CE], not CE at the mean: a
            # posterior wide enough to be cheap must still reconstruct
            recon_final = sum(recon_bits_of(draw()).item() for _ in range(mc_samples)) / mc_samples
        else:
            recon_final = recon_bits_of(means).item()
        # net carries the posterior mean -- the decoder's point estimate,
        # which is what exact_match should be judged on
        torch.nn.utils.vector_to_parameters(th_final, net.parameters())

        th_log_sigma = None
        if variational:
            if library is None:
                th_log_sigma = log_sigmas[0].detach()
            else:
                var = (library.B.pow(2) @ torch.exp(2.0 * log_sigmas[0])) + torch.exp(2.0 * log_sigmas[1])
                th_log_sigma = (0.5 * torch.log(var)).detach()

    return FitResult(
        bits_total=recon_final + w_bits_final,
        bits_recon=recon_final,
        bits_weights=w_bits_final,
        steps=steps,
        seconds=dt,
        peak_vram_mb=peak,
        theta=th_final.detach().cpu(),
        param_names=names,
        param_shapes=shapes,
        net=net,
        theta_log_sigma=th_log_sigma,
    )


@dataclass
class MixtureFitResult:
    fit: FitResult          # the full fit under the selected component
    bits_total: float       # fit.bits_total + mixture.bits_index
    selected: int           # index of the component that won the probe
    probe_bits: list        # per-component probe bits, for auditing the choice


def fit_task_mixture(
    train_pairs: list,
    mixture,                 # kolmo.library.Mixture
    probe_steps: int = 200,
    select_by: str = "auto",
    **kw,
) -> MixtureFitResult:
    """Select a prototype with a short probe fit, then fit it fully.

    Two-speed by design (the plan's step 1 vs. step 2): a cheap pass ranks
    the prototypes, one expensive pass commits. The probe sees only the
    demonstration pairs -- like fit_task, this function never receives a
    task id, a label, or the test pairs, so the selection cannot be
    informed by the answer.

    Probes are run at a fixed seed so the ranking is a property of the
    prototypes, not of initialization luck; the committed fit uses the
    caller's seed.

    select_by controls what the probe ranks on:
      "total" -- the probe's own bits_total, under whatever codec is in use.
      "fast"  -- reconstruction only. Under the variational codec a short
                 probe is dominated by an un-converged KL floor identical
                 across prototypes: measured separation between prototypes
                 collapses from 86% (Gaussian, 200 steps) to 0.46%, i.e.
                 the selection becomes a coin flip. Ranking on the term
                 that converges fast restores it. This does NOT weaken the
                 accounting -- the selected prototype is still charged the
                 full code, index included; only the ranking is cheap.
      "auto"  -- "fast" under the variational codec, "total" otherwise.
    """
    mode = select_by
    if mode == "auto":
        mode = "fast" if kw.get("codec", GAUSSIAN) == VARIATIONAL else "total"
    assert mode in ("total", "fast"), f"unknown select_by: {select_by}"

    probe_bits = []
    for comp in mixture.components:
        probe = fit_task(train_pairs, **{**kw, "steps": probe_steps, "seed": 0, "library": comp})
        probe_bits.append(probe.bits_recon if mode == "fast" else probe.bits_total)
    selected = int(min(range(len(probe_bits)), key=probe_bits.__getitem__))
    full = fit_task(train_pairs, **{**kw, "library": mixture.components[selected]})
    return MixtureFitResult(
        fit=full,
        bits_total=full.bits_total + mixture.bits_index,
        selected=selected,
        probe_bits=probe_bits,
    )


@torch.no_grad()
def exact_match(net: torch.nn.Module, pairs: list) -> bool:
    """True iff every held-out pair is predicted exactly, cell for cell."""
    for pair in pairs:
        logits = net(pair.x)
        pred = logits.argmax(dim=0)
        correct = (pred == pair.y_idx) | ~pair.mask
        if not bool(correct.all()):
            return False
    return True

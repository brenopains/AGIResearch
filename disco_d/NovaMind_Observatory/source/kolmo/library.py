"""Continuous weight-space library: W = mu + B @ a + r.

Phase 1's construction of B is deliberately simple: PCA over the fitted
weight vectors of independently-solved "construction" tasks (Phase 0
style, library=None). This is NOT yet the MDL-driven promotion / pruning /
retroactive re-description that Phase 2 ("sono") describes in the plan --
it is the minimum viable "real" library needed to run the Phase 1
death-criterion ablation (real vs. shuffled vs. empty) honestly. If this
ablation doesn't show real beating shuffled, the retroactive-promotion
machinery in Phase 2 has nothing to build on, so it's the right thing to
build first.

Phase 1.5 change: the mean of the fitted weight vectors is KEPT as `mu`
instead of being discarded after centering. The mean is the single most
shared piece of structure across tasks (the "generic grid-net" every fit
converges near), and under the tight residual prior (sigma_r) it was the
dominant per-task cost in every condition -- burying whatever advantage
the real basis had over the shuffled one. mu is library content: it is
paid for once at construction time and is identical in size across the
conditions being compared, so it does not enter the per-task bits (same
argument codec.py makes for the dropped entropy constant). What differs
between conditions is only whether mu/B carry structure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Library:
    mu: torch.Tensor   # (P,), mean of construction fits; per-task fits start here
    B: torch.Tensor    # (P, K), frozen basis
    sigma_a: float       # prior std for coefficients -- generous: cheap to use
    sigma_r: float       # prior std for residual -- tight: expensive to deviate

    @staticmethod
    def from_weight_vectors(thetas: torch.Tensor, k: int, sigma_a: float = 3.0, sigma_r: float = 0.3) -> "Library":
        """thetas: (N, P) fitted weight vectors from N independent task fits, N << P."""
        mu = thetas.mean(dim=0)
        centered = thetas - mu.unsqueeze(0)
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        k = min(k, Vh.shape[0])
        B = Vh[:k].T.contiguous()  # (P, k), orthonormal columns
        return Library(mu=mu, B=B, sigma_a=sigma_a, sigma_r=sigma_r)

    def shuffled(self, seed: int = 0) -> "Library":
        """Same shape and per-vector value distribution; structure destroyed.

        mu and each atom (column of B) independently permuted across their
        P entries. This is the control: if bits under this library aren't
        worse than the real one, the real library isn't encoding anything
        the shuffled one doesn't -- see plan's death criterion for Phase 1.
        mu is shuffled too: the real mean's structure is part of what the
        library claims to know, so the control must destroy it as well.
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        p = self.B.shape[0]
        mu_shuf = self.mu[torch.randperm(p, generator=g)].clone()
        B_shuf = self.B.clone()
        for j in range(B_shuf.shape[1]):
            perm = torch.randperm(p, generator=g)
            B_shuf[:, j] = self.B[perm, j]
        return Library(mu=mu_shuf, B=B_shuf, sigma_a=self.sigma_a, sigma_r=self.sigma_r)

    def empty(self) -> "Library":
        """mu and B zeroed out: everything must go through r.

        Holds sigma_a/sigma_r and the presence of the (a, r) split constant
        across conditions -- the only thing that changes is whether mu/B
        carry structure. This is the same test as the NovaMind cge=None
        ablation that 4 of 7 trials failed to run -- run here on purpose,
        before any claim is made.
        """
        return Library(
            mu=torch.zeros_like(self.mu), B=torch.zeros_like(self.B),
            sigma_a=self.sigma_a, sigma_r=self.sigma_r,
        )


@dataclass
class Mixture:
    """M specialized libraries; a task selects one and pays log2(M) bits.

    Phase 2's differentiation step. Phase 1.5 measured a single global mu
    winning hugely on 6 of 15 held-out tasks and losing slightly on 9 --
    the signature of one average prior that fits a subpopulation and drags
    on everything else. A single prototype is a single cell type. This
    splits it: cluster the (aligned) construction fits, give each cluster
    its own mu and local basis, and let each task select the prototype
    that describes it in fewest bits.

    The selection is NOT free: log2(M) bits are added to every task's cost,
    so a mixture that splits without cause loses to the global library on
    the ledger. That price is what stops M from growing to N (one
    prototype per task = memorizing the construction set).
    """
    components: list      # list[Library]
    bits_index: float     # log2(M): cost of naming the selected prototype

    @staticmethod
    def from_components(components: list) -> "Mixture":
        assert components, "a mixture needs at least one component"
        return Mixture(components=components, bits_index=math.log2(len(components)))

    @staticmethod
    def from_weight_vectors(thetas: torch.Tensor, m: int, k: int, seed: int = 0,
                            sigma_a: float = 3.0, sigma_r: float = 0.3,
                            iters: int = 25) -> "Mixture":
        """k-means over fitted weight vectors, then one Library per cluster."""
        labels = kmeans(thetas, m, seed=seed, iters=iters)
        comps = []
        for c in range(int(labels.max().item()) + 1):
            member = thetas[labels == c]
            # a cluster of 1 has no spread to PCA: k collapses to 0 columns,
            # leaving a pure prototype (mu) with no local basis, which is
            # the honest description of "one example, no variation seen"
            k_c = min(k, max(0, member.shape[0] - 1))
            comps.append(Library.from_weight_vectors(member, k=k_c, sigma_a=sigma_a, sigma_r=sigma_r))
        return Mixture.from_components(comps)

    def shuffled(self, seed: int = 0) -> "Mixture":
        return Mixture(components=[c.shuffled(seed + i) for i, c in enumerate(self.components)],
                       bits_index=self.bits_index)

    def empty(self) -> "Mixture":
        return Mixture(components=[c.empty() for c in self.components], bits_index=self.bits_index)

    def keep(self, idx: list) -> "Mixture":
        """Pruned mixture holding only components `idx`, re-priced.

        Dropping unused prototypes is not cosmetic: it lowers bits_index
        for every remaining task. Pruning that does not pay for itself
        shows up immediately as a worse total.
        """
        return Mixture.from_components([self.components[i] for i in idx])


def kmeans(x: torch.Tensor, m: int, seed: int = 0, iters: int = 25) -> torch.Tensor:
    """Plain k-means++ init + Lloyd. x: (N, P). Returns (N,) long labels.

    Empty clusters are dropped rather than re-seeded, so the returned label
    set may name fewer than m clusters -- an honest "the data did not
    support m prototypes" instead of a padded count.
    """
    n = x.shape[0]
    m = min(m, n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = [x[torch.randint(n, (1,), generator=g).item()]]
    for _ in range(m - 1):
        d = torch.stack([(x - c).pow(2).sum(dim=1) for c in centers]).min(dim=0).values
        probs = d / d.sum() if d.sum() > 0 else torch.full((n,), 1.0 / n, device=x.device)
        centers.append(x[torch.multinomial(probs.cpu(), 1, generator=g).item()])
    C = torch.stack(centers)
    labels = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        labels = torch.cdist(x, C).argmin(dim=1)
        new = []
        for c in range(C.shape[0]):
            member = x[labels == c]
            if member.shape[0] > 0:
                new.append(member.mean(dim=0))
        C_new = torch.stack(new)
        if C_new.shape == C.shape and torch.allclose(C_new, C):
            C = C_new
            break
        C = C_new
    labels = torch.cdist(x, C).argmin(dim=1)
    # relabel to a contiguous 0..K-1 range after any cluster died
    uniq = torch.unique(labels)
    remap = torch.zeros(int(uniq.max().item()) + 1, dtype=torch.long, device=x.device)
    for new_id, old in enumerate(uniq):
        remap[old] = new_id
    return remap[labels]

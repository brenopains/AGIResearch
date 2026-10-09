"""Contract tests for the Phase 2 mixture: differentiation, priced selection,
and pruning. The point of these is that a mixture must not be able to look
good for free -- the log2(M) index is what stops "one prototype per task"."""
import math

import torch

from kolmo.fit import fit_task, fit_task_mixture
from kolmo.grids import grid_to_pair
from kolmo.library import Library, Mixture, kmeans


def _pairs(kind: str):
    if kind == "stripes":
        g = [[(r % 2) * 3 for _ in range(8)] for r in range(8)]
    else:
        g = [[(c % 2) * 5 for c in range(8)] for _ in range(8)]
    return [grid_to_pair(g, g, device="cpu")]


def _fit(pairs, **kw):
    return fit_task(pairs, hidden=8, depth=1, steps=150, lr=2e-2, seed=0, device="cpu", **kw)


def test_index_cost_is_charged_and_scales_with_m():
    lib = Library.from_weight_vectors(torch.randn(4, 60), k=2)
    one = Mixture.from_components([lib])
    four = Mixture.from_components([lib] * 4)
    assert one.bits_index == 0.0, "a single prototype names itself for free"
    assert abs(four.bits_index - 2.0) < 1e-9, "4 prototypes cost log2(4) = 2 bits"


def test_mixture_of_one_equals_plain_library_plus_zero_bits():
    pairs = _pairs("stripes")
    base = _fit(pairs)
    lib = Library.from_weight_vectors(base.theta.unsqueeze(0).repeat(3, 1), k=1)
    plain = _fit(pairs, library=lib)
    mixed = fit_task_mixture(pairs, Mixture.from_components([lib]), probe_steps=20,
                             hidden=8, depth=1, steps=150, lr=2e-2, seed=0, device="cpu")
    assert mixed.selected == 0
    assert abs(mixed.bits_total - plain.bits_total) < 1e-4


def test_probe_selects_the_matching_prototype():
    # two dissimilar tasks -> two prototypes; each task must pick its own
    fits = {k: _fit(_pairs(k)) for k in ("stripes", "bars")}
    comps = [
        Library.from_weight_vectors(fits[k].theta.unsqueeze(0).repeat(3, 1), k=1)
        for k in ("stripes", "bars")
    ]
    mix = Mixture.from_components(comps)
    for idx, kind in enumerate(("stripes", "bars")):
        got = fit_task_mixture(_pairs(kind), mix, probe_steps=60, hidden=8, depth=1,
                               steps=150, lr=2e-2, seed=0, device="cpu")
        assert got.selected == idx, (
            f"task '{kind}' selected prototype {got.selected}, probe bits {got.probe_bits}"
        )


def test_kmeans_separates_two_clear_clusters():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(6, 40, generator=g) * 0.05 + 10.0
    b = torch.randn(6, 40, generator=g) * 0.05 - 10.0
    labels = kmeans(torch.cat([a, b]), m=2, seed=0)
    assert len(set(labels[:6].tolist())) == 1 and len(set(labels[6:].tolist())) == 1
    assert labels[0] != labels[6]


def test_kmeans_drops_empty_clusters_instead_of_padding():
    x = torch.randn(3, 20)
    labels = kmeans(x, m=8, seed=0)
    assert int(labels.max().item()) + 1 <= 3, "cannot have more prototypes than points"
    assert set(labels.tolist()) == set(range(int(labels.max().item()) + 1)), "labels must be contiguous"


def test_pruning_reprices_the_index():
    lib = Library.from_weight_vectors(torch.randn(4, 60), k=2)
    mix = Mixture.from_components([lib] * 8)
    kept = mix.keep([0, 1])
    assert len(kept.components) == 2
    assert abs(kept.bits_index - 1.0) < 1e-9, "pruning 8 -> 2 prototypes must lower the index cost"


def test_controls_preserve_component_count_and_index():
    thetas = torch.randn(8, 60)
    mix = Mixture.from_weight_vectors(thetas, m=2, k=2, seed=0)
    for ctrl in (mix.shuffled(seed=0), mix.empty()):
        assert len(ctrl.components) == len(mix.components)
        assert ctrl.bits_index == mix.bits_index, (
            "controls must differ only in whether mu/B carry structure"
        )

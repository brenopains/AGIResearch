"""Contract tests for kolmo.align. The first test is the module's license
to exist: if a "function-preserving" permutation changes the net's output,
every aligned theta is garbage and so is the PCA built on top of them."""
import torch

from kolmo.align import NetPerm, align_thetas, find_perm, permute_theta
from kolmo.fit import _param_layout, _split_theta
from kolmo.grids import IN_CHANNELS
from kolmo.net import TinyGridNet

HIDDEN, DEPTH = 16, 2  # num_groups = 8, group size 2 -- exercises both levels


def _random_group_perm(h: int, num_groups: int, g: torch.Generator) -> torch.Tensor:
    """Random GroupNorm-compatible permutation: permute groups, then within."""
    gs = h // num_groups
    group_order = torch.randperm(num_groups, generator=g)
    perm = torch.empty(h, dtype=torch.long)
    for new_g, old_g in enumerate(group_order):
        within = torch.randperm(gs, generator=g)
        perm[new_g * gs:(new_g + 1) * gs] = old_g * gs + within
    return perm


def _random_netperm(h: int, depth: int, seed: int) -> NetPerm:
    g = torch.Generator().manual_seed(seed)
    return NetPerm(
        stream=_random_group_perm(h, min(8, h), g),
        inter=[_random_group_perm(h, min(8, h), g) for _ in range(depth)],
    )


def _theta_of(net: TinyGridNet) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(net.parameters()).detach()


def _load(net: TinyGridNet, theta: torch.Tensor) -> TinyGridNet:
    torch.nn.utils.vector_to_parameters(theta, net.parameters())
    return net


def test_permutation_preserves_function():
    torch.manual_seed(0)
    net = TinyGridNet(hidden=HIDDEN, depth=DEPTH)
    names, shapes = _param_layout(net)
    theta = _theta_of(net)
    perm = _random_netperm(HIDDEN, DEPTH, seed=1)
    theta_p = permute_theta(theta, names, shapes, perm)

    x = torch.randn(IN_CHANNELS, 15, 15)
    with torch.no_grad():
        y0 = net(x)
        y1 = _load(TinyGridNet(hidden=HIDDEN, depth=DEPTH), theta_p)(x)
    # identical up to float summation reorder inside conv/GroupNorm
    assert torch.allclose(y0, y1, atol=1e-5), (
        f"permuted net diverged: max abs diff {(y0 - y1).abs().max().item():.3e}"
    )


def test_alignment_recovers_artificial_permutation():
    torch.manual_seed(0)
    net = TinyGridNet(hidden=HIDDEN, depth=DEPTH)
    names, shapes = _param_layout(net)
    theta = _theta_of(net)
    theta_p = permute_theta(theta, names, shapes, _random_netperm(HIDDEN, DEPTH, seed=2))

    dist_before = (theta - theta_p).norm().item()
    perm = find_perm(theta, theta_p, names, shapes)
    theta_rec = permute_theta(theta_p, names, shapes, perm)
    dist_after = (theta - theta_rec).norm().item()

    assert dist_after < 1e-5, (
        f"alignment failed to undo a pure permutation: residual {dist_after:.3e} "
        f"(unaligned distance was {dist_before:.3e})"
    )


def test_align_thetas_keeps_reference_and_shapes():
    torch.manual_seed(0)
    net = TinyGridNet(hidden=HIDDEN, depth=DEPTH)
    names, shapes = _param_layout(net)
    thetas = torch.stack([
        _theta_of(TinyGridNet(hidden=HIDDEN, depth=DEPTH)) for _ in range(3)
    ])
    aligned = align_thetas(thetas, names, shapes)
    assert aligned.shape == thetas.shape
    assert torch.equal(aligned[0], thetas[0])
    # aligned rows are permutations of the originals: same sorted values
    for i in range(1, 3):
        pd_orig = _split_theta(thetas[i], names, shapes)
        pd_alig = _split_theta(aligned[i], names, shapes)
        for n in names:
            assert torch.allclose(
                pd_orig[n].reshape(-1).sort().values,
                pd_alig[n].reshape(-1).sort().values,
            ), f"{n}: aligned tensor is not a permutation of the original"

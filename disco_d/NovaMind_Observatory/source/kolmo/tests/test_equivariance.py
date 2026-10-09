"""Contract tests for the D4-symmetric net.

The first test is the license to use it at all: if net(g.x) != g.net(x),
the "symmetry" is decoration and the parameter saving is a lie. The second
checks the saving is real, since that is the entire reason the symmetric
net can win an MDL selection the augmentation framing could not.
"""
import torch

from kolmo.grids import IN_CHANNELS, NUM_COLORS
from kolmo.net import D4Conv2d, TinyGridNet, param_count

HIDDEN, DEPTH, SIZE = 8, 2, 9


def _d4_elements():
    """The 8 elements of D4 as (rotations, flip) applied to the last 2 dims."""
    return [(k, f) for f in (False, True) for k in range(4)]


def _apply(t: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        t = torch.flip(t, dims=[-1])
    return torch.rot90(t, k, dims=(-2, -1))


def test_symmetric_net_is_equivariant_under_all_of_d4():
    torch.manual_seed(0)
    net = TinyGridNet(hidden=HIDDEN, depth=DEPTH, symmetric=True)
    x = torch.randn(IN_CHANNELS, SIZE, SIZE)
    with torch.no_grad():
        y = net(x)
        for k, flip in _d4_elements():
            got = net(_apply(x, k, flip))
            want = _apply(y, k, flip)
            assert torch.allclose(got, want, atol=1e-5), (
                f"not equivariant under (rot90 x{k}, flip={flip}): "
                f"max abs diff {(got - want).abs().max().item():.3e}"
            )


def test_plain_net_is_not_equivariant():
    """Characterization: the saving only means something if the baseline
    genuinely lacks the symmetry. If this ever passes, the two variants
    stopped being different hypotheses."""
    torch.manual_seed(0)
    net = TinyGridNet(hidden=HIDDEN, depth=DEPTH, symmetric=False)
    x = torch.randn(IN_CHANNELS, SIZE, SIZE)
    with torch.no_grad():
        y, got = net(x), net(_apply(x, 1, False))
    assert not torch.allclose(got, _apply(y, 1, False), atol=1e-5)


def test_symmetric_net_transmits_fewer_parameters():
    plain = param_count(TinyGridNet(hidden=HIDDEN, depth=DEPTH, symmetric=False))
    sym = param_count(TinyGridNet(hidden=HIDDEN, depth=DEPTH, symmetric=True))
    assert sym < plain, f"symmetric net ({sym}) must cost fewer parameters than plain ({plain})"
    # every 3x3 kernel drops 9 -> 3 coefficients; the 1x1 head is unchanged
    n_kernels = (IN_CHANNELS * HIDDEN) + DEPTH * 2 * HIDDEN * HIDDEN
    assert plain - sym == 6 * n_kernels, (
        f"expected exactly 6 fewer coefficients per 3x3 kernel ({6 * n_kernels}), got {plain - sym}"
    )


def test_d4conv_kernel_lies_in_the_invariant_subspace():
    torch.manual_seed(0)
    conv = D4Conv2d(2, 3)
    weight = torch.einsum("oik,khw->oihw", conv.coef, conv.basis)
    for k, flip in _d4_elements():
        assert torch.allclose(_apply(weight, k, flip), weight, atol=1e-6), (
            "an expanded kernel must be invariant under every element of D4"
        )


def test_symmetric_net_still_produces_correct_output_shape():
    net = TinyGridNet(hidden=HIDDEN, depth=DEPTH, symmetric=True)
    with torch.no_grad():
        assert net(torch.randn(IN_CHANNELS, SIZE, SIZE)).shape == (NUM_COLORS, SIZE, SIZE)
        assert net(torch.randn(4, IN_CHANNELS, SIZE, SIZE)).shape == (4, NUM_COLORS, SIZE, SIZE)

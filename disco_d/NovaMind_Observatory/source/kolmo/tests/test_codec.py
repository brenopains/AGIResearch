"""Unit tests for the bits accounting. If these don't hold, nothing built
on top of codec.py can be trusted."""
import math

import torch

from kolmo.codec import LN2, bits_crossentropy, bits_gaussian


def test_bits_crossentropy_matches_manual_calc():
    # 1x1 grid, 2 possible colors, logits picked so p(correct)=0.75
    logits = torch.zeros(2, 1, 1)
    logits[1, 0, 0] = math.log(3.0)  # softmax -> [0.25, 0.75]
    target = torch.tensor([[1]])
    mask = torch.tensor([[True]])
    bits = bits_crossentropy(logits, target, mask).item()
    expected = -math.log2(0.75)
    assert abs(bits - expected) < 1e-4


def test_bits_crossentropy_ignores_masked_cells():
    logits = torch.randn(3, 2, 2)
    target = torch.zeros(2, 2, dtype=torch.long)
    mask_all = torch.ones(2, 2, dtype=torch.bool)
    mask_none = torch.zeros(2, 2, dtype=torch.bool)
    assert bits_crossentropy(logits, target, mask_none).item() == 0.0
    assert bits_crossentropy(logits, target, mask_all).item() > 0.0


def test_bits_gaussian_zero_at_origin():
    params = torch.zeros(100)
    assert bits_gaussian(params, sigma=1.0).item() == 0.0


def test_bits_gaussian_scales_with_squared_norm():
    p1 = torch.ones(10)
    p2 = torch.ones(10) * 2.0
    b1 = bits_gaussian(p1, sigma=1.0).item()
    b2 = bits_gaussian(p2, sigma=1.0).item()
    # doubling the params should exactly quadruple the (purely quadratic) bits
    assert abs(b2 - 4 * b1) < 1e-4


def test_bits_gaussian_tighter_prior_costs_more():
    params = torch.ones(10) * 0.5
    loose = bits_gaussian(params, sigma=2.0).item()
    tight = bits_gaussian(params, sigma=0.2).item()
    assert tight > loose


def test_bits_gaussian_matches_closed_form():
    params = torch.tensor([1.0, -2.0, 3.0])
    sigma = 0.7
    got = bits_gaussian(params, sigma).item()
    expected = float(params.pow(2).sum()) / (2 * sigma * sigma * LN2)
    assert abs(got - expected) < 1e-5

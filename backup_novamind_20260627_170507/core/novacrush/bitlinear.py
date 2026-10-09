"""
NovaCrush — BitLinear: Ternary Weight Neural Layers
=====================================================
Implements BitNet b1.58 — weights constrained to {-1, 0, +1}.
Replaces FP matrix multiplication with integer add/subtract.
10x less memory, 71x less energy per operation.

References:
    - Ma et al. (2024): "The Era of 1-bit LLMs"
    - Wang et al. (2023): "BitNet: Scaling 1-Bit Transformers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class StraightThroughEstimator(torch.autograd.Function):
    """STE for ternary quantization. Forward: quantize. Backward: identity."""

    @staticmethod
    def forward(ctx, input_tensor, threshold=0.3):
        ctx.save_for_backward(input_tensor)
        output = torch.zeros_like(input_tensor)
        output[input_tensor > threshold] = 1.0
        output[input_tensor < -threshold] = -1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_mask = (input_tensor.abs() <= 1.0).float()
        return grad_output * grad_mask, None


def ternary_quantize(weights, threshold=0.3):
    """Functional ternary quantization with STE."""
    return StraightThroughEstimator.apply(weights, threshold)


class AbsMeanQuantizer:
    """Adaptive threshold using mean absolute value scaling."""

    @staticmethod
    def quantize(weights, gamma=0.7):
        abs_mean = weights.abs().mean()
        threshold = gamma * abs_mean
        scale = abs_mean.item()
        quantized = StraightThroughEstimator.apply(weights, threshold.item())
        return quantized, scale


class BitLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.
    Drop-in replacement for nn.Linear.

    Memory: 512x512 FP32 = 1MB → BitLinear = 64KB (16x savings)
    FLOPs: multiply-accumulate → add/subtract only (~5x throughput)
    """

    def __init__(self, in_features, out_features, bias=False, gamma=0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma

        # Full-precision shadow weights for gradient accumulation
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.activation_scale = nn.Parameter(torch.ones(1))

        self._sparsity_ratio = 0.0
        self._quantization_error = 0.0

    def forward(self, x):
        # Quantize weights to {-1, 0, +1}
        w_quant, w_scale = AbsMeanQuantizer.quantize(self.weight, self.gamma)

        with torch.no_grad():
            self._sparsity_ratio = (w_quant == 0).float().mean().item()
            self._quantization_error = (self.weight - w_quant * w_scale).abs().mean().item()

        # Quantize activations to INT8
        x_norm = self._quantize_activations(x)

        # Ternary matmul (add/subtract, no multiply)
        output = F.linear(x_norm, w_quant, self.bias)
        return output * w_scale * self.activation_scale

    def _quantize_activations(self, x):
        with torch.no_grad():
            abs_max = x.abs().max()
            if abs_max < 1e-8:
                return x
            scale = 127.0 / abs_max
        x_quant = torch.clamp(torch.round(x * scale), -128, 127) / scale
        return x + (x_quant - x).detach()

    def get_compression_stats(self):
        fp32_bytes = self.in_features * self.out_features * 4
        ternary_bytes = self.in_features * self.out_features * 2 / 8
        return {
            'fp32_bytes': fp32_bytes,
            'ternary_bytes': ternary_bytes,
            'compression_ratio': fp32_bytes / max(1, ternary_bytes),
            'sparsity': self._sparsity_ratio,
            'quantization_error': self._quantization_error,
        }


class BitLinearDynamic(BitLinear):
    """
    Dynamic BitLinear with runtime neurogenesis support.
    Can grow new neurons dynamically for continuous learning.
    """

    def __init__(self, in_features, out_features, max_growth=4.0, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.initial_in = in_features
        self.initial_out = out_features
        self.max_out = int(out_features * max_growth)
        self.growth_count = 0

    def expand_output(self, num_new=1):
        """Grow output dimension (add neurons). Returns count added."""
        new_out = min(self.out_features + num_new, self.max_out)
        added = new_out - self.out_features
        if added <= 0:
            return 0
        with torch.no_grad():
            new_w = torch.randn(added, self.in_features, device=self.weight.device) * 0.01
            self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=0))
            self.out_features = new_out
            if self.bias is not None:
                new_b = torch.zeros(added, device=self.bias.device)
                self.bias = nn.Parameter(torch.cat([self.bias.data, new_b]))
        self.growth_count += added
        return added

    def get_growth_stats(self):
        return {
            'initial_size': f'{self.initial_in}x{self.initial_out}',
            'current_size': f'{self.in_features}x{self.out_features}',
            'total_growth': self.growth_count,
            **self.get_compression_stats(),
        }

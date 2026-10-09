"""
NovaCrush — Forward-Forward Training Engine
==============================================
Hinton's Forward-Forward algorithm: local layer-wise training
without backpropagation. Each layer learns its own "goodness" function.

Memory advantage: O(N) vs O(N*L) for backprop — independent of depth.
For 100 layers: ~100x less training memory.

References:
    - Hinton (2022): "The Forward-Forward Algorithm"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ForwardForwardLayer(nn.Module):
    """
    A single layer that trains via the Forward-Forward algorithm.
    
    Instead of backprop, each layer learns to produce:
    - HIGH "goodness" (sum of squared activations) for positive data
    - LOW "goodness" for negative data
    
    Threshold θ separates positive from negative.
    No gradient graph needed — each layer is independent!
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)

        # Statistics
        self._pos_goodness = 0.0
        self._neg_goodness = 0.0
        self._accuracy = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward — normalize then linear + ReLU."""
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x_norm))

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness = sum of squared activations per sample."""
        h = self.forward(x)
        return h.pow(2).sum(dim=-1)  # [batch_size]

    def train_step(self, pos_data: torch.Tensor, neg_data: torch.Tensor) -> Dict:
        """
        One Forward-Forward training step.
        
        Goal: goodness(positive) > θ AND goodness(negative) < θ
        Loss = -log(σ(goodness_pos - θ)) - log(σ(θ - goodness_neg))
        
        KEY: No backprop through previous layers! Only local gradients.
        Memory: O(batch_size × layer_width) — constant per layer.
        """
        self.optimizer.zero_grad()

        # Compute goodness for positive and negative data
        g_pos = self.goodness(pos_data)
        g_neg = self.goodness(neg_data)

        # Loss: push pos above threshold, neg below
        loss_pos = -F.logsigmoid(g_pos - self.threshold).mean()
        loss_neg = -F.logsigmoid(self.threshold - g_neg).mean()
        loss = loss_pos + loss_neg

        # LOCAL gradient update — no graph needed beyond this layer
        loss.backward()
        self.optimizer.step()

        # Statistics
        with torch.no_grad():
            self._pos_goodness = g_pos.mean().item()
            self._neg_goodness = g_neg.mean().item()
            correct_pos = (g_pos > self.threshold).float().mean()
            correct_neg = (g_neg < self.threshold).float().mean()
            self._accuracy = ((correct_pos + correct_neg) / 2).item()

        return {
            'loss': loss.item(),
            'pos_goodness': self._pos_goodness,
            'neg_goodness': self._neg_goodness,
            'accuracy': self._accuracy,
        }

    def get_output_for_next(self, x: torch.Tensor) -> torch.Tensor:
        """Get detached output to feed to next layer (no grad graph)."""
        with torch.no_grad():
            return self.forward(x)


class ForwardForwardNetwork(nn.Module):
    """
    Multi-layer Forward-Forward network.
    
    Each layer trains independently — total training memory is
    O(batch × max_layer_width) instead of O(batch × total_params).
    
    For a network with layers [512, 512, 512, 512]:
        Backprop memory: ~4x (all layers stored for backward)
        FF memory: ~1x (only current layer in memory)
    """

    def __init__(self, layer_sizes: List[int], threshold: float = 2.0,
                 lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                ForwardForwardLayer(layer_sizes[i], layer_sizes[i + 1],
                                   threshold, lr)
            )
        self.layer_sizes = layer_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train_all_layers(self, pos_data: torch.Tensor,
                         neg_data: torch.Tensor) -> Dict:
        """
        Train all layers using Forward-Forward.
        
        Each layer:
        1. Gets its input (detached — no gradient flow between layers!)
        2. Trains locally to distinguish positive from negative
        3. Passes its output to the next layer
        """
        reports = []
        pos_input = pos_data
        neg_input = neg_data

        for i, layer in enumerate(self.layers):
            report = layer.train_step(pos_input, neg_input)
            report['layer'] = i
            reports.append(report)

            # Get outputs for next layer — DETACHED (this is the key!)
            pos_input = layer.get_output_for_next(pos_input)
            neg_input = layer.get_output_for_next(neg_input)

        avg_loss = sum(r['loss'] for r in reports) / len(reports)
        avg_acc = sum(r['accuracy'] for r in reports) / len(reports)

        return {
            'layer_reports': reports,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_acc,
            'num_layers': len(self.layers),
        }

    def predict_goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Get total goodness across all layers (for classification)."""
        total_goodness = torch.zeros(x.shape[0], device=x.device)
        current = x
        for layer in self.layers:
            current = layer.forward(current)
            total_goodness += current.pow(2).sum(dim=-1)
        return total_goodness

    def get_memory_comparison(self) -> Dict:
        """Compare memory usage vs equivalent backprop network."""
        total_params = sum(p.numel() for p in self.parameters())
        max_layer_params = max(
            sum(p.numel() for p in layer.parameters())
            for layer in self.layers
        )
        backprop_memory = total_params * 4 * 3  # weights + grads + optimizer
        ff_memory = max_layer_params * 4 * 3  # only largest layer

        return {
            'total_params': total_params,
            'max_layer_params': max_layer_params,
            'backprop_bytes': backprop_memory,
            'ff_bytes': ff_memory,
            'memory_ratio': backprop_memory / max(1, ff_memory),
            'savings_percent': (1 - ff_memory / max(1, backprop_memory)) * 100,
        }

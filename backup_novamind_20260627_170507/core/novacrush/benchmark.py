"""
NovaCrush — Benchmark: Compare Original vs Crushed Substrate
===============================================================
Runs side-by-side comparison of DynamicPredictiveNetwork (FP32)
vs CrushedSubstrate (BitLinear + FF + HDC) on identical inputs.

Measures: VRAM, speed, surprise convergence, topology quality.
"""

import torch
import time
import sys
import os
import numpy as np
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.neural_substrate import DynamicPredictiveNetwork
from core.novacrush.crushed_substrate import CrushedSubstrate


def format_bytes(b: float) -> str:
    """Format bytes into human-readable string."""
    if b < 1024:
        return f"{b:.0f} B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 ** 3:
        return f"{b / (1024 ** 2):.1f} MB"
    else:
        return f"{b / (1024 ** 3):.2f} GB"


def get_gpu_memory() -> Dict:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return {
            'allocated': allocated,
            'reserved': reserved,
            'allocated_str': format_bytes(allocated),
            'reserved_str': format_bytes(reserved),
        }
    return {'allocated': 0, 'reserved': 0, 'allocated_str': 'CPU', 'reserved_str': 'CPU'}


def count_parameters(model) -> Dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_bytes = total * 4
    ternary_bytes = total * 2 / 8  # 2 bits per ternary value
    return {
        'total': total,
        'trainable': trainable,
        'fp32_bytes': fp32_bytes,
        'fp32_str': format_bytes(fp32_bytes),
        'ternary_bytes': ternary_bytes,
        'ternary_str': format_bytes(ternary_bytes),
    }


def benchmark_training(model, test_data: List[List[str]], label: str) -> Dict:
    """Benchmark training speed and surprise convergence."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mem_before = get_gpu_memory()
    surprises = []
    times = []

    for epoch, tokens in enumerate(test_data):
        t0 = time.perf_counter()
        surprise = model.continuous_train(tokens)
        t1 = time.perf_counter()
        surprises.append(surprise)
        times.append(t1 - t0)

    mem_after = get_gpu_memory()

    return {
        'label': label,
        'total_time_ms': sum(times) * 1000,
        'avg_time_ms': np.mean(times) * 1000,
        'initial_surprise': surprises[0] if surprises else 0,
        'final_surprise': surprises[-1] if surprises else 0,
        'surprise_reduction': (surprises[0] - surprises[-1]) / max(0.001, surprises[0]) if surprises else 0,
        'convergence_trend': 'decreasing' if len(surprises) > 2 and surprises[-1] < surprises[0] else 'stable',
        'mem_before': mem_before,
        'mem_after': mem_after,
        'mem_delta_bytes': mem_after['allocated'] - mem_before['allocated'],
    }


def run_benchmark(initial_concepts: int = 32, embedding_dim: int = 64,
                  num_iterations: int = 50):
    """Run full benchmark comparing Original vs NovaCrush."""

    print("=" * 70)
    print("  NovaCrush Benchmark: Original FP32 vs CrushedSubstrate")
    print("=" * 70)
    print(f"\nConfig: {initial_concepts} initial concepts, {embedding_dim}-dim embeddings")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {format_bytes(torch.cuda.get_device_properties(0).total_mem)}")

    # Generate test data
    test_corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["artificial", "intelligence", "requires", "computation"],
        ["consciousness", "emerges", "from", "complexity"],
        ["neural", "networks", "learn", "patterns"],
        ["hyperbolic", "geometry", "models", "hierarchy"],
        ["quantum", "computing", "uses", "superposition"],
        ["the", "brain", "processes", "information", "continuously"],
        ["learning", "requires", "surprise", "and", "prediction"],
        ["geometric", "algebra", "encodes", "transformations"],
        ["memory", "consolidation", "prevents", "forgetting"],
    ]
    # Repeat to get enough iterations
    test_data = (test_corpus * (num_iterations // len(test_corpus) + 1))[:num_iterations]

    # === ORIGINAL MODEL ===
    print("\n" + "-" * 40)
    print("  Testing ORIGINAL (FP32)")
    print("-" * 40)
    original = DynamicPredictiveNetwork(initial_concepts, embedding_dim)
    orig_params = count_parameters(original)
    print(f"  Parameters: {orig_params['total']:,} ({orig_params['fp32_str']})")

    orig_results = benchmark_training(original, test_data, "Original FP32")
    print(f"  Training time: {orig_results['total_time_ms']:.1f} ms")
    print(f"  Surprise: {orig_results['initial_surprise']:.4f} -> {orig_results['final_surprise']:.4f}")
    print(f"  Vocab size after: {original.vocab_size}")

    # === NOVACRUSH MODEL ===
    print("\n" + "-" * 40)
    print("  Testing NOVACRUSH (BitLinear + FF + HDC)")
    print("-" * 40)
    crushed = CrushedSubstrate(initial_concepts, embedding_dim, hdc_dim=4096)
    crush_params = count_parameters(crushed)
    print(f"  Parameters: {crush_params['total']:,} ({crush_params['fp32_str']} FP32 / {crush_params['ternary_str']} ternary)")

    crush_results = benchmark_training(crushed, test_data, "NovaCrush")
    print(f"  Training time: {crush_results['total_time_ms']:.1f} ms")
    print(f"  Surprise: {crush_results['initial_surprise']:.4f} -> {crush_results['final_surprise']:.4f}")
    print(f"  Vocab size after: {crushed.vocab_size}")

    # === COMPARISON ===
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    # Compression stats
    crush_stats = crushed.get_compression_stats()
    orig_total = crush_stats['original_fp32']['total_bytes']
    crush_inference = crush_stats['novacrush_inference']['total_bytes']

    print(f"\n  Memory (training mode):")
    print(f"    Original:  {format_bytes(orig_total)}")
    print(f"    NovaCrush: {format_bytes(crush_stats['novacrush_training']['total_bytes'])}")

    print(f"\n  Memory (inference mode, ternary-packed):")
    print(f"    Original:  {format_bytes(orig_total)}")
    print(f"    NovaCrush: {format_bytes(crush_inference)}")
    print(f"    Compression: {crush_stats['novacrush_inference']['compression_vs_fp32']:.1f}x")

    print(f"\n  Speed:")
    speedup = orig_results['total_time_ms'] / max(0.01, crush_results['total_time_ms'])
    print(f"    Original:  {orig_results['avg_time_ms']:.2f} ms/step")
    print(f"    NovaCrush: {crush_results['avg_time_ms']:.2f} ms/step")
    print(f"    Ratio: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    print(f"\n  Transition layer sparsity: {crush_stats['transition_stats']['sparsity']:.1%}")

    # HDC stats
    hdc_stats = crush_stats['hdc_stats']
    print(f"\n  HDC Memory:")
    print(f"    Items stored: {hdc_stats['items_stored']}")
    print(f"    Total memory: {format_bytes(hdc_stats['total_bytes'])}")
    print(f"    Codebook size: {hdc_stats['codebook_size']} concepts")

    # Forward-Forward stats
    ff_stats = crushed.get_ff_stats()
    if ff_stats.get('accuracy'):
        print(f"\n  Forward-Forward Layer:")
        print(f"    Positive goodness: {ff_stats['pos_goodness']:.3f}")
        print(f"    Negative goodness: {ff_stats['neg_goodness']:.3f}")
        print(f"    Accuracy: {ff_stats['accuracy']:.1%}")

    print(f"\n  Neurogenesis events: {crush_stats['neurogenesis_events']}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)

    return {
        'original': orig_results,
        'novacrush': crush_results,
        'compression_stats': crush_stats,
    }


if __name__ == '__main__':
    results = run_benchmark(
        initial_concepts=32,
        embedding_dim=64,
        num_iterations=50
    )

"""
NovaCrush -- Integration Test: Genetic Compression + Spike Engine
===================================================================
Tests the DNA-inspired compression and event-driven computation together.
"""

import torch
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.novacrush.genetic_memory import Codon, Gene, Genome, GeneticOptimizer
from core.novacrush.spike_engine import SpikeLayer, SpikeDetector, StochasticLearner
from core.novacrush.crushed_substrate import CrushedSubstrate


def test_genetic_compression():
    """Test: Can a tiny genome encode a large weight matrix?"""
    print("=" * 60)
    print("  TEST 1: Genetic Compression (DNA -> Weights)")
    print("=" * 60)

    # Create a gene that encodes a 64x64 matrix with just 5 codons
    gene = Gene(
        name="transition_matrix",
        target_shape=(64, 64),
        codons=[
            Codon('zero', (0, 64, 0, 64), 0.0),        # Clear all
            Codon('identity', (0, 64, 0, 64), 0.8),     # Set diagonal
            Codon('noise', (0, 64, 0, 64), 0.02),       # Add noise
            Codon('set', (0, 16, 0, 16), 0.5),           # Strong top-left block
            Codon('mirror', (0, 64, 0, 64), 1.0),        # Symmetrize
        ]
    )

    # Express the gene
    tensor = gene.express()

    # Stats
    fp32_bytes = 64 * 64 * 4  # 16,384 bytes
    gene_bytes = gene.byte_size()  # 55 bytes

    print(f"  Gene: {len(gene.codons)} codons = {gene_bytes} bytes")
    print(f"  Expressed tensor: {tensor.shape} = {fp32_bytes:,} bytes (FP32)")
    print(f"  Compression ratio: {gene.compression_ratio():.0f}x")
    print(f"  Tensor mean: {tensor.mean():.4f}, std: {tensor.std():.4f}")
    print(f"  Non-zero elements: {(tensor != 0).sum().item()}/{64*64}")

    # Serialize the genome
    genome = Genome()
    genome.add_gene(gene)

    # Add an embedding gene
    embed_gene = Gene(
        name="embeddings",
        target_shape=(128, 64),
        codons=[
            Codon('noise', (0, 128, 0, 64), 0.02),
            Codon('scale', (0, 128, 0, 64), 1.5),
        ]
    )
    genome.add_gene(embed_gene)

    serialized = genome.serialize()
    stats = genome.get_stats()

    print(f"\n  Full genome:")
    print(f"    Genes: {stats['num_genes']}")
    print(f"    Total codons: {stats['total_codons']}")
    print(f"    Genome size: {stats['genome_bytes']} bytes")
    print(f"    FP32 equivalent: {stats['fp32_equivalent_bytes']:,} bytes")
    print(f"    Total compression: {stats['compression_ratio']:.0f}x")
    print(f"    Serialized (zlib): {len(serialized)} bytes")

    # Test deserialization
    restored = Genome.deserialize(serialized)
    restored_tensor = restored.genes['transition_matrix'].express()
    match = torch.allclose(tensor, restored_tensor, atol=1e-4)
    print(f"    Serialization fidelity: {'PASS' if match else 'FAIL'}")

    return stats


def test_evolutionary_compression():
    """Test: Can evolution discover a genome that reproduces target weights?"""
    print("\n" + "=" * 60)
    print("  TEST 2: Evolutionary Weight Compression")
    print("=" * 60)

    # Create a "target" weight matrix (what we want to compress)
    torch.manual_seed(42)
    target = {
        'layer1': torch.randn(32, 32) * 0.1,
        'layer2': torch.eye(16) * 0.5 + torch.randn(16, 16) * 0.01,
    }

    target_bytes = sum(t.numel() * 4 for t in target.values())
    print(f"  Target weights: {target_bytes:,} bytes (FP32)")

    # Run evolution
    optimizer = GeneticOptimizer(population_size=30, mutation_rate=0.2)
    best_genome = optimizer.compress_weights(target, generations=30)

    if best_genome:
        genome_stats = best_genome.get_stats()
        print(f"  Best genome: {genome_stats['genome_bytes']} bytes")
        print(f"  Compression: {genome_stats['compression_ratio']:.0f}x")
        print(f"  Fitness: {optimizer.best_fitness:.6f}")
        print(f"  Total codons: {genome_stats['total_codons']}")

        # Check reconstruction quality
        expressed = best_genome.express_all()
        for name, tgt in target.items():
            if name in expressed:
                mse = ((expressed[name] - tgt) ** 2).mean().item()
                print(f"  Reconstruction MSE ({name}): {mse:.6f}")
    else:
        print("  Evolution did not converge (expected for small population)")

    return optimizer.get_stats()


def test_spike_engine():
    """Test: Does spike-gating actually skip redundant computation?"""
    print("\n" + "=" * 60)
    print("  TEST 3: Spike-Gated Computation")
    print("=" * 60)

    # Create a simple linear layer wrapped in spike detection
    inner = torch.nn.Linear(64, 64)
    spike_layer = SpikeLayer(inner, spike_threshold=0.05, name="test_spike")

    # Simulate inputs: mostly similar (correlated in time)
    base_input = torch.randn(1, 64)
    total_time_spike = 0
    total_time_normal = 0

    for i in range(100):
        if i % 10 == 0:
            # Every 10th input: big change (should spike)
            x = torch.randn(1, 64)
        else:
            # Otherwise: small perturbation (should cache)
            x = base_input + torch.randn(1, 64) * 0.01

        t0 = time.perf_counter()
        _ = spike_layer(x)
        total_time_spike += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = inner(x)
        total_time_normal += time.perf_counter() - t0

    stats = spike_layer.get_stats()
    print(f"  Total checks: {stats['total_checks']}")
    print(f"  Spikes fired: {stats['spike_count']}")
    print(f"  Skip rate: {stats['skip_rate']:.1%}")
    print(f"  Effective speedup: {stats['effective_speedup']:.1f}x")
    print(f"  Time (spike): {total_time_spike*1000:.1f} ms")
    print(f"  Time (normal): {total_time_normal*1000:.1f} ms")

    return stats


def test_stochastic_learning():
    """Test: Does stochastic learning escape local minima?"""
    print("\n" + "=" * 60)
    print("  TEST 4: Stochastic Langevin Learning")
    print("=" * 60)

    learner = StochasticLearner(temperature=0.1, annealing_rate=0.99)

    # Simple 1D optimization: find minimum of f(x) = (x-3)^2 + sin(10x)
    # This has MANY local minima. Stochastic learning should escape them.
    x = torch.tensor([0.0], requires_grad=True)

    trajectory = []
    for step in range(200):
        loss = (x - 3.0) ** 2 + torch.sin(10 * x)
        loss.backward()
        with torch.no_grad():
            learner.update(x, x.grad, lr=0.05)
            x.grad.zero_()
        trajectory.append(x.item())

    stats = learner.get_stats()
    print(f"  Start: x = 0.000")
    print(f"  End:   x = {trajectory[-1]:.3f} (target: ~3.0)")
    print(f"  Final temperature: {stats['temperature']:.6f}")
    print(f"  Uncertainty: {stats['uncertainty']:.2%}")
    print(f"  Steps: {stats['step_count']}")

    close_to_target = abs(trajectory[-1] - 3.0) < 1.0
    print(f"  Reached near-global minimum: {'YES' if close_to_target else 'NO (local minimum)'}")

    return stats


def test_full_pipeline():
    """Test: Complete NovaCrush pipeline with all subsystems."""
    print("\n" + "=" * 60)
    print("  TEST 5: Full NovaCrush Pipeline")
    print("=" * 60)

    # Create CrushedSubstrate
    substrate = CrushedSubstrate(initial_concepts=16, embedding_dim=32, hdc_dim=2048)

    # Train on some data
    corpus = [
        ["the", "brain", "learns", "continuously"],
        ["artificial", "intelligence", "seeks", "understanding"],
        ["compression", "is", "intelligence"],
        ["genes", "encode", "programs", "not", "data"],
        ["spikes", "are", "efficient", "computation"],
    ]

    for tokens in corpus * 5:
        substrate.continuous_train(tokens)

    # Now compress the trained weights into a genome
    optimizer = GeneticOptimizer(population_size=15, mutation_rate=0.2)
    weights = {
        'embeddings': substrate.embeddings.data.clone(),
        'transition': substrate.transition.weight.data.clone(),
    }

    genome = optimizer.compress_weights(weights, generations=20)
    sub_stats = substrate.get_compression_stats()

    print(f"\n  Substrate after training:")
    print(f"    Vocab: {sub_stats['vocab_size']} concepts")
    print(f"    Train steps: {sub_stats['total_train_steps']}")
    print(f"    Transition sparsity: {sub_stats['transition_stats']['sparsity']:.1%}")

    if genome:
        g_stats = genome.get_stats()
        orig_bytes = sub_stats['original_fp32']['total_bytes']
        genome_bytes = g_stats['genome_bytes']
        serialized = genome.serialize()

        print(f"\n  Genetic compression results:")
        print(f"    Original FP32: {orig_bytes:,} bytes")
        print(f"    Genome (raw): {genome_bytes} bytes")
        print(f"    Genome (serialized+zlib): {len(serialized)} bytes")
        print(f"    Compression: {orig_bytes / max(1, len(serialized)):.0f}x")

    print(f"\n  HDC memory: {sub_stats['hdc_stats']['items_stored']} items")
    print(f"  Neurogenesis events: {sub_stats['neurogenesis_events']}")

    return sub_stats


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("  NovaCrush Integration Tests")
    print("#" * 60)

    test_genetic_compression()
    test_evolutionary_compression()
    test_spike_engine()
    test_stochastic_learning()
    test_full_pipeline()

    print("\n" + "#" * 60)
    print("  ALL TESTS COMPLETE")
    print("#" * 60)

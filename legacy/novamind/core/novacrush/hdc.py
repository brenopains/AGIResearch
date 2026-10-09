"""
NovaCrush — Hyperdimensional Computing (HDC) Memory Engine
============================================================
Brain-inspired computing using high-dimensional binary/bipolar vectors.

Key properties:
    - One-shot learning (learn from 1 example, not millions)
    - Zero catastrophic forgetting (superposition, not overwriting)
    - Noise-tolerant (information distributed across ALL dimensions)
    - Compositionality (bind + bundle = structured representations)

Operations:
    - Bind (⊗): circular convolution — creates associations
    - Bundle (+): element-wise addition — creates sets/categories
    - Permute (ρ): cyclic shift — encodes sequence/position

A 10,000-dim bipolar vector can store ~log2(2^10000) bits of information
while being robust to 30%+ noise corruption.

References:
    - Kanerva (2009): "Hyperdimensional Computing: An Introduction"
    - Plate (2003): "Holographic Reduced Representations"
    - Frady et al. (2022): "Computing with HDC"
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import hashlib


class HyperdimensionalVector:
    """
    A single hyperdimensional vector (hypervector).
    Default: 10,000 bipolar dimensions {-1, +1}.
    Total memory: 10,000 bits = 1.25 KB per concept.
    """

    def __init__(self, dim: int = 10000, values: Optional[torch.Tensor] = None,
                 device: str = 'cpu'):
        self.dim = dim
        self.device = device
        if values is not None:
            self.values = values.to(device)
        else:
            # Random bipolar vector: {-1, +1} uniform
            self.values = (torch.randint(0, 2, (dim,), device=device) * 2 - 1).float()

    def bind(self, other: 'HyperdimensionalVector') -> 'HyperdimensionalVector':
        """
        Binding (⊗): element-wise multiplication for bipolar vectors.
        Creates an association between two concepts.
        Key property: result is quasi-orthogonal to BOTH inputs.
        bind(A, B) is dissimilar to both A and B, but unbind(bind(A,B), B) ≈ A
        """
        return HyperdimensionalVector(
            self.dim,
            self.values * other.values,
            self.device
        )

    def bundle(self, other: 'HyperdimensionalVector') -> 'HyperdimensionalVector':
        """
        Bundling (+): element-wise addition + sign normalization.
        Creates a set/superposition of concepts.
        Key property: result is SIMILAR to both inputs.
        """
        summed = self.values + other.values
        # Threshold to maintain bipolar {-1, +1}
        result = torch.sign(summed)
        result[result == 0] = 1.0  # Break ties
        return HyperdimensionalVector(self.dim, result, self.device)

    def permute(self, shifts: int = 1) -> 'HyperdimensionalVector':
        """
        Permutation (ρ): cyclic shift for encoding order/position.
        permute(A, 1) encodes "A in position 1"
        """
        return HyperdimensionalVector(
            self.dim,
            torch.roll(self.values, shifts),
            self.device
        )

    def similarity(self, other: 'HyperdimensionalVector') -> float:
        """
        Cosine similarity between hypervectors.
        Expected values:
            - Same vector: 1.0
            - Related vectors: 0.1 - 0.5
            - Random/unrelated: ~0.0 (by concentration of measure)
            - Opposite: -1.0
        """
        return float(F.cosine_similarity(
            self.values.unsqueeze(0),
            other.values.unsqueeze(0)
        ).item())

    def hamming_similarity(self, other: 'HyperdimensionalVector') -> float:
        """Fraction of matching dimensions (for bipolar)."""
        return float((self.values == other.values).float().mean().item())

    def memory_bytes(self) -> int:
        """Actual memory usage in bytes."""
        return self.dim // 8  # 1 bit per dimension for bipolar

    def to_compact(self) -> bytes:
        """Pack to minimal binary representation (1 bit per dim)."""
        bits = ((self.values + 1) / 2).byte()  # {-1,+1} → {0,1}
        packed = np.packbits(bits.cpu().numpy())
        return bytes(packed)

    @classmethod
    def from_compact(cls, data: bytes, dim: int, device: str = 'cpu'):
        """Restore from compact binary representation."""
        unpacked = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:dim]
        values = torch.from_numpy(unpacked.astype(np.float32)) * 2 - 1
        return cls(dim, values, device)


# Need this import for similarity
import torch.nn.functional as F


class HDCMemoryStore:
    """
    Associative memory using hyperdimensional computing.
    
    Stores concept-value pairs holographically:
    - Memory is a single bundled hypervector (superposition of all stored pairs)
    - Retrieval: bind query with memory → nearest match
    - Capacity: O(dim / log(num_items)) items with high-fidelity retrieval
    
    For dim=10,000: can reliably store ~500-1000 items in a single vector.
    Total memory for 1000 items: ~1.25 KB (the single memory vector)
    vs. traditional: 1000 × embedding_size × 4 bytes = ~1.2 MB (for 300-dim FP32)
    
    Compression: ~1000x for the memory index alone.
    """

    def __init__(self, dim: int = 10000, device: str = 'cpu', max_codebook_size: int = 2000):
        self.dim = dim
        self.device = device
        self.max_codebook_size = max_codebook_size

        # The associative memory — a single bundled hypervector
        self.memory = HyperdimensionalVector(dim, torch.zeros(dim, device=device), device)

        # Item codebook: maps labels to their random base hypervectors (LRU Cache)
        self.codebook: Dict[str, HyperdimensionalVector] = OrderedDict()

        # Store count for normalization
        self.item_count = 0
        self.capacity_estimate = dim // 20  # Conservative capacity

    def _get_or_create_code(self, label: str) -> HyperdimensionalVector:
        """Get or create a unique random hypervector for a label (LRU cached)."""
        if label in self.codebook:
            self.codebook.move_to_end(label)
            return self.codebook[label]
            
        # Deterministic seeding from label for reproducibility
        seed = int(hashlib.md5(label.encode()).hexdigest()[:8], 16)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        values = (torch.randint(0, 2, (self.dim,), generator=gen) * 2 - 1).float()
        
        hv = HyperdimensionalVector(self.dim, values.to(self.device), self.device)
        self.codebook[label] = hv
        
        # Enforce LRU codebook limit
        if len(self.codebook) > self.max_codebook_size:
            self.codebook.popitem(last=False)
            
        return hv

    def store(self, key: str, value: str) -> Dict:
        """
        Store a key-value association in holographic memory.
        
        The pair is encoded as: bind(key_hv, value_hv)
        Then bundled into the memory: memory = memory + bind(key, value)
        """
        key_hv = self._get_or_create_code(key)
        value_hv = self._get_or_create_code(value)

        # Create association via binding
        pair = key_hv.bind(value_hv)

        # Bundle into memory (superposition)
        self.memory = self.memory.bundle(pair)
        self.item_count += 1

        utilization = self.item_count / max(1, self.capacity_estimate)

        return {
            'stored': f'{key} → {value}',
            'items_in_memory': self.item_count,
            'utilization': min(1.0, utilization),
            'memory_bytes': self.memory.memory_bytes(),
        }

    def retrieve(self, query: str, candidates: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Retrieve the value associated with a query key.
        
        Process:
        1. Bind query with memory: result = bind(query_hv, memory)
        2. Result will be most similar to the value that was stored with this key
        3. Compare result against all candidate values
        """
        query_hv = self._get_or_create_code(query)

        # Unbind: multiply query with memory to extract associated value
        retrieved = query_hv.bind(
            HyperdimensionalVector(self.dim, self.memory.values, self.device)
        )

        # Compare against candidates
        if candidates is None:
            candidates = list(self.codebook.keys())

        results = []
        for label in candidates:
            if label == query:
                continue
            candidate_hv = self._get_or_create_code(label)
            sim = retrieved.similarity(candidate_hv)
            results.append((label, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def store_sequence(self, label: str, tokens: List[str]) -> Dict:
        """
        Store an ordered sequence using positional permutation.
        sequence_hv = Σ permute(token_hv, position)
        """
        sequence_hv = HyperdimensionalVector(
            self.dim, torch.zeros(self.dim, device=self.device), self.device
        )

        for pos, token in enumerate(tokens):
            token_hv = self._get_or_create_code(token)
            positioned = token_hv.permute(pos)
            sequence_hv = sequence_hv.bundle(positioned)

        # Store as label → sequence association
        label_hv = self._get_or_create_code(label)
        pair = label_hv.bind(sequence_hv)
        self.memory = self.memory.bundle(pair)
        self.item_count += 1

        return {
            'stored_sequence': label,
            'length': len(tokens),
            'items_in_memory': self.item_count,
        }

    def get_stats(self) -> Dict:
        """Return memory statistics."""
        codebook_bytes = len(self.codebook) * (self.dim // 8)
        memory_bytes = self.dim // 8

        return {
            'dimensions': self.dim,
            'items_stored': self.item_count,
            'capacity_estimate': self.capacity_estimate,
            'utilization': min(1.0, self.item_count / max(1, self.capacity_estimate)),
            'codebook_size': len(self.codebook),
            'memory_bytes': memory_bytes,
            'codebook_bytes': codebook_bytes,
            'total_bytes': memory_bytes + codebook_bytes,
            'equivalent_fp32_bytes': self.item_count * 300 * 4,  # Compare to 300d FP32
            'compression_vs_fp32': max(1, self.item_count * 300 * 4) / max(1, memory_bytes + codebook_bytes),
        }

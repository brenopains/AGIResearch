"""
NovaCrush -- Genetic Memory: DNA/RNA-Inspired Knowledge Compression
======================================================================

The most radical compression the universe ever produced:

    Human DNA: 3.2 billion base pairs = ~750 MB of data
    Human Brain: ~2.5 petabytes of synaptic information
    Compression ratio: ~3,333,333:1

HOW? DNA doesn't store DATA. It stores GENERATIVE PROGRAMS.
The genome is a set of rules that BUILD the brain, not the brain itself.

This module implements the same principle for NovaMind:
    - Instead of storing weights (data), store RULES that generate weights
    - Instead of 100GB of parameters, store the generating program
    - This IS Kolmogorov complexity made practical

Architecture inspired by molecular biology:
    CODONS  = atomic instructions (3-element tuples like DNA codons)
    GENES   = functional units (sequences of codons that build one component)
    GENOME  = complete blueprint (all genes needed to reconstruct the mind)
    
    EXPRESSION = gene -> protein -> function (codon -> weights -> computation)
    MUTATION   = stochastic perturbation (exploration/creativity)
    SELECTION  = keep what reduces surprise (learning)
    
Key insight: Evolution found that PROGRAMS compress better than DATA.
    A 750MB genome generates a 2.5PB brain.
    A 10KB NovaMind genome could generate a 10MB neural substrate.
    That IS your "100GB -> KBs" dream -- not of raw data, but of
    the PROGRAM that generates and regenerates the knowledge.

References:
    - Kolmogorov (1965): "Three Approaches to the Definition of Information"
    - Solomonoff (1964): "A Formal Theory of Inductive Inference"
    - Levin (1973): "Universal Optimal Search"
    - Schmidhuber (1997): "Discovering Neural Nets with Low Kolmogorov Complexity"
    - Stanley & Miikkulainen (2002): "NEAT: Evolving Neural Network Topologies"
"""

import torch
import torch.nn as nn
import numpy as np
import zlib
import json
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time
import copy


# =============================================================================
#  MOLECULAR PRIMITIVES
# =============================================================================

@dataclass
class Codon:
    """
    Atomic instruction -- analogous to a DNA codon (3 nucleotides -> 1 amino acid).
    
    A codon encodes ONE operation that modifies a weight tensor:
        opcode:  what to do (set, add, scale, rotate, mirror, zero, noise)
        target:  which tensor region to modify (row, col, block indices)
        value:   the parameter of the operation
        
    3 fields x average 8 bits each = ~24 bits per codon = 3 bytes.
    Compare: one FP32 weight = 32 bits = 4 bytes.
    But a single codon can SET an entire row/block of weights.
    """
    opcode: str      # Operation: set, add, scale, rotate, mirror, zero, noise, repeat
    target: tuple    # (start_row, end_row, start_col, end_col) or special indices
    value: float     # Parameter for the operation
    
    def execute(self, tensor: torch.Tensor) -> torch.Tensor:
        """Execute this codon on a weight tensor."""
        r0, r1, c0, c1 = self.target
        r0, r1 = max(0, r0), min(tensor.shape[0], r1)
        c0, c1 = max(0, c0), min(tensor.shape[1], c1)
        
        if r0 >= r1 or c0 >= c1:
            return tensor
            
        with torch.no_grad():
            if self.opcode == 'set':
                tensor[r0:r1, c0:c1] = self.value
            elif self.opcode == 'add':
                tensor[r0:r1, c0:c1] += self.value
            elif self.opcode == 'scale':
                tensor[r0:r1, c0:c1] *= self.value
            elif self.opcode == 'zero':
                tensor[r0:r1, c0:c1] = 0.0
            elif self.opcode == 'noise':
                noise = torch.randn_like(tensor[r0:r1, c0:c1]) * abs(self.value)
                tensor[r0:r1, c0:c1] += noise
            elif self.opcode == 'mirror':
                # Copy upper triangle to lower (symmetry)
                block = tensor[r0:r1, c0:c1]
                if block.shape[0] == block.shape[1]:
                    tensor[r0:r1, c0:c1] = (block + block.T) / 2
            elif self.opcode == 'identity':
                # Set to scaled identity
                block = tensor[r0:r1, c0:c1]
                eye = torch.eye(min(block.shape), device=tensor.device)
                if block.shape[0] != block.shape[1]:
                    eye = eye[:block.shape[0], :block.shape[1]]
                tensor[r0:r1, c0:c1] = eye * self.value
            elif self.opcode == 'repeat':
                # Repeat first row across all rows (pattern compression)
                if r1 > r0 + 1:
                    pattern = tensor[r0:r0+1, c0:c1].clone()
                    tensor[r0:r1, c0:c1] = pattern.expand(r1 - r0, -1) * self.value
        return tensor
    
    def to_bytes(self) -> bytes:
        """Serialize codon to minimal bytes."""
        opcode_map = {'set': 0, 'add': 1, 'scale': 2, 'zero': 3, 
                      'noise': 4, 'mirror': 5, 'identity': 6, 'repeat': 7}
        op_byte = opcode_map.get(self.opcode, 0)
        # Pack: 1 byte opcode + 4x2 bytes target + 2 bytes value (FP16)
        target_bytes = b''.join(t.to_bytes(2, 'little', signed=True) 
                                for t in self.target)
        val_half = np.float16(self.value).tobytes()
        return bytes([op_byte]) + target_bytes + val_half
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Codon':
        """Deserialize from bytes."""
        opcode_map = {0: 'set', 1: 'add', 2: 'scale', 3: 'zero',
                      4: 'noise', 5: 'mirror', 6: 'identity', 7: 'repeat'}
        op = opcode_map.get(data[0], 'set')
        target = tuple(int.from_bytes(data[1+i*2:3+i*2], 'little', signed=True) 
                       for i in range(4))
        val = float(np.frombuffer(data[9:11], dtype=np.float16)[0])
        return cls(op, target, val)
    
    def byte_size(self) -> int:
        return 11  # 1 + 8 + 2 bytes


@dataclass
class Gene:
    """
    Functional unit -- a sequence of codons that builds one component.
    
    Analogous to a biological gene that encodes one protein.
    A gene might encode:
        - An entire embedding matrix (via a few codons)
        - A transition rule set
        - A connectivity pattern
        
    Example: A 64x64 identity-like transition matrix
        Gene with 3 codons:
            1. zero(all)       -> clear matrix  
            2. identity(diag)  -> set diagonal
            3. noise(all, 0.01) -> add small perturbation
        Total: 33 bytes to encode a 64x64 matrix (16,384 bytes in FP32)
        Compression: 496x
    """
    name: str
    codons: List[Codon] = field(default_factory=list)
    target_shape: Tuple[int, int] = (64, 64)
    fitness: float = 0.0
    generation: int = 0
    mutations: int = 0
    
    def express(self, device: str = 'cpu') -> torch.Tensor:
        """
        Gene expression: execute all codons to build a weight tensor.
        This is the genotype -> phenotype transformation.
        """
        tensor = torch.zeros(*self.target_shape, device=device)
        for codon in self.codons:
            tensor = codon.execute(tensor)
        return tensor
    
    def mutate(self, rate: float = 0.1) -> 'Gene':
        """
        Mutation: stochastic perturbation of the gene.
        Like biological point mutations, insertions, deletions.
        """
        mutated = Gene(
            name=self.name,
            codons=[copy.deepcopy(c) for c in self.codons],
            target_shape=self.target_shape,
            fitness=self.fitness,
            generation=self.generation + 1,
            mutations=self.mutations
        )
        
        for codon in mutated.codons:
            if np.random.random() < rate:
                # Mutate value
                codon.value += np.random.randn() * 0.1
                mutated.mutations += 1
                
        # Insertion mutation (add new codon)
        if np.random.random() < rate * 0.3:
            opcodes = ['set', 'add', 'scale', 'noise', 'repeat']
            new_codon = Codon(
                opcode=np.random.choice(opcodes),
                target=(
                    np.random.randint(0, self.target_shape[0]),
                    np.random.randint(1, self.target_shape[0] + 1),
                    np.random.randint(0, self.target_shape[1]),
                    np.random.randint(1, self.target_shape[1] + 1),
                ),
                value=np.random.randn() * 0.1
            )
            pos = np.random.randint(0, len(mutated.codons) + 1)
            mutated.codons.insert(pos, new_codon)
            mutated.mutations += 1
            
        # Deletion mutation (remove a codon)
        if np.random.random() < rate * 0.1 and len(mutated.codons) > 2:
            idx = np.random.randint(0, len(mutated.codons))
            mutated.codons.pop(idx)
            mutated.mutations += 1
            
        return mutated
    
    def byte_size(self) -> int:
        """Total size in bytes."""
        return sum(c.byte_size() for c in self.codons)
    
    def compression_ratio(self) -> float:
        """Compression vs storing the full tensor in FP32."""
        fp32_bytes = self.target_shape[0] * self.target_shape[1] * 4
        gene_bytes = max(1, self.byte_size())
        return fp32_bytes / gene_bytes


@dataclass
class Genome:
    """
    Complete blueprint -- all genes needed to reconstruct the entire mind.
    
    The genome is the "seed" from which the neural substrate can be
    fully regenerated. Like DNA, it's tiny compared to what it builds.
    
    A NovaMind genome of ~10-50 KB could encode a neural substrate
    that would be 1-10 MB in raw FP32 weights.
    """
    genes: Dict[str, Gene] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    generation: int = 0
    
    def add_gene(self, gene: Gene):
        self.genes[gene.name] = gene
        
    def express_all(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Express all genes into weight tensors."""
        return {name: gene.express(device) for name, gene in self.genes.items()}
    
    def total_bytes(self) -> int:
        return sum(g.byte_size() for g in self.genes.values())
    
    def total_fp32_equivalent(self) -> int:
        return sum(g.target_shape[0] * g.target_shape[1] * 4 
                   for g in self.genes.values())
    
    def compression_ratio(self) -> float:
        return self.total_fp32_equivalent() / max(1, self.total_bytes())
    
    def serialize(self) -> bytes:
        """Serialize entire genome to bytes (the ultimate compressed form)."""
        data = {
            'generation': self.generation,
            'genes': {}
        }
        for name, gene in self.genes.items():
            data['genes'][name] = {
                'shape': gene.target_shape,
                'codons': [c.to_bytes().hex() for c in gene.codons],
                'fitness': gene.fitness,
            }
        raw = json.dumps(data).encode('utf-8')
        return zlib.compress(raw, 9)  # Further compress the genome itself
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'Genome':
        """Reconstruct genome from bytes."""
        raw = json.loads(zlib.decompress(data).decode('utf-8'))
        genome = cls(generation=raw['generation'])
        for name, gdata in raw['genes'].items():
            gene = Gene(
                name=name,
                target_shape=tuple(gdata['shape']),
                fitness=gdata['fitness'],
                codons=[Codon.from_bytes(bytes.fromhex(h)) for h in gdata['codons']]
            )
            genome.add_gene(gene)
        return genome
    
    def get_stats(self) -> Dict:
        return {
            'num_genes': len(self.genes),
            'total_codons': sum(len(g.codons) for g in self.genes.values()),
            'genome_bytes': self.total_bytes(),
            'fp32_equivalent_bytes': self.total_fp32_equivalent(),
            'compression_ratio': self.compression_ratio(),
            'generation': self.generation,
            'per_gene': {
                name: {
                    'codons': len(g.codons),
                    'bytes': g.byte_size(),
                    'shape': g.target_shape,
                    'compression': g.compression_ratio(),
                    'fitness': g.fitness,
                }
                for name, g in self.genes.items()
            }
        }


# =============================================================================
#  EVOLUTIONARY ENGINE
# =============================================================================

class GeneticOptimizer:
    """
    Evolutionary optimizer that discovers minimal genetic programs
    to encode neural weight matrices.
    
    Process:
    1. Start with random genome
    2. Express genome -> weight tensors  
    3. Evaluate fitness (how well the model performs)
    4. Mutate best genomes
    5. Select fittest
    6. Repeat
    
    This finds the SHORTEST PROGRAM that generates good weights --
    the practical realization of Kolmogorov complexity minimization.
    """
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.15):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_genome: Optional[Genome] = None
        self.fitness_history: List[float] = []
        
    def create_initial_genome(self, layer_specs: Dict[str, Tuple[int, int]]) -> Genome:
        """
        Create an initial genome with basic codons for each layer.
        
        layer_specs: {'embeddings': (vocab, dim), 'transition': (dim, dim)}
        """
        genome = Genome()
        
        for name, shape in layer_specs.items():
            # Start with a simple but reasonable initialization
            codons = [
                # Base: small random noise everywhere
                Codon('noise', (0, shape[0], 0, shape[1]), 0.02),
            ]
            
            # For square matrices, add identity-like structure
            if shape[0] == shape[1]:
                codons.append(
                    Codon('identity', (0, shape[0], 0, shape[1]), 0.5)
                )
            
            gene = Gene(name=name, codons=codons, target_shape=shape)
            genome.add_gene(gene)
            
        return genome
    
    def evolve_step(self, population: List[Genome], 
                    fitness_fn) -> List[Genome]:
        """
        One generation of evolution.
        
        1. Evaluate fitness
        2. Select top performers
        3. Mutate to create next generation
        """
        self.generation += 1
        
        # Evaluate
        scored = []
        for genome in population:
            fitness = fitness_fn(genome)
            genome.generation = self.generation
            scored.append((fitness, genome))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Track best
        if scored[0][0] > self.best_fitness:
            self.best_fitness = scored[0][0]
            self.best_genome = copy.deepcopy(scored[0][1])
        self.fitness_history.append(scored[0][0])
        
        # Select top 30% as parents
        n_parents = max(2, self.population_size // 3)
        parents = [g for _, g in scored[:n_parents]]
        
        # Create next generation through mutation
        next_gen = [copy.deepcopy(parents[0])]  # Elitism: keep best unchanged
        
        while len(next_gen) < self.population_size:
            parent = parents[np.random.randint(0, len(parents))]
            child = Genome(generation=self.generation)
            for name, gene in parent.genes.items():
                child.add_gene(gene.mutate(self.mutation_rate))
            next_gen.append(child)
            
        return next_gen
    
    def compress_weights(self, target_tensors: Dict[str, torch.Tensor],
                         generations: int = 50) -> Genome:
        """
        Given target weight tensors, find the minimal genome that reproduces them.
        
        This is reverse-engineering: going from phenotype (weights) back to
        genotype (minimal program). The key compression step.
        """
        # Create layer specs from targets
        specs = {name: tuple(t.shape) for name, t in target_tensors.items()}
        
        # Initialize population
        population = [self.create_initial_genome(specs) 
                      for _ in range(self.population_size)]
        
        def fitness(genome: Genome) -> float:
            expressed = genome.express_all()
            total_error = 0.0
            for name, target in target_tensors.items():
                if name in expressed:
                    pred = expressed[name]
                    if pred.shape == target.shape:
                        # Negative MSE (higher = better fit)
                        mse = ((pred - target.cpu()) ** 2).mean().item()
                        total_error -= mse
                    else:
                        total_error -= 100.0  # Shape mismatch penalty
                        
            # Bonus for shorter genomes (Occam's razor / MDL)
            genome_len = genome.total_bytes()
            brevity_bonus = 1.0 / (1.0 + genome_len * 0.001)
            
            return total_error + brevity_bonus * 0.1
        
        # Evolve
        for gen in range(generations):
            population = self.evolve_step(population, fitness)
            
        return self.best_genome
    
    def get_stats(self) -> Dict:
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'fitness_history': self.fitness_history[-50:],
            'best_genome_stats': self.best_genome.get_stats() if self.best_genome else None,
        }

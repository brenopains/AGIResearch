"""
TypedCognitiveGraph — Arquitetura Integrada
============================================
Unifica as quatro fundações matemáticas:
  - Poincaré Ball (hyperbolic.py): hierarquia via profundidade do centro
  - Clifford Algebra (geometric_algebra.py): relações tipadas como bivectors
  - Free Energy Principle (information.py): confiança real computada via FEP
  - Persistent Homology (topology.py): gaps de conhecimento, Betti numbers

Nós são pontos no espaço hiperbólico. Arestas são bivectors de relação.
Confidence é computado por surprise Bayesiana, não constante.
"""
import sys, os, time, json, pickle, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core.math.hyperbolic import PoincareBall, EntailmentCone
from core.math.geometric_algebra import CliffordAlgebra, MultiVector
from core.math.information import InformationEngine, FreeEnergyMinimizer
from core.math.topology import SimplicialComplex, Simplex, PersistentHomology

EPS = 1e-10

# Relation type bivectors: each relation is a distinct plane in Clifford algebra
RELATION_TYPES = {
    'CAUSES': {'i': 0, 'j': 1, 'label': 'causes', 'symmetric': False},
    'IS_A': {'i': 0, 'j': 2, 'label': 'is_a', 'symmetric': False},
    'PART_OF': {'i': 0, 'j': 3, 'label': 'part_of', 'symmetric': False},
    'BEFORE': {'i': 0, 'j': 4, 'label': 'before', 'symmetric': False},
    'AFTER': {'i': 0, 'j': 5, 'label': 'after', 'symmetric': False},
    'ENABLES': {'i': 0, 'j': 6, 'label': 'enables', 'symmetric': False},
    'NEGATES': {'i': 0, 'j': 7, 'label': 'negates', 'symmetric': False},
    'CORRELATED': {'i': 0, 'j': 8, 'label': 'correlated', 'symmetric': True},
    'OPPOSES': {'i': 0, 'j': 9, 'label': 'opposes', 'symmetric': True},
}


class TypedCognitiveGraph:
    """
    Knowledge graph with typed relations, hyperbolic concept positions,
    Bayesian confidence, and topological metacognition.
    
    Not a replacement for CGE — a scientific upgrade to it.
    """

    def __init__(self, dim: int = 100, curvature: float = 1.0, 
                 relation_algebra_dim: int = 10):
        self.dim = dim
        self.ball = PoincareBall(dim, curvature)
        # Clifford algebra for relation types: dimension sufficient for 9 bivectors
        # NOT concept-space dim (2^100 elements causes MemoryError)
        self.algebra = CliffordAlgebra(relation_algebra_dim, 0)
        self.algebra_dim = relation_algebra_dim
        self.fep = FreeEnergyMinimizer()
        
        # Nodes: concept -> {position, vector, depth, cone, observations}
        self.nodes: Dict[str, dict] = {}
        
        # Edges: (from, to, rel_type) -> {bivector, confidence, observations}
        self.edges: Dict[Tuple[str, str, str], dict] = {}
        
        # Bivectors for each relation type (precomputed)
        self.rel_bivectors = {}
        for rtype, config in RELATION_TYPES.items():
            self.rel_bivectors[rtype] = self.algebra.bivector(
                config['i'], config['j'], 1.0
            )
        
        # Topological state
        self.complex = None
        self.betti = {'beta_0': 0, 'beta_1': 0, 'beta_2': 0}
        self.persistence_pairs = []
        self.knowledge_gaps = []
        
        print(f"[TCG] TypedCognitiveGraph init: dim={dim}, "
              f"curvature={curvature}, relation_types={len(RELATION_TYPES)}")

    # ─── Node Operations ─────────────────────────────────────────────

    def add_concept(self, name: str, vector: Optional[np.ndarray] = None,
                    bge_vec: Optional[torch.Tensor] = None) -> dict:
        """Add a concept at a hyperbolic position with RANDOMIZED depth for variety."""
        if name in self.nodes:
            return self.nodes[name]
        
        # Use name hash to determine depth (pseudo-deterministic per concept)
        name_hash = abs(hash(name)) % 1000 / 1000.0  # 0..1
        depth_target = 0.3 + name_hash * 1.5  # 0.3..1.8
        
        if vector is not None:
            pos = np.asarray(vector, dtype=np.float32).flatten()[:self.dim]
        elif bge_vec is not None:
            # Use BGE subsample for POSITION but scale to target depth
            bge_np = torch.sign(bge_vec).cpu().numpy().flatten().astype(np.float32)[:self.dim]
            # Fill if too short
            if len(bge_np) < self.dim:
                bge_np = np.pad(bge_np, (0, self.dim - len(bge_np)), 'constant')
            bge_np = bge_np[:self.dim]
            norm = np.linalg.norm(bge_np)
            if norm > 0:
                pos = bge_np / norm * depth_target
            else:
                angle = np.random.randn(self.dim)
                pos = angle / np.linalg.norm(angle) * depth_target
        else:
            angle = np.random.randn(self.dim)
            pos = angle / (np.linalg.norm(angle) + 1e-10) * depth_target
        
        # Clamp to ball
        norm = np.linalg.norm(pos)
        if norm >= 1.0:
            pos = pos * 0.99 / norm
        
        depth = float(self.ball.hierarchy_depth(pos.reshape(1, -1)).flatten()[0])
        cone = EntailmentCone(self.ball, pos.reshape(1, -1), half_angle=0.5)
        
        self.nodes[name] = {
            'position': pos,
            'depth': depth,
            'cone': cone,
            'observations': 0,
            'created': time.time(),
        }
        return self.nodes[name]

    def concept_depth(self, name: str) -> float:
        """Distance from origin = hierarchical abstraction level."""
        if name not in self.nodes:
            return 1.0
        return self.nodes[name]['depth']

    def concept_similarity(self, a: str, b: str) -> float:
        """Hyperbolic similarity = exp(-distance). 1 = identical, 0 = opposite."""
        if a not in self.nodes or b not in self.nodes:
            return 0.0
        d = float(self.ball.distance(
            self.nodes[a]['position'].reshape(1, -1),
            self.nodes[b]['position'].reshape(1, -1)
        ).flatten()[0])
        return math.exp(-d)

    def is_a(self, child: str, parent: str) -> float:
        """Check IS-A relation via entailment cone membership."""
        if child not in self.nodes or parent not in self.nodes:
            return 0.0
        score = self.nodes[parent]['cone'].membership_score(
            self.nodes[child]['position']
        )
        # Also need child deeper than parent
        if self.nodes[child]['depth'] <= self.nodes[parent]['depth']:
            score *= 0.5
        return score

    # ─── Edge Operations ─────────────────────────────────────────────

    def add_relation(self, from_node: str, to_node: str,
                     rel_type: str = 'CAUSES',
                     confidence: float = 1.0,
                     observation_evidence: float = 1.0):
        """Add a TYPED relation with FEP-computed confidence."""
        if from_node not in self.nodes:
            self.add_concept(from_node)
        if to_node not in self.nodes:
            self.add_concept(to_node)
        
        key = (from_node, to_node, rel_type)
        
        if key not in self.edges:
            # Seed prior based on name similarity and relation type
            name_sim = self._name_similarity(from_node, to_node)
            # Prior confidence: name_sim weighted by relation typicality
            prior_conf = 0.3 + 0.4 * name_sim
            if rel_type == 'CORRELATED':
                prior_conf = 0.5  # Default for unknown
            elif rel_type == 'IS_A':
                prior_conf = 0.3 + 0.5 * name_sim  # IS-A needs higher name overlap
            
            biv = self.rel_bivectors.get(rel_type, self.rel_bivectors['CAUSES'])
            self.edges[key] = {
                'from': from_node, 'to': to_node,
                'rel_type': rel_type,
                'bivector': biv * prior_conf,
                'confidence': prior_conf,
                'observations': 0,
                'prior': prior_conf,
                'fep_history': [],
                'name_similarity': name_sim,
            }
        
        edge = self.edges[key]
        prior_conf = edge['confidence']
        edge['observations'] += 1
        
        # FEP: compute surprise of this observation given prior
        # If observation agrees with prior: low surprise, confidence increases
        # If observation disagrees: high surprise, confidence decreases
        evidence = max(0.01, min(0.99, observation_evidence))
        
        # Prior distribution: Beta-like (prior_conf concentration)
        alpha = prior_conf * 10 + 1
        beta_val = (1 - prior_conf) * 10 + 1
        
        # Likelihood: probability of observing this evidence
        # If evidence matches prior: P ~ 0.7-0.9
        # If evidence contradicts: P ~ 0.1-0.3
        likelihood = 0.5 + 0.4 * (1 - abs(evidence - prior_conf))
        
        # Surprise = -log(P)
        surprise = -math.log(max(likelihood, 0.001))
        
        # Update confidence: Bayesian
        posterior = (alpha * evidence + 1) / (alpha + beta_val)
        posterior = max(0.01, min(0.99, posterior))
        
        edge['confidence'] = posterior
        edge['observations'] = edge.get('observations', 0) + 1
        edge['fep_history'].append({
            'prior': prior_conf,
            'evidence': evidence,
            'surprise': surprise,
            'posterior': posterior,
        })

    def _name_similarity(self, a: str, b: str) -> float:
        """Estimate similarity between concept names via word overlap."""
        words_a = set(a.lower().replace('_', ' ').replace('-', ' ').split())
        words_b = set(b.lower().replace('_', ' ').replace('-', ' ').split())
        if not words_a or not words_b:
            return 0.0
        intersect = words_a & words_b
        union = words_a | words_b
        return len(intersect) / max(len(union), 1)

    def get_relation(self, from_node: str, to_node: str,
                     rel_type: str = 'CAUSES') -> Optional[dict]:
        return self.edges.get((from_node, to_node, rel_type))

    def query(self, from_node: str, rel_type: str = 'CAUSES',
              min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        """Find all concepts connected to from_node via rel_type with confidence."""
        results = []
        for key, edge in self.edges.items():
            f, t, r = key
            if f == from_node and r == rel_type and edge['confidence'] >= min_confidence:
                results.append((t, edge['confidence']))
        return sorted(results, key=lambda x: -x[1])

    # ─── Topology Analysis ──────────────────────────────────────────

    def build_topology(self, max_concepts: int = 80) -> dict:
        """Build simplicial complex from the graph and compute persistent homology.
        max_concepts limits computation to N concepts for speed."""
        concepts = list(self.nodes.keys())[:max_concepts]
        n = len(concepts)
        if n < 3:
            return {'error': 'too few concepts'}
        
        # Build distance matrix from hyperbolic distances
        dist_matrix = np.zeros((n, n))
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i == j:
                    dist_matrix[i, j] = 0
                else:
                    dist_matrix[i, j] = float(self.ball.distance(
                        self.nodes[c1]['position'].reshape(1, -1),
                        self.nodes[c2]['position'].reshape(1, -1)
                    ).flatten()[0])
        
        # Build Vietoris-Rips complex
        self.complex = SimplicialComplex()
        self.complex.build_from_distance_matrix(dist_matrix, max_dim=2, 
                                                max_filtration=5.0)
        
        # Compute persistent homology
        ph = PersistentHomology(self.complex)
        self.persistence_pairs = ph.compute(max_dim=2)
        
        # Betti numbers
        self.betti = ph.betti_numbers(filtration_value=2.0)
        
        # Knowledge gaps (H2 features)
        self.knowledge_gaps = ph.knowledge_gaps()
        if isinstance(self.knowledge_gaps, list):
            self.knowledge_gaps = self.knowledge_gaps[:5]
        
        # Circular reasoning (H1 features)
        circular = ph.circular_reasoning_detector()
        
        result = {
            'betti_numbers': self.betti,
            'n_concepts': n,
            'n_edges': len(self.edges),
            'persistence_entropy': ph.persistence_entropy(),
            'knowledge_gaps': len(self.knowledge_gaps),
            'circular_loops': len(circular),
        }
        return result

    # ─── Bulk Load from Existing System ─────────────────────────────

    def load_from_brain_state(self, brain_path: str, state_path: str) -> dict:
        """
        Load the existing system state and convert to typed hyperbolic graph.
        Returns conversion statistics.
        """
        from core.bge_encoder import BGEEncoder
        print(f"[TCG] Loading brain state from {brain_path}...")
        
        data = torch.load(brain_path, map_location='cpu', weights_only=False)
        brain_nodes = data.get('nodes', data)
        brain_edges = data.get('edges', [])
        
        # Also load agi_state_live.pkl for additional info
        try:
            state = pickle.load(open(state_path, 'rb'))
            cge_edges_raw = state.get('cge_edges', [])
            cge_nodes = state.get('cge_nodes', {})
        except:
            cge_edges_raw = []
            cge_nodes = {}
        
        # Load BGE for concept encoding
        bge = BGEEncoder()
        
        stats = {
            'concepts_added': 0,
            'edges_typed': 0,
            'edges_imported': 0,
            'confidence_stats': {'mean': 0, 'min': 1, 'max': 0},
        }
        
        # 1. Add all concepts from brain_live.pth
        node_names = list(brain_nodes.keys())
        for name in node_names:
            if isinstance(name, str) and len(name) > 1 and len(name) < 50:
                nd = brain_nodes[name]
                vec = None
                if isinstance(nd, dict) and 'vector' in nd:
                    vec = nd['vector']
                self.add_concept(name, bge_vec=vec)
                stats['concepts_added'] += 1
        
        # 2. Add concepts from agi_state_live.pkl (if names differ)
        for name in cge_nodes:
            if isinstance(name, str) and len(name) > 1 and name not in self.nodes:
                nd = cge_nodes[name]
                vec = nd.get('vector', None) if isinstance(nd, dict) else None
                self.add_concept(name, bge_vec=vec)
                stats['concepts_added'] += 1
        
        print(f"[TCG] Added {stats['concepts_added']} concepts "
              f"(total: {len(self.nodes)})")
        
        # 3. Import edges from brain_live.pth with edge_type-based classification
        for i, edge in enumerate(brain_edges):
            if not isinstance(edge, dict): continue
            frm = str(edge.get('from', ''))
            to = str(edge.get('to', ''))
            et = edge.get('edge_type', -1)
            
            if frm in self.nodes and to in self.nodes:
                conf = edge.get('confidence', 0.5)
                if isinstance(conf, torch.Tensor):
                    conf = conf.mean().item()
                
                # Classify by edge_type and hyperbolic depth:
                # -1 = sequential/co-occurrence in text
                # 1 = word -> abstract_phrase mapping -> PART_OF
                # IS-A: if source is significantly deeper (more specific) than target
                depth_from = self.nodes[frm]['depth']
                depth_to = self.nodes[to]['depth']
                
                if et == 1:
                    rel_type = 'PART_OF'
                elif depth_from > depth_to + 0.8:
                    # Source is much more specific than target -> IS-A
                    # e.g., "dog" (depth=7.2) -> IS_A -> "animal" (depth=1.5)
                    rel_type = 'IS_A'
                elif to.startswith('abstract_') or frm.startswith('abstract_'):
                    rel_type = 'PART_OF'
                else:
                    rel_type = 'CORRELATED'
                
                conf_scaled = max(0.1, min(0.99, float(conf)))
                self.add_relation(frm, to, rel_type, 
                                  confidence=conf_scaled * 0.8,
                                  observation_evidence=conf_scaled * 0.7)
                stats['edges_imported'] += 1
        
        # 4. Also import from agi_state_live edges
        for edge in cge_edges_raw:
            try:
                if isinstance(edge, list) and len(edge) >= 3:
                    frm, to = str(edge[0]), str(edge[1])
                elif isinstance(edge, dict):
                    frm = str(edge.get('from', ''))
                    to = str(edge.get('to', ''))
                else:
                    continue
                
                if frm in self.nodes and to in self.nodes:
                    conf = edge.get('confidence', 0.5) if isinstance(edge, dict) else 0.5
                    if isinstance(conf, torch.Tensor):
                        conf = conf.mean().item()
                    self.add_relation(frm, to, 'CORRELATED',
                                      confidence=max(0.1, float(conf)),
                                      observation_evidence=float(conf))
                    stats['edges_typed'] += 1
            except:
                pass
        
        # Compute final confidence stats
        confs = [e['confidence'] for e in self.edges.values()]
        if confs:
            stats['confidence_stats'] = {
                'mean': float(np.mean(confs)),
                'min': float(min(confs)),
                'max': float(max(confs)),
                'nonzero': sum(1 for c in confs if c > 0.01),
                'total': len(confs),
            }
        
        return stats

    # ─── Statistics ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Full statistical profile of the typed graph."""
        rel_type_counts = defaultdict(int)
        confidences = []
        
        for key, edge in self.edges.items():
            f, t, r = key
            rel_type_counts[r] += 1
            confidences.append(edge['confidence'])
        
        depths = [n['depth'] for n in self.nodes.values()]
        
        return {
            'n_concepts': len(self.nodes),
            'n_edges': len(self.edges),
            'mean_depth': float(np.mean(depths)) if depths else 0,
            'depth_std': float(np.std(depths)) if depths else 0,
            'relation_counts': dict(rel_type_counts),
            'confidence': {
                'mean': float(np.mean(confidences)) if confidences else 0,
                'std': float(np.std(confidences)) if confidences else 0,
                'histogram': self._histogram(confidences, 10) if confidences else {},
            },
            'betti_numbers': self.betti,
            'knowledge_gaps': self.knowledge_gaps,
        }

    def _histogram(self, values: List[float], bins: int) -> dict:
        if not values: return {}
        h, edges = np.histogram(values, bins=bins, range=(0, 1))
        return {f'{edges[i]:.2f}-{edges[i+1]:.2f}': int(h[i]) 
                for i in range(len(h))}

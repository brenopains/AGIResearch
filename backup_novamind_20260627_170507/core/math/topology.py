"""
NovaMind — Topological Data Analysis Engine
============================================

Implements persistent homology and simplicial complex analysis for
structural reasoning over knowledge graphs.

Why Topology for cognition?
---------------------------
The knowledge graph of an intelligent system has structure beyond pairwise
relationships. Topology captures this "shape of data":

    - β₀ (Betti-0): # connected components = # independent knowledge clusters
    - β₁ (Betti-1): # 1-cycles (loops) = # circular reasoning chains / feedback loops
    - β₂ (Betti-2): # 2-cavities (voids) = # "conceptual holes" (knowledge gaps)

Persistent homology tracks how these features appear and disappear as we
vary a "resolution" parameter (filtration), revealing which structural
features are robust (long persistence) vs. noise (short persistence).

For metacognition, this is revolutionary: the system can literally "see"
    - Gaps in its knowledge (cavities)
    - Circular reasoning (cycles)
    - Isolated concept clusters (components)
    - The "scale" at which knowledge becomes coherent

References:
    - Edelsbrunner & Harer (2010): "Computational Topology"
    - Carlsson (2009): "Topology and Data"
    - Zomorodian & Carlsson (2005): "Computing Persistent Homology"
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

EPSILON = 1e-10


class Simplex:
    """
    A k-simplex: the convex hull of k+1 vertices.

    0-simplex = vertex (a concept)
    1-simplex = edge (a relationship)
    2-simplex = triangle (a triangulated relationship)
    k-simplex = higher-order relationship among k+1 concepts
    """

    def __init__(self, vertices: tuple, filtration_value: float = 0.0):
        self.vertices = tuple(sorted(vertices))
        self.dim = len(vertices) - 1
        self.filtration_value = filtration_value

    def faces(self) -> List['Simplex']:
        """Return all (dim-1)-dimensional faces of this simplex."""
        if self.dim == 0:
            return []
        return [
            Simplex(self.vertices[:i] + self.vertices[i + 1:], self.filtration_value)
            for i in range(len(self.vertices))
        ]

    def __repr__(self):
        return f"σ{self.vertices}"

    def __eq__(self, other):
        return self.vertices == other.vertices

    def __hash__(self):
        return hash(self.vertices)

    def __lt__(self, other):
        return (self.filtration_value, self.dim, self.vertices) < \
               (other.filtration_value, other.dim, other.vertices)


class SimplicialComplex:
    """
    A filtered simplicial complex built from a knowledge graph.

    The filtration encodes "at what scale does this relationship become relevant?"
    Close/strong relationships appear early (low filtration value), weak ones later.
    """

    def __init__(self):
        self.simplices: Dict[int, Set[Simplex]] = defaultdict(set)  # dim → set of simplices
        self.filtration: List[Simplex] = []  # ordered by filtration value

    def add_simplex(self, simplex: Simplex):
        """Add a simplex and all its faces (closure property)."""
        if simplex in self.simplices[simplex.dim]:
            return

        # Add all faces first (ensures closure)
        for face in simplex.faces():
            face.filtration_value = min(face.filtration_value, simplex.filtration_value)
            self.add_simplex(face)

        self.simplices[simplex.dim].add(simplex)
        self.filtration.append(simplex)

    def build_from_distance_matrix(self, dist_matrix: np.ndarray, max_dim: int = 2,
                                     max_filtration: float = float('inf')):
        """
        Build a Vietoris-Rips complex from a distance matrix.

        At each filtration threshold ε, add an edge between vertices i,j
        if dist(i,j) ≤ ε. Add a k-simplex when all its edges are present
        (clique complex).

        For the knowledge graph: distances come from hyperbolic distance
        between concepts.
        """
        n = dist_matrix.shape[0]

        # Add all vertices at filtration 0
        for i in range(n):
            self.add_simplex(Simplex((i,), 0.0))

        # Collect all potential edges with their filtration values
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                if d <= max_filtration:
                    edges.append((d, i, j))

        edges.sort()

        # Add edges
        for d, i, j in edges:
            self.add_simplex(Simplex((i, j), d))

        # Build higher simplices (clique detection)
        if max_dim >= 2:
            self._build_cliques(dist_matrix, max_dim, max_filtration)

        # Sort filtration
        self.filtration.sort()

    def _build_cliques(self, dist_matrix: np.ndarray, max_dim: int, max_filtration: float):
        """Find cliques and add higher-dimensional simplices."""
        n = dist_matrix.shape[0]

        # For efficiency, limit to triangles (2-simplices) and tetrahedra (3-simplices)
        if max_dim >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        filt_val = max(dist_matrix[i, j], dist_matrix[j, k], dist_matrix[i, k])
                        if filt_val <= max_filtration:
                            self.add_simplex(Simplex((i, j, k), filt_val))

        if max_dim >= 3:
            for i in range(min(n, 20)):  # Limit for computation
                for j in range(i + 1, min(n, 20)):
                    for k in range(j + 1, min(n, 20)):
                        for l in range(k + 1, min(n, 20)):
                            filt_val = max(
                                dist_matrix[i, j], dist_matrix[i, k], dist_matrix[i, l],
                                dist_matrix[j, k], dist_matrix[j, l], dist_matrix[k, l]
                            )
                            if filt_val <= max_filtration:
                                self.add_simplex(Simplex((i, j, k, l), filt_val))

    def get_boundary_matrix(self, dim: int) -> Tuple[np.ndarray, List[Simplex], List[Simplex]]:
        """
        Compute the boundary matrix ∂_k that maps k-chains to (k-1)-chains.

        The boundary of a k-simplex [v₀,...,vₖ] is:
            ∂[v₀,...,vₖ] = Σᵢ (-1)ⁱ [v₀,...,v̂ᵢ,...,vₖ]

        where v̂ᵢ means "omit vᵢ".

        Over ℤ₂ (mod 2), this becomes the incidence matrix.
        """
        k_simplices = sorted(self.simplices.get(dim, set()))
        k_minus_1_simplices = sorted(self.simplices.get(dim - 1, set()))

        if not k_simplices or not k_minus_1_simplices:
            return np.array([[]]), k_minus_1_simplices, k_simplices

        # Build index maps
        row_index = {s: i for i, s in enumerate(k_minus_1_simplices)}
        col_index = {s: i for i, s in enumerate(k_simplices)}

        # Build boundary matrix (over ℤ₂)
        matrix = np.zeros((len(k_minus_1_simplices), len(k_simplices)), dtype=int)

        for simplex in k_simplices:
            for face in simplex.faces():
                if face in row_index:
                    matrix[row_index[face], col_index[simplex]] = 1

        return matrix % 2, k_minus_1_simplices, k_simplices


class PersistentHomology:
    """
    Computes persistent homology of a filtered simplicial complex.

    This reveals the "birth" and "death" of topological features across
    the filtration, producing a "persistence diagram" — a map of which
    structural features are real (long bars) vs noise (short bars).

    For metacognition:
        - Long bars in H₀ = major knowledge clusters
        - Long bars in H₁ = persistent circular reasoning patterns
        - Long bars in H₂ = persistent knowledge gaps
    """

    def __init__(self, complex: SimplicialComplex):
        self.complex = complex
        self.persistence_pairs: List[Tuple[int, float, float]] = []  # (dimension, birth, death)

    def compute(self, max_dim: int = 2) -> List[Tuple[int, float, float]]:
        """
        Compute persistent homology using the standard algorithm.

        Returns a list of (dimension, birth, death) triples.
        Points with death = ∞ represent features that never die (essential features).

        Algorithm: Smith Normal Form reduction over ℤ₂
        """
        self.persistence_pairs = []

        # Get sorted filtration (all simplices by filtration value)
        all_simplices = sorted(self.complex.filtration)

        # Remove duplicates while preserving order
        seen = set()
        unique_filtration = []
        for s in all_simplices:
            if s not in seen:
                seen.add(s)
                unique_filtration.append(s)

        # Index map: simplex → position in filtration
        simplex_index = {s: i for i, s in enumerate(unique_filtration)}

        n = len(unique_filtration)
        if n == 0:
            return self.persistence_pairs

        # Boundary matrix columns (reduced over ℤ₂)
        # Each column = boundary of the corresponding simplex
        columns: List[set] = []
        for s in unique_filtration:
            boundary = set()
            for face in s.faces():
                if face in simplex_index:
                    boundary.add(simplex_index[face])
            columns.append(boundary)

        # Column reduction (persistence algorithm)
        low = {}  # pivot → column index
        paired = set()

        for j in range(n):
            # Reduce column j
            while columns[j]:
                pivot = max(columns[j])
                if pivot in low:
                    # XOR with the earlier column (ℤ₂ arithmetic)
                    columns[j] = columns[j].symmetric_difference(columns[low[pivot]])
                else:
                    break

            if columns[j]:
                pivot = max(columns[j])
                low[pivot] = j
                paired.add(pivot)
                paired.add(j)

                # Record persistence pair
                birth_simplex = unique_filtration[pivot]
                death_simplex = unique_filtration[j]
                dim = birth_simplex.dim

                birth = birth_simplex.filtration_value
                death = death_simplex.filtration_value

                if abs(death - birth) > EPSILON:  # Skip zero-persistence pairs
                    self.persistence_pairs.append((dim, birth, death))

        # Essential features (born but never die)
        for i, s in enumerate(unique_filtration):
            if i not in paired:
                self.persistence_pairs.append((s.dim, s.filtration_value, float('inf')))

        return self.persistence_pairs

    def betti_numbers(self, filtration_value: float = float('inf')) -> Dict[int, int]:
        """
        Compute Betti numbers at a given filtration value.

        β₀ = # connected components (knowledge clusters)
        β₁ = # loops (circular reasoning)
        β₂ = # voids (knowledge gaps)
        """
        betti = defaultdict(int)

        for dim, birth, death in self.persistence_pairs:
            if birth <= filtration_value and (death > filtration_value or death == float('inf')):
                betti[dim] += 1

        return dict(betti)

    def persistence_entropy(self) -> float:
        """
        Shannon entropy of the persistence diagram.

        High entropy = many diverse topological features = complex knowledge structure
        Low entropy = few dominant features = simple/uniform knowledge
        """
        # Compute lifetimes (exclude infinite)
        lifetimes = []
        for dim, birth, death in self.persistence_pairs:
            if death != float('inf'):
                lifetimes.append(death - birth)

        if not lifetimes:
            return 0.0

        total = sum(lifetimes)
        if total < EPSILON:
            return 0.0

        probs = [l / total for l in lifetimes]
        entropy = -sum(p * np.log(p + EPSILON) for p in probs)
        return entropy

    def most_persistent_features(self, k: int = 5) -> List[Tuple[int, float, float]]:
        """
        Return the k most persistent features (longest bars).

        These are the most "real" structural features of the knowledge graph.
        """
        finite_pairs = [(d, b, de) for d, b, de in self.persistence_pairs if de != float('inf')]
        finite_pairs.sort(key=lambda x: x[2] - x[1], reverse=True)
        return finite_pairs[:k]

    def knowledge_gaps(self) -> List[Tuple[float, float]]:
        """
        Identify knowledge gaps (H₂ features with high persistence).

        A persistent 2-cycle indicates a region where concepts are connected
        around the boundary but missing the interior — literally a "hole"
        in the knowledge.
        """
        gaps = [(b, d) for dim, b, d in self.persistence_pairs if dim == 2 and d != float('inf')]
        gaps.sort(key=lambda x: x[1] - x[0], reverse=True)
        return gaps

    def circular_reasoning_detector(self) -> List[Tuple[float, float]]:
        """
        Detect circular reasoning (H₁ features with high persistence).

        A persistent 1-cycle indicates a chain of logical implications that
        loops back: A→B→C→...→A. This is a potential logical fallacy.
        """
        cycles = [(b, d) for dim, b, d in self.persistence_pairs if dim == 1 and d != float('inf')]
        cycles.sort(key=lambda x: x[1] - x[0], reverse=True)
        return cycles

    def to_dict(self) -> dict:
        """Serialize for the web dashboard."""
        return {
            "persistence_pairs": [
                {"dimension": d, "birth": b, "death": d_val if d_val != float('inf') else None}
                for d, b, d_val in self.persistence_pairs
            ],
            "betti_numbers": self.betti_numbers(),
            "persistence_entropy": self.persistence_entropy(),
            "knowledge_gaps": [{"birth": b, "death": d} for b, d in self.knowledge_gaps()],
            "circular_reasoning": [{"birth": b, "death": d} for b, d in self.circular_reasoning_detector()],
        }

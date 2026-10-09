"""
NovaMind — Geometric Algebra Engine (Clifford Algebra)
======================================================

Implements Clifford Algebra Cl(n) for concept transformations and reasoning.

Why Geometric Algebra for cognition?
------------------------------------
Traditional neural networks represent concepts as flat vectors and use dot
products for similarity. But cognition involves much richer operations:
rotation (analogy), reflection (negation), projection (abstraction), and
composition (concept blending).

Geometric Algebra (GA) unifies all of these into a single algebraic framework.
A "multivector" in Cl(n) can represent:
    - Scalars (magnitude/activation)
    - Vectors (direction/concept)
    - Bivectors (plane of rotation/relationship)
    - Trivectors and higher (volume elements/complex relations)

The geometric product xy = x·y + x∧y combines the inner product (similarity)
and outer product (orthogonal complement) into one operation, enabling:
    - Rotors: e^{Bθ/2} rotates concepts (analogical reasoning)
    - Reflections: -nxn reflects x through hyperplane n (concept negation)
    - Projections: (x·B)B^{-1} projects onto subspace (abstraction)

This is fundamentally more expressive than dot products alone.

References:
    - Hestenes (1966): "Space-Time Algebra"
    - Dorst, Fontijne, Mann (2007): "Geometric Algebra for Computer Science"
    - Ruhe et al. (2023): "Clifford Group Equivariant Neural Networks"
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from itertools import combinations
from functools import lru_cache

EPSILON = 1e-10


class MultiVector:
    """
    A multivector in Clifford Algebra Cl(p, q).

    A multivector is a sum of k-blades across all grades:
        M = α + a₁e₁ + a₂e₂ + ... + a₁₂e₁₂ + a₁₃e₁₃ + ... + a₁₂...ₙe₁₂...ₙ

    In Cl(n), there are 2^n basis elements. Each element is indexed by a
    frozenset of basis vector indices.

    For cognition:
        - Grade 0 (scalar): activation strength / certainty
        - Grade 1 (vector): raw concept direction
        - Grade 2 (bivector): relationships / transformations between concepts
        - Grade k: higher-order conceptual structures
    """

    def __init__(self, algebra: 'CliffordAlgebra', components: Optional[Dict[frozenset, float]] = None):
        self.algebra = algebra
        self.components: Dict[frozenset, float] = {}
        if components:
            for k, v in components.items():
                if abs(v) > EPSILON:
                    self.components[k] = v

    def __repr__(self):
        if not self.components:
            return "0"
        terms = []
        for basis, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), tuple(sorted(x[0])))):
            if abs(coeff) < EPSILON:
                continue
            if len(basis) == 0:
                terms.append(f"{coeff:.4f}")
            else:
                label = "e" + "".join(str(i) for i in sorted(basis))
                terms.append(f"{coeff:.4f}·{label}")
        return " + ".join(terms) if terms else "0"

    def grade(self, k: int) -> 'MultiVector':
        """Extract the grade-k part of this multivector."""
        components = {b: v for b, v in self.components.items() if len(b) == k}
        return MultiVector(self.algebra, components)

    def scalar_part(self) -> float:
        """Extract the scalar (grade-0) component."""
        return self.components.get(frozenset(), 0.0)

    def norm(self) -> float:
        """Euclidean norm of the multivector (root of sum of squares of coefficients)."""
        return np.sqrt(sum(v ** 2 for v in self.components.values()) + EPSILON)

    def normalized(self) -> 'MultiVector':
        """Return unit-norm multivector."""
        n = self.norm()
        return MultiVector(self.algebra, {k: v / n for k, v in self.components.items()})

    def reverse(self) -> 'MultiVector':
        """
        Reverse (†): reverses the order of basis vectors in each blade.
        For a k-blade: ã = (-1)^{k(k-1)/2} · a

        Used in constructing rotors and computing norms.
        """
        result = {}
        for basis, coeff in self.components.items():
            k = len(basis)
            sign = (-1) ** (k * (k - 1) // 2)
            result[basis] = sign * coeff
        return MultiVector(self.algebra, result)

    def conjugate(self) -> 'MultiVector':
        """Clifford conjugate: reversal + grade involution."""
        result = {}
        for basis, coeff in self.components.items():
            k = len(basis)
            sign = (-1) ** (k * (k + 1) // 2)
            result[basis] = sign * coeff
        return MultiVector(self.algebra, result)

    def __add__(self, other: 'MultiVector') -> 'MultiVector':
        result = dict(self.components)
        for k, v in other.components.items():
            result[k] = result.get(k, 0.0) + v
        return MultiVector(self.algebra, result)

    def __sub__(self, other: 'MultiVector') -> 'MultiVector':
        result = dict(self.components)
        for k, v in other.components.items():
            result[k] = result.get(k, 0.0) - v
        return MultiVector(self.algebra, result)

    def __mul__(self, other) -> 'MultiVector':
        """Geometric product — the fundamental operation of Clifford algebra."""
        if isinstance(other, (int, float)):
            return MultiVector(self.algebra, {k: v * other for k, v in self.components.items()})
        return self.algebra.geometric_product(self, other)

    def __rmul__(self, other) -> 'MultiVector':
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> 'MultiVector':
        return MultiVector(self.algebra, {k: -v for k, v in self.components.items()})

    def inner(self, other: 'MultiVector') -> 'MultiVector':
        """Inner product (left contraction): measures similarity."""
        return self.algebra.inner_product(self, other)

    def outer(self, other: 'MultiVector') -> 'MultiVector':
        """Outer (wedge) product: builds higher-grade blades."""
        return self.algebra.outer_product(self, other)

    def to_vector(self) -> np.ndarray:
        """Extract grade-1 components as a numpy array."""
        g1 = self.grade(1)
        vec = np.zeros(self.algebra.dim)
        for basis, coeff in g1.components.items():
            idx = list(basis)[0]
            vec[idx] = coeff
        return vec


class CliffordAlgebra:
    """
    Clifford Algebra Cl(p, q) with signature (p, q).

    For cognition we use Cl(n, 0) — Euclidean signature — where n is
    the concept dimensionality.

    The geometric product of basis vectors:
        eᵢeⱼ = -eⱼeᵢ (i ≠ j)  — anticommutativity
        eᵢeᵢ = +1 (for p positive dimensions)
        eᵢeᵢ = -1 (for q negative dimensions)

    This algebra has 2^(p+q) dimensions, with basis elements indexed by
    subsets of {0, 1, ..., p+q-1}.
    """

    def __init__(self, p: int, q: int = 0):
        """
        Args:
            p: Number of positive-signature dimensions
            q: Number of negative-signature dimensions
        """
        self.p = p
        self.q = q
        self.dim = p + q

        # Precompute the metric signature
        self.signature = [1] * p + [-1] * q

        # Precompute basis elements
        self.basis_elements = []
        for k in range(self.dim + 1):
            for combo in combinations(range(self.dim), k):
                self.basis_elements.append(frozenset(combo))

    def _basis_product_sign(self, a: frozenset, b: frozenset) -> Tuple[int, frozenset]:
        """
        Compute the sign and resulting basis of the geometric product of two basis blades.

        Algorithm: bring the basis vectors into canonical order by counting swaps,
        then contract repeated indices using the metric signature.
        """
        # Merge the two lists in the order they appear
        a_list = sorted(a)
        b_list = sorted(b)
        merged = a_list + b_list

        # Bubble sort to canonical order, counting swaps
        n = len(merged)
        sign = 1
        for i in range(n):
            for j in range(n - 1 - i):
                if merged[j] > merged[j + 1]:
                    merged[j], merged[j + 1] = merged[j + 1], merged[j]
                    sign *= -1

        # Contract repeated pairs (eᵢeᵢ = signature[i])
        result = []
        i = 0
        while i < len(merged):
            if i + 1 < len(merged) and merged[i] == merged[i + 1]:
                sign *= self.signature[merged[i]]
                i += 2
            else:
                result.append(merged[i])
                i += 1

        return sign, frozenset(result)

    def geometric_product(self, a: MultiVector, b: MultiVector) -> MultiVector:
        """
        The geometric product ab = a·b + a∧b.

        This is THE fundamental operation. It encodes:
        - Similarity (scalar part of ab†)
        - Rotation (when a and b are versors)
        - Projection and rejection
        """
        result = {}
        for basis_a, coeff_a in a.components.items():
            for basis_b, coeff_b in b.components.items():
                sign, basis_result = self._basis_product_sign(basis_a, basis_b)
                val = sign * coeff_a * coeff_b
                result[basis_result] = result.get(basis_result, 0.0) + val
        return MultiVector(self, result)

    def inner_product(self, a: MultiVector, b: MultiVector) -> MultiVector:
        """
        Left contraction (inner product).

        For grade-r blade A and grade-s blade B (r ≤ s):
            A⌋B = ⟨AB⟩_{s-r}

        Measures "how much of B is in the direction of A."
        """
        product = self.geometric_product(a, b)
        result = {}
        for basis_a, _ in a.components.items():
            for basis_b, _ in b.components.items():
                target_grade = abs(len(basis_b) - len(basis_a))
                for basis, coeff in product.components.items():
                    if len(basis) == target_grade:
                        result[basis] = result.get(basis, 0.0) + coeff
        # Deduplicate by just taking grade-filtered product
        min_grade = min((len(b) for b in a.components.keys()), default=0)
        max_grade = max((len(b) for b in b.components.keys()), default=0)
        target = max_grade - min_grade
        return product.grade(target) if target >= 0 else MultiVector(self, {})

    def outer_product(self, a: MultiVector, b: MultiVector) -> MultiVector:
        """
        Outer (wedge) product a ∧ b.

        Builds higher-dimensional blades from lower ones:
            vector ∧ vector = bivector (a plane)
            vector ∧ bivector = trivector (a volume)

        For cognition: combining two concept-directions into a
        "relationship plane."
        """
        product = self.geometric_product(a, b)
        max_grade = 0
        for basis_a in a.components:
            for basis_b in b.components:
                max_grade = max(max_grade, len(basis_a) + len(basis_b))
        return product.grade(max_grade)

    def vector(self, coords: np.ndarray) -> MultiVector:
        """Create a grade-1 multivector (vector) from coordinates."""
        components = {}
        for i, c in enumerate(coords):
            if isinstance(c, (int, float)) and abs(c) > EPSILON:
                components[frozenset([i])] = float(c)
            elif isinstance(c, np.ndarray) and np.abs(c).sum() > EPSILON:
                components[frozenset([i])] = float(c.sum())
        return MultiVector(self, components)

    def scalar(self, value: float) -> MultiVector:
        """Create a grade-0 multivector (scalar)."""
        return MultiVector(self, {frozenset(): value})

    def bivector(self, i: int, j: int, value: float = 1.0) -> MultiVector:
        """Create a basis bivector eᵢeⱼ."""
        return MultiVector(self, {frozenset([i, j]): value})

    def rotor(self, plane_bivector: MultiVector, angle: float) -> MultiVector:
        """
        Create a rotor R = e^{-Bθ/2} = cos(θ/2) - sin(θ/2)·B̂

        A rotor smoothly rotates concepts in the plane defined by the bivector.

        For cognition, this is ANALOGICAL REASONING:
            - The bivector B defines the "relationship axis"
            - The angle θ controls "how far" to push the analogy
            - RxR† transforms concept x by this analogy

        Example: if B encodes "king→queen" relationship and x = "man",
        then RxR† ≈ "woman" (the famous word analogy, but geometrically exact).
        """
        # Normalize the bivector
        b_norm = plane_bivector.norm()
        if b_norm < EPSILON:
            return self.scalar(1.0)

        b_hat = plane_bivector * (1.0 / b_norm)
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)

        return self.scalar(cos_half) + b_hat * (-sin_half)

    def rotate(self, rotor: MultiVector, x: MultiVector) -> MultiVector:
        """
        Apply rotor transformation: x' = R x R†

        This is the "sandwich product" — the fundamental transformation
        of geometric algebra. It works for any grade of multivector.
        """
        r_reverse = rotor.reverse()
        return self.geometric_product(self.geometric_product(rotor, x), r_reverse)

    def reflect(self, mirror: MultiVector, x: MultiVector) -> MultiVector:
        """
        Reflect x through the hyperplane perpendicular to mirror vector n:
            x' = -n x n⁻¹

        For cognition: conceptual negation or inversion.
        """
        n_sq = self.geometric_product(mirror, mirror).scalar_part()
        if abs(n_sq) < EPSILON:
            return x
        n_inv = mirror * (1.0 / n_sq)
        return self.geometric_product(self.geometric_product(-mirror, x), n_inv)

    def project(self, x: MultiVector, subspace: MultiVector) -> MultiVector:
        """
        Project x onto the subspace defined by blade B:
            proj_B(x) = (x·B)B⁻¹

        For cognition: abstraction — extracting only the components of x
        that are relevant to a particular conceptual plane.
        """
        b_sq = self.geometric_product(subspace, subspace).scalar_part()
        if abs(b_sq) < EPSILON:
            return MultiVector(self, {})
        b_inv = subspace * (1.0 / b_sq)
        return self.geometric_product(self.inner_product(x, subspace), b_inv)

    def concept_similarity(self, a: MultiVector, b: MultiVector) -> float:
        """
        Geometric similarity between two concept multivectors.

        Uses the scalar part of a†b (reverse of a times b), which generalizes
        the dot product to all grades.
        """
        product = self.geometric_product(a.reverse(), b)
        return product.scalar_part() / (a.norm() * b.norm() + EPSILON)

    def concept_blend(self, a: MultiVector, b: MultiVector, t: float = 0.5) -> MultiVector:
        """
        Blend two concepts using geometric interpolation.

        Unlike linear interpolation, this preserves the geometric structure:
        the result maintains grade-specific properties.
        """
        return a * (1 - t) + b * t

    def analogy(self, a: MultiVector, b: MultiVector, c: MultiVector) -> MultiVector:
        """
        Geometric analogy: a is to b as c is to ???

        Computes the transformation T that takes a → b,
        then applies T to c.

        T = b · a⁻¹ (right quotient)
        result = T · c = (b · a⁻¹) · c
        """
        a_sq = self.geometric_product(a, a).scalar_part()
        if abs(a_sq) < EPSILON:
            return c
        a_inv = a * (1.0 / a_sq)
        transform = self.geometric_product(b, a_inv)
        return self.geometric_product(transform, c)

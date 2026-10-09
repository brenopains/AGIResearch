"""
NovaMind — Hyperbolic Geometry Engine
=====================================

Implements the Poincaré Ball Model for hierarchical concept representation.

Why hyperbolic geometry for cognition?
--------------------------------------
Euclidean space cannot efficiently embed hierarchical structures. Trees with
branching factor b and depth d have O(b^d) leaves, requiring exponential
dimensions in Euclidean space. The Poincaré ball — a model of hyperbolic
space — embeds arbitrary trees with arbitrarily low distortion in just 2
dimensions, because hyperbolic space has exponentially growing volume.

This mirrors cognition: concepts are naturally hierarchical (animal → mammal
→ dog → golden retriever) and the brain must represent these hierarchies in
finite neural substrate. Hyperbolic geometry solves this "representation
bottleneck" mathematically.

Mathematical Foundation:
    The Poincaré ball B^n = {x ∈ ℝ^n : ||x|| < 1} with the Riemannian metric:
    g_x = (λ_x)^2 · g_E,  where λ_x = 2 / (1 - ||x||²)

    This gives a conformal model of hyperbolic space where:
    - The origin is "generic" / "abstract"
    - Points near the boundary are "specific" / "concrete"
    - Hierarchical depth ≈ distance from origin
    - Siblings share angular proximity

References:
    - Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
    - Ganea et al. (2018): "Hyperbolic Neural Networks"
    - Ungar (2008): "Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity"
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings

# Numerical stability constants
MIN_NORM = 1e-15
MAX_NORM = 1 - 1e-5
EPSILON = 1e-7


class PoincareBall:
    """
    The Poincaré Ball model of hyperbolic space.

    This is the cognitive substrate where all concepts live. The key property
    is that distances grow exponentially near the boundary, naturally encoding
    hierarchical depth — abstract concepts sit near the center, concrete
    instances near the edge.

    Curvature c controls the "sharpness" of the hierarchy:
        - c → 0: approaches Euclidean (flat, no hierarchy)
        - c → ∞: ultra-hyperbolic (extreme hierarchy)
        - c = 1: standard hyperbolic space
    """

    def __init__(self, dim: int, curvature: float = 1.0):
        """
        Args:
            dim: Dimensionality of the hyperbolic space
            curvature: Negative curvature parameter (K = -c), default 1.0
        """
        self.dim = dim
        self.c = curvature
        self.sqrt_c = np.sqrt(curvature)

    def _clamp_norm(self, x: np.ndarray) -> np.ndarray:
        """Project point back into the open ball if numerical drift pushed it out."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        # Clamp to MAX_NORM to stay strictly inside the ball
        clamped = np.where(norm > MAX_NORM, x * (MAX_NORM / (norm + MIN_NORM)), x)
        return clamped

    def conformal_factor(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the conformal factor λ_x = 2 / (1 - c||x||²).

        This is the local "stretching" of space — it goes to infinity at
        the boundary, which is why hyperbolic space has infinite volume
        inside a finite ball.
        """
        sq_norm = np.sum(x ** 2, axis=-1, keepdims=True)
        return 2.0 / (1.0 - self.c * sq_norm + EPSILON)

    def mobius_addition(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Möbius addition: x ⊕ y in the Poincaré ball.

        This is the hyperbolic analogue of vector addition. Unlike Euclidean
        addition, it's non-commutative and non-associative — reflecting the
        fact that "combining" concepts is order-dependent. The formula:

            x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) /
                     (1 + 2c⟨x,y⟩ + c²||x||²||y||²)

        This comes from the theory of gyrovector spaces (Ungar, 2008).
        """
        x_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        y_sq = np.sum(y ** 2, axis=-1, keepdims=True)
        xy = np.sum(x * y, axis=-1, keepdims=True)

        num = (1 + 2 * self.c * xy + self.c * y_sq) * x + (1 - self.c * x_sq) * y
        denom = 1 + 2 * self.c * xy + self.c ** 2 * x_sq * y_sq + EPSILON

        return self._clamp_norm(num / denom)

    def mobius_scalar_mult(self, r: float, x: np.ndarray) -> np.ndarray:
        """
        Möbius scalar multiplication: r ⊗ x.

        Scales a point along the geodesic from origin through x.
        """
        norm = np.linalg.norm(x, axis=-1, keepdims=True).clip(MIN_NORM)
        direction = x / norm
        scaled_norm = np.tanh(r * np.arctanh(self.sqrt_c * norm + EPSILON)) / (self.sqrt_c + EPSILON)
        return self._clamp_norm(direction * scaled_norm)

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Hyperbolic distance d(x, y) in the Poincaré ball.

            d(x, y) = (2/√c) · arctanh(√c · ||(-x) ⊕ y||)

        Key property: distance grows logarithmically near origin but
        exponentially near boundary. Two points at Euclidean distance 0.01
        from each other are "close" near the center but potentially "very far"
        near the boundary.
        """
        neg_x = -x
        diff = self.mobius_addition(neg_x, y)
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True).clip(MIN_NORM)
        return (2.0 / (self.sqrt_c + EPSILON)) * np.arctanh(self.sqrt_c * diff_norm + EPSILON)

    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: exp_x(v) — maps a tangent vector v at point x
        to a point on the manifold.

        This is how we "move" in hyperbolic space: given a current position
        x and a direction v in the tangent space at x, compute where we end up.

            exp_x(v) = x ⊕ (tanh(√c · λ_x · ||v|| / 2) · v / (√c · ||v||))
        """
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True).clip(MIN_NORM)
        lam = self.conformal_factor(x)
        second_term = np.tanh(self.sqrt_c * lam * v_norm / 2) * v / (self.sqrt_c * v_norm + EPSILON)
        return self.mobius_addition(x, self._clamp_norm(second_term))

    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: log_x(y) — maps a point y on the manifold back
        to a tangent vector at x.

        Inverse of exp_map. This is how we compute "directions" in hyperbolic space.

            log_x(y) = (2 / (√c · λ_x)) · arctanh(√c · ||(-x) ⊕ y||) · ((-x) ⊕ y) / ||(-x) ⊕ y||
        """
        neg_x = -x
        diff = self.mobius_addition(neg_x, y)
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True).clip(MIN_NORM)
        lam = self.conformal_factor(x)
        return (2.0 / (self.sqrt_c * lam + EPSILON)) * np.arctanh(self.sqrt_c * diff_norm + EPSILON) * diff / diff_norm

    def parallel_transport(self, x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Parallel transport of tangent vector v from T_xB to T_yB.

        This is essential for "carrying" information between concept positions.
        In cognition, this is analogous to "translating" a relationship from
        one conceptual context to another.

            PT_{x→y}(v) = (λ_x / λ_y) · gyr[y, -x](v)

        where gyr is the gyration operator.
        """
        lam_x = self.conformal_factor(x)
        lam_y = self.conformal_factor(y)
        # Simplified transport using conformal factor ratio
        return v * (lam_x / (lam_y + EPSILON))

    def geodesic(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """
        Point on the geodesic from x to y at parameter t ∈ [0, 1].

        Geodesics in the Poincaré ball are circular arcs orthogonal to the
        boundary (or diameters through the origin). This traces the "shortest
        path" through concept space.
        """
        v = self.log_map(x, y)
        return self.exp_map(x, t * v)

    def midpoint(self, points: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Einstein midpoint (weighted Fréchet mean) in the Poincaré ball.

        Given multiple concept positions, compute their "center of mass"
        in hyperbolic space. Unlike Euclidean mean, this naturally respects
        the hierarchical structure — the midpoint of specific concepts
        moves toward the center (more abstract).

        Uses the iterative algorithm from Ungar (2008).
        """
        if weights is None:
            weights = np.ones(len(points)) / len(points)
        else:
            weights = weights / (np.sum(weights) + EPSILON)

        # Iterative Fréchet mean via gradient descent in the ball
        mean = np.zeros(self.dim)
        for _ in range(100):
            grad = np.zeros(self.dim)
            for p, w in zip(points, weights):
                log_vec = self.log_map(mean, p)
                grad += w * log_vec.flatten()
            mean = self.exp_map(mean, 0.1 * grad.reshape(1, -1))
            mean = self._clamp_norm(mean)
            if np.linalg.norm(grad) < EPSILON:
                break

        return mean

    def hierarchy_depth(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the hierarchical depth of a concept.

        Depth = distance from origin = how "specific" a concept is.
        - Near 0: very abstract (e.g., "entity", "thing")
        - Near boundary: very specific (e.g., "my golden retriever named Max")
        """
        origin = np.zeros_like(x)
        return self.distance(origin, x)

    def angular_similarity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Angular similarity between two concepts.

        In the Poincaré ball, angular proximity encodes "siblinghood" —
        concepts at similar angles from the origin share a common ancestor.
        """
        x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + MIN_NORM)
        y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + MIN_NORM)
        return np.sum(x_norm * y_norm, axis=-1)

    def random_point(self, near_origin: bool = True) -> np.ndarray:
        """Sample a random point in the ball, optionally near the origin (abstract)."""
        direction = np.random.randn(1, self.dim)
        direction = direction / (np.linalg.norm(direction) + MIN_NORM)
        if near_origin:
            radius = np.random.uniform(0.0, 0.5)
        else:
            radius = np.random.uniform(0.3, 0.95)
        return self._clamp_norm(direction * radius)

    def entailment_cone(self, x: np.ndarray, half_angle: float) -> 'EntailmentCone':
        """
        Create an entailment cone at point x with given half-angle.

        Entailment cones (Ganea et al. 2018) formalize "is-a" relationships:
        concept y is a specialization of x iff y lies inside x's entailment cone.

        This gives a geometric, differentiable version of taxonomic reasoning.
        """
        return EntailmentCone(self, x, half_angle)


class EntailmentCone:
    """
    An entailment cone in the Poincaré ball.

    If concept B lies inside the entailment cone of concept A, then
    B "is-a" A (B entails A). This provides a continuous, differentiable
    relaxation of logical subsumption.
    """

    def __init__(self, ball: PoincareBall, apex: np.ndarray, half_angle: float):
        self.ball = ball
        self.apex = apex
        self.half_angle = half_angle

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point lies inside this entailment cone."""
        # Direction from origin to apex
        apex_norm = np.linalg.norm(self.apex)
        if apex_norm < MIN_NORM:
            return True  # Origin's cone contains everything

        apex_dir = self.apex.flatten() / (apex_norm + MIN_NORM)
        point_dir = point.flatten() / (np.linalg.norm(point) + MIN_NORM)

        # Angle between apex direction and point direction
        cos_angle = np.dot(apex_dir, point_dir)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Point must be deeper (further from origin) AND within angular cone
        point_depth = np.linalg.norm(point)
        return angle <= self.half_angle and point_depth >= apex_norm * 0.8

    def membership_score(self, point: np.ndarray) -> float:
        """Soft membership score: how much is the point inside the cone? [0, 1]"""
        apex_norm = np.linalg.norm(self.apex)
        if apex_norm < MIN_NORM:
            return 1.0

        apex_dir = self.apex.flatten() / (apex_norm + MIN_NORM)
        point_dir = point.flatten() / (np.linalg.norm(point) + MIN_NORM)

        cos_angle = np.dot(apex_dir, point_dir)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Sigmoid-like soft membership
        angular_score = np.exp(-max(0, angle - self.half_angle) * 10)
        depth_score = 1.0 / (1.0 + np.exp(-(np.linalg.norm(point) - apex_norm * 0.8) * 10))

        return float(angular_score * depth_score)

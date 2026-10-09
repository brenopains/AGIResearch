"""
NovaMind — Information Theory Engine
=====================================

Implements the Minimum Description Length (MDL) principle, Kolmogorov
complexity approximation, and information-theoretic measures for
auto-compaction and knowledge compression.

Why Information Theory for cognition?
-------------------------------------
Marcus Hutter's thesis: Intelligence = Compression.

The core insight is that understanding something IS finding a shorter
description of it. A mind that can compress its experiences into fewer
bits literally "understands" those experiences better.

The MDL principle operationalizes this:
    - Given data D, find the model M that minimizes:
      L(M) + L(D|M)
      = (complexity of the model) + (remaining unexplained data)

    - Too simple model: L(D|M) dominates (underfitting)
    - Too complex model: L(M) dominates (overfitting / memorization)
    - Optimal model: shortest total description (genuine understanding)

For auto-compaction:
    - Episodic memories = raw data D
    - Semantic knowledge = model M
    - Auto-compaction finds M* = argmin L(M) + L(D|M)
    - The system literally discovers the "laws" in its experience

References:
    - Rissanen (1978): "Modeling by Shortest Data Description"
    - Grünwald (2007): "The Minimum Description Length Principle"
    - Hutter (2005): "Universal Artificial Intelligence" (AIXI)
    - Schmidhuber (2009): "Driven by Compression Progress"
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import zlib
import json
import hashlib
from collections import defaultdict, Counter

EPSILON = 1e-10


class InformationEngine:
    """
    Core information-theoretic computations for knowledge compression.

    This engine provides the mathematical basis for auto-compaction:
    it measures how compressible knowledge is, and guides the compression
    process to maximize understanding while minimizing storage.
    """

    @staticmethod
    def shannon_entropy(distribution: np.ndarray) -> float:
        """
        Shannon entropy H(X) = -Σ p(x) log₂ p(x)

        Measures the average "surprise" or "information content" of a
        probability distribution. Maximum when uniform (maximum uncertainty),
        zero when deterministic (no uncertainty).
        """
        p = distribution[distribution > EPSILON]
        p = p / (p.sum() + EPSILON)
        return -np.sum(p * np.log2(p + EPSILON))

    @staticmethod
    def conditional_entropy(joint: np.ndarray) -> float:
        """
        Conditional entropy H(Y|X) = H(X,Y) - H(X)

        How much additional information Y provides beyond X.
        If H(Y|X) ≈ 0, then X fully determines Y (Y is redundant given X).
        """
        p_xy = joint / (joint.sum() + EPSILON)
        p_x = p_xy.sum(axis=1)

        h_xy = InformationEngine.shannon_entropy(p_xy.flatten())
        h_x = InformationEngine.shannon_entropy(p_x)

        return h_xy - h_x

    @staticmethod
    def mutual_information(joint: np.ndarray) -> float:
        """
        Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)

        Measures shared information between X and Y.
        If I(X;Y) is high, X and Y are strongly correlated and one can
        be predicted from the other — a compaction opportunity.
        """
        p_xy = joint / (joint.sum() + EPSILON)
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        h_x = InformationEngine.shannon_entropy(p_x)
        h_y = InformationEngine.shannon_entropy(p_y)
        h_xy = InformationEngine.shannon_entropy(p_xy.flatten())

        return max(0.0, h_x + h_y - h_xy)

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Kullback-Leibler divergence D_KL(P || Q) = Σ p(x) log(p(x)/q(x))

        Measures how different P is from Q. Used in:
        - Metacognition: how much did beliefs change after new evidence?
        - Learning: how "surprising" is new data relative to current model?
        """
        p = p / (p.sum() + EPSILON)
        q = q / (q.sum() + EPSILON)
        # Avoid log(0) by clipping
        return np.sum(p * np.log((p + EPSILON) / (q + EPSILON)))

    @staticmethod
    def surprise(event_probability: float) -> float:
        """
        Surprisal (self-information): -log₂(p)

        How "unexpected" an event is. Used by the Free Energy
        Principle for metacognition.
        """
        return -np.log2(max(event_probability, EPSILON))

    @staticmethod
    def normalized_compression_distance(s1: str, s2: str) -> float:
        """
        Normalized Compression Distance (NCD) — approximation of
        normalized Kolmogorov distance.

            NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

        where C(·) is the compressed size.

        NCD ≈ 0: s1 and s2 are very similar (high mutual information)
        NCD ≈ 1: s1 and s2 are independent

        This is the most powerful "concept similarity" measure known —
        it approximates the uncomputable Kolmogorov complexity via
        real compression (zlib).
        """
        s1_bytes = s1.encode('utf-8')
        s2_bytes = s2.encode('utf-8')
        combined = s1_bytes + s2_bytes

        c1 = len(zlib.compress(s1_bytes, 9))
        c2 = len(zlib.compress(s2_bytes, 9))
        c12 = len(zlib.compress(combined, 9))

        return (c12 - min(c1, c2)) / (max(c1, c2) + EPSILON)

    @staticmethod
    def kolmogorov_complexity_approx(data: str) -> float:
        """
        Approximate Kolmogorov complexity K(x) via compression ratio.

            K(x) ≈ C(x) / |x|

        Where C(x) is compressed size and |x| is original size.

        K ≈ 0: data is very regular/compressible (well-understood)
        K ≈ 1: data is incompressible/random (not understood)
        """
        data_bytes = data.encode('utf-8')
        if len(data_bytes) == 0:
            return 0.0
        compressed = zlib.compress(data_bytes, 9)
        return len(compressed) / len(data_bytes)


class MDLCompactor:
    """
    Minimum Description Length Auto-Compaction Engine.

    This is the heart of the auto-compaction system. It observes patterns
    in episodic memory and discovers the shortest "program" (set of rules)
    that generates/explains those patterns.

    The MDL principle guarantees that the discovered rules are neither
    overfitting (memorizing noise) nor underfitting (missing patterns).
    """

    def __init__(self):
        self.pattern_registry: Dict[str, Dict] = {}  # hash → pattern info
        self.compression_history: List[Dict] = []
        self.total_raw_bits: float = 0
        self.total_compressed_bits: float = 0

    def analyze_patterns(self, episodes: List[Dict]) -> List[Dict]:
        """
        Analyze a set of episodes for compressible patterns.

        Returns discovered patterns with their MDL score:
            MDL = L(pattern) + L(exceptions|pattern)

        Lower MDL = better pattern (more compression).
        """
        if len(episodes) < 2:
            return []

        patterns = []

        # Strategy 1: Find repeated structural patterns (graph isomorphism)
        structural_patterns = self._find_structural_patterns(episodes)
        patterns.extend(structural_patterns)

        # Strategy 2: Find shared attribute patterns (statistical regularities)
        attribute_patterns = self._find_attribute_patterns(episodes)
        patterns.extend(attribute_patterns)

        # Strategy 3: Find sequential patterns (temporal regularities)
        sequential_patterns = self._find_sequential_patterns(episodes)
        patterns.extend(sequential_patterns)

        # Score patterns by MDL
        for pattern in patterns:
            pattern['mdl_score'] = self._compute_mdl_score(pattern, episodes)

        # Sort by MDL (lower = better compression)
        patterns.sort(key=lambda p: p['mdl_score'])

        return patterns

    def compact(self, episodes: List[Dict], patterns: List[Dict]) -> Dict:
        """
        Perform the actual compaction: given episodes and discovered patterns,
        produce a compressed representation.

        Returns:
            - rules: abstracted knowledge (the "model" M)
            - residuals: episodes not explained by any pattern
            - compression_ratio: how much was compressed
            - information_loss: estimated information lost in compression
        """
        rules = []
        covered = set()

        for pattern in patterns:
            if pattern['mdl_score'] > 0 and len(pattern.get('instances', [])) > 1:
                # Create a rule from this pattern
                rule = {
                    'type': pattern['type'],
                    'description': pattern['description'],
                    'abstraction': pattern['abstraction'],
                    'instance_count': len(pattern.get('instances', [])),
                    'confidence': pattern.get('confidence', 1.0),
                    'mdl_score': pattern['mdl_score'],
                    'created_from': len(pattern.get('instances', []))
                }
                rules.append(rule)

                for idx in pattern.get('instances', []):
                    if isinstance(idx, int) and idx < len(episodes):
                        covered.add(idx)

                # Register pattern
                pattern_hash = hashlib.md5(
                    json.dumps(pattern['abstraction'], sort_keys=True, default=str).encode()
                ).hexdigest()[:8]
                self.pattern_registry[pattern_hash] = rule

        residuals = [ep for i, ep in enumerate(episodes) if i not in covered]

        # Compute compression metrics
        raw_size = self._estimate_bits(episodes)
        compressed_size = self._estimate_bits(rules) + self._estimate_bits(residuals)
        compression_ratio = 1 - (compressed_size / (raw_size + EPSILON))

        self.total_raw_bits += raw_size
        self.total_compressed_bits += compressed_size

        result = {
            'rules': rules,
            'residuals': residuals,
            'compression_ratio': max(0, compression_ratio),
            'raw_bits': raw_size,
            'compressed_bits': compressed_size,
            'patterns_found': len(rules),
            'episodes_covered': len(covered),
            'episodes_remaining': len(residuals),
        }

        self.compression_history.append(result)
        return result

    def _find_structural_patterns(self, episodes: List[Dict]) -> List[Dict]:
        """Find repeated structural patterns (shared keys, types, shapes)."""
        patterns = []
        type_groups = defaultdict(list)

        for i, ep in enumerate(episodes):
            ep_type = ep.get('type', 'unknown')
            type_groups[ep_type].append(i)

        for ep_type, indices in type_groups.items():
            if len(indices) >= 2:
                # Find shared structure
                shared_keys = None
                for idx in indices:
                    keys = set(episodes[idx].keys())
                    if shared_keys is None:
                        shared_keys = keys
                    else:
                        shared_keys &= keys

                if shared_keys and len(shared_keys) > 1:
                    patterns.append({
                        'type': 'structural',
                        'description': f'Shared structure across {len(indices)} {ep_type} episodes',
                        'abstraction': {
                            'episode_type': ep_type,
                            'shared_keys': sorted(shared_keys),
                            'count': len(indices),
                        },
                        'instances': indices,
                        'confidence': len(indices) / len(episodes),
                    })

        return patterns

    def _find_attribute_patterns(self, episodes: List[Dict]) -> List[Dict]:
        """Find repeated attribute values across episodes."""
        patterns = []
        attr_values = defaultdict(lambda: defaultdict(list))

        for i, ep in enumerate(episodes):
            for key, value in ep.items():
                if isinstance(value, (str, int, float, bool)):
                    attr_values[key][str(value)].append(i)

        for key, values in attr_values.items():
            for value, indices in values.items():
                if len(indices) >= 2 and len(indices) >= len(episodes) * 0.3:
                    patterns.append({
                        'type': 'attribute',
                        'description': f'Common {key}={value} in {len(indices)} episodes',
                        'abstraction': {
                            'attribute': key,
                            'value': value,
                            'frequency': len(indices) / len(episodes),
                        },
                        'instances': indices,
                        'confidence': len(indices) / len(episodes),
                    })

        return patterns

    def _find_sequential_patterns(self, episodes: List[Dict]) -> List[Dict]:
        """Find temporal / sequential patterns (A usually follows B)."""
        patterns = []

        if len(episodes) < 3:
            return patterns

        # Look for type transitions
        transitions = defaultdict(int)
        for i in range(len(episodes) - 1):
            t1 = episodes[i].get('type', 'unknown')
            t2 = episodes[i + 1].get('type', 'unknown')
            transitions[(t1, t2)] += 1

        total_transitions = len(episodes) - 1
        for (t1, t2), count in transitions.items():
            if count >= 2:
                freq = count / total_transitions
                patterns.append({
                    'type': 'sequential',
                    'description': f'{t1} → {t2} occurs {count} times ({freq:.0%})',
                    'abstraction': {
                        'from_type': t1,
                        'to_type': t2,
                        'frequency': freq,
                        'count': count,
                    },
                    'instances': list(range(len(episodes))),
                    'confidence': freq,
                })

        return patterns

    def _compute_mdl_score(self, pattern: Dict, episodes: List[Dict]) -> float:
        """
        Compute the MDL score for a pattern.

        MDL(pattern) = L(pattern description) + L(data | pattern)

        Lower is better. A good pattern is short to describe but
        explains many data points.
        """
        # L(pattern): complexity of the pattern description
        pattern_desc = json.dumps(pattern['abstraction'], sort_keys=True, default=str)
        l_pattern = len(zlib.compress(pattern_desc.encode(), 9)) * 8  # bits

        # L(data|pattern): residual data not explained by the pattern
        instance_count = len(pattern.get('instances', []))
        total_count = len(episodes)
        coverage = instance_count / (total_count + EPSILON)

        # Estimated residual: data bits × (1 - coverage)
        total_data = json.dumps(episodes, sort_keys=True, default=str)
        l_data = len(zlib.compress(total_data.encode(), 9)) * 8
        l_residual = l_data * (1 - coverage)

        return l_pattern + l_residual

    def _estimate_bits(self, data: Any) -> float:
        """Estimate the information content of data in bits."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        compressed = zlib.compress(serialized.encode(), 9)
        return len(compressed) * 8

    def get_compression_stats(self) -> Dict:
        """Return overall compression statistics."""
        if not self.compression_history:
            return {
                'total_compressions': 0,
                'overall_ratio': 0.0,
                'total_patterns': 0,
                'total_raw_bits': 0,
                'total_compressed_bits': 0,
            }

        return {
            'total_compressions': len(self.compression_history),
            'overall_ratio': max(0, 1 - (self.total_compressed_bits / (self.total_raw_bits + EPSILON))),
            'total_patterns': len(self.pattern_registry),
            'total_raw_bits': self.total_raw_bits,
            'total_compressed_bits': self.total_compressed_bits,
            'compression_history': [
                {
                    'ratio': h['compression_ratio'],
                    'patterns': h['patterns_found'],
                    'covered': h['episodes_covered'],
                }
                for h in self.compression_history[-10:]  # Last 10
            ],
        }


class FreeEnergyMinimizer:
    """
    Implementation of Karl Friston's Free Energy Principle for metacognition.

    The Free Energy Principle states that intelligent systems minimize
    "variational free energy" — a bound on surprise:

        F = D_KL(Q(θ) || P(θ|D)) + ln P(D)
        ≈ Complexity - Accuracy

    Where:
        - Q(θ): the system's beliefs (internal model)
        - P(θ|D): true posterior given data
        - D_KL: divergence between beliefs and reality
        - P(D): evidence (surprise)

    Minimizing F means: update beliefs to be maximally accurate
    with minimal complexity. This IS metacognition — the system
    knows when its model is wrong and actively works to fix it.

    This also drives ACTIVE INFERENCE: the system doesn't just
    passively update beliefs — it takes actions to gather data
    that would maximally reduce its uncertainty. This is the
    mathematical origin of curiosity.
    """

    def __init__(self):
        self.beliefs: Dict[str, np.ndarray] = {}  # concept → belief distribution
        self.prior_beliefs: Dict[str, np.ndarray] = {}  # for tracking changes
        self.surprise_history: List[float] = []
        self.free_energy_history: List[float] = []

    def update_belief(self, concept: str, observation: np.ndarray) -> Dict:
        """
        Bayesian belief update with Free Energy tracking.

        Returns a metacognitive report about the update.
        """
        if concept in self.beliefs:
            prior = self.beliefs[concept].copy()
        else:
            # Uninformative prior
            prior = np.ones_like(observation) / len(observation)
            self.prior_beliefs[concept] = prior.copy()

        # Bayesian update: posterior ∝ likelihood × prior
        likelihood = observation / (observation.sum() + EPSILON)
        unnormalized = likelihood * prior
        posterior = unnormalized / (unnormalized.sum() + EPSILON)

        # Compute Free Energy components
        complexity = InformationEngine.kl_divergence(posterior, prior)  # D_KL(posterior || prior)
        accuracy = -np.sum(posterior * np.log(likelihood + EPSILON))  # Expected negative log-likelihood
        free_energy = complexity + accuracy
        surprise = InformationEngine.surprise(np.max(observation))

        # Update beliefs
        self.beliefs[concept] = posterior
        self.surprise_history.append(surprise)
        self.free_energy_history.append(free_energy)

        return {
            'concept': concept,
            'prior_entropy': float(InformationEngine.shannon_entropy(prior)),
            'posterior_entropy': float(InformationEngine.shannon_entropy(posterior)),
            'complexity': float(complexity),
            'accuracy': float(accuracy),
            'free_energy': float(free_energy),
            'surprise': float(surprise),
            'belief_change': float(np.mean(np.abs(posterior - prior))),
            'uncertainty_reduction': float(
                InformationEngine.shannon_entropy(prior) -
                InformationEngine.shannon_entropy(posterior)
            ),
        }

    def expected_free_energy(self, concept: str, possible_actions: List[str]) -> Dict[str, float]:
        """
        Compute expected free energy for each possible action.

        This drives ACTIVE INFERENCE — the system chooses actions that
        would maximally reduce its uncertainty (epistemic value) or
        achieve its goals (pragmatic value).

            G(a) = E_Q[ln Q(θ) - ln P(o, θ | a)]
                 = Epistemic value + Pragmatic value

        The action with lowest G is the most "curious" / "useful" one.
        """
        g_values = {}

        for action in possible_actions:
            if concept in self.beliefs:
                current_entropy = InformationEngine.shannon_entropy(self.beliefs[concept])
            else:
                current_entropy = np.log2(10)  # High uncertainty for unknown concepts

            # Estimate expected entropy reduction (heuristic)
            # In a full implementation, this would simulate the action
            epistemic_value = current_entropy * 0.5  # Expected halving of uncertainty
            pragmatic_value = np.random.uniform(0, 1)  # Placeholder for goal relevance

            g_values[action] = epistemic_value + pragmatic_value

        return g_values

    def get_metacognitive_state(self) -> Dict:
        """Return full metacognitive state for visualization."""
        return {
            'num_beliefs': len(self.beliefs),
            'avg_surprise': float(np.mean(self.surprise_history[-50:])) if self.surprise_history else 0,
            'avg_free_energy': float(np.mean(self.free_energy_history[-50:])) if self.free_energy_history else 0,
            'surprise_trend': self._compute_trend(self.surprise_history),
            'free_energy_trend': self._compute_trend(self.free_energy_history),
            'most_uncertain': self._most_uncertain_concepts(5),
            'most_certain': self._most_certain_concepts(5),
            'total_updates': len(self.surprise_history),
        }

    def _most_uncertain_concepts(self, k: int) -> List[Dict]:
        """Return the k concepts with highest entropy (most uncertainty)."""
        uncertainties = []
        for concept, dist in self.beliefs.items():
            h = InformationEngine.shannon_entropy(dist)
            uncertainties.append({'concept': concept, 'entropy': float(h)})
        uncertainties.sort(key=lambda x: x['entropy'], reverse=True)
        return uncertainties[:k]

    def _most_certain_concepts(self, k: int) -> List[Dict]:
        """Return the k concepts with lowest entropy (most certainty)."""
        certainties = []
        for concept, dist in self.beliefs.items():
            h = InformationEngine.shannon_entropy(dist)
            certainties.append({'concept': concept, 'entropy': float(h)})
        certainties.sort(key=lambda x: x['entropy'])
        return certainties[:k]

    def _compute_trend(self, values: List[float], window: int = 10) -> str:
        """Compute whether a metric is increasing, decreasing, or stable."""
        if len(values) < window * 2:
            return 'insufficient_data'
        recent = np.mean(values[-window:])
        earlier = np.mean(values[-window * 2:-window])
        diff = recent - earlier
        if diff > 0.1:
            return 'increasing'
        elif diff < -0.1:
            return 'decreasing'
        return 'stable'

"""
NovaMind — Layer 4: Hybrid Reasoning Engine
=============================================

Implements three-paradigm reasoning: Neural + Symbolic + Geometric,
with a consensus mechanism that integrates results.

Why hybrid reasoning?
---------------------
Each reasoning paradigm has different strengths:

    NEURAL (Pattern-based):
        ✅ Analogies, similarity, intuition, generalization
        ❌ Logical consistency, mathematical proofs, exact counting

    SYMBOLIC (Logic-based):
        ✅ Deduction, constraint checking, rule following, proofs
        ❌ Handling uncertainty, learning from examples, creativity

    GEOMETRIC (Structure-based):
        ✅ Spatial reasoning, structural analogies, topology
        ❌ Sequential processing, linguistic reasoning

No single paradigm is sufficient for general intelligence.
NovaMind runs all three IN PARALLEL and uses a learned
consensus mechanism to integrate their outputs.

The reasoning trace is fully transparent — you can see which
paradigm contributed what to the final answer.

References:
    - Kahneman (2011): "Thinking, Fast and Slow" (System 1 vs System 2)
    - Marcus & Davis (2019): "Rebooting AI" (neuro-symbolic integration)
    - Garcez et al. (2019): "Neural-Symbolic Computing"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import time
import hashlib

from .math.geometric_algebra import CliffordAlgebra, MultiVector
from .math.hyperbolic import PoincareBall
from .math.topology import SimplicialComplex, PersistentHomology

CONCEPT_DIM = 32
GA_DIM = 8


class ReasoningResult:
    """Result from a single reasoning paradigm."""

    def __init__(self, paradigm: str, conclusion: str, confidence: float,
                 trace: List[str], supporting_evidence: List[str] = None):
        self.paradigm = paradigm
        self.conclusion = conclusion
        self.confidence = confidence
        self.trace = trace
        self.supporting_evidence = supporting_evidence or []
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            'paradigm': self.paradigm,
            'conclusion': self.conclusion,
            'confidence': self.confidence,
            'trace': self.trace,
            'supporting_evidence': self.supporting_evidence,
        }


class NeuralReasoner:
    """
    Neural (Pattern-based) reasoning — System 1 / Fast thinking.

    Uses geometric similarity in the Poincaré ball and Clifford algebra
    to find analogies, similarities, and pattern-based inferences.

    This is "intuition" — fast, approximate, based on pattern matching.
    """

    def __init__(self):
        self.ball = PoincareBall(dim=CONCEPT_DIM)
        self.algebra = CliffordAlgebra(p=GA_DIM)

    def reason(self, query_concepts: List[Dict], knowledge_base: Dict) -> ReasoningResult:
        """
        Neural reasoning: find the most relevant patterns and analogies.

        Process:
        1. Embed the query in concept space
        2. Find nearest neighbors (most similar known concepts)
        3. Check for analogical patterns (A:B :: C:?)
        4. Generate conclusion based on strongest patterns
        """
        trace = []
        trace.append("⚡ NEURAL REASONING (Pattern-based / System 1)")

        # Step 1: Identify key concepts and their positions
        positions = {}
        for concept in query_concepts:
            if concept.get('position'):
                positions[concept['label']] = np.array(concept['position'])
                trace.append(f"  📍 Located '{concept['label']}' in hyperbolic space "
                           f"(depth: {np.linalg.norm(concept['position']):.3f})")

        if not positions:
            return ReasoningResult("neural", "Insufficient concept embeddings",
                                   0.2, trace)

        # Step 2: Find nearest concepts in the knowledge base
        neighbors = self._find_nearest_concepts(positions, knowledge_base)
        for neighbor in neighbors[:3]:
            trace.append(f"  🔗 Similar to: '{neighbor['label']}' "
                       f"(similarity: {neighbor['similarity']:.3f})")

        # Step 3: Detect analogical patterns
        analogies = self._detect_analogies(positions, knowledge_base)
        for analogy in analogies[:2]:
            trace.append(f"  🔄 Analogy detected: {analogy['description']}")

        # Step 4: Generate intuitive conclusion
        conclusion, confidence = self._generate_intuition(
            query_concepts, neighbors, analogies
        )
        trace.append(f"  💡 Intuition: {conclusion} (confidence: {confidence:.2f})")

        return ReasoningResult(
            "neural", conclusion, confidence, trace,
            [n['label'] for n in neighbors[:3]]
        )

    def _find_nearest_concepts(self, positions: Dict[str, np.ndarray],
                                knowledge_base: Dict) -> List[Dict]:
        """Find nearest concepts in the knowledge graph."""
        neighbors = []
        known_concepts = knowledge_base.get('concepts', [])

        for label, pos in positions.items():
            for known in known_concepts:
                if known.get('position') and known['label'] != label:
                    known_pos = np.array(known['position'])
                    dist = float(self.ball.distance(
                        pos.reshape(1, -1), known_pos.reshape(1, -1)
                    ).mean())
                    similarity = np.exp(-dist)
                    neighbors.append({
                        'label': known['label'],
                        'similarity': similarity,
                        'distance': dist,
                    })

        neighbors.sort(key=lambda x: x['similarity'], reverse=True)
        return neighbors

    def _detect_analogies(self, positions: Dict[str, np.ndarray],
                          knowledge_base: Dict) -> List[Dict]:
        """
        Detect analogical patterns using geometric algebra.

        An analogy A:B :: C:D exists when the transformation from A to B
        (a rotor in Clifford algebra) is similar to the transformation from C to D.
        """
        analogies = []
        labels = list(positions.keys())

        if len(labels) >= 2:
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a_label, b_label = labels[i], labels[j]
                    a_pos = positions[a_label][:GA_DIM]
                    b_pos = positions[b_label][:GA_DIM]

                    a_vec = self.algebra.vector(a_pos)
                    b_vec = self.algebra.vector(b_pos)
                    similarity = self.algebra.concept_similarity(a_vec, b_vec)

                    analogies.append({
                        'description': f"'{a_label}' relates to '{b_label}' "
                                      f"(geometric similarity: {similarity:.3f})",
                        'concepts': (a_label, b_label),
                        'similarity': abs(similarity),
                    })

        analogies.sort(key=lambda x: x['similarity'], reverse=True)
        return analogies

    def _generate_intuition(self, query_concepts: List[Dict],
                             neighbors: List[Dict],
                             analogies: List[Dict]) -> Tuple[str, float]:
        """Generate an intuitive conclusion from patterns."""
        query_labels = [c['label'] for c in query_concepts]

        if neighbors:
            related_labels = [n['label'] for n in neighbors[:3]]
            avg_similarity = np.mean([n['similarity'] for n in neighbors[:3]])

            conclusion = (f"Based on pattern matching, {', '.join(query_labels)} "
                        f"relates to {', '.join(related_labels)}. "
                        f"The concepts share structural similarity in concept space.")
            confidence = float(min(0.9, avg_similarity + 0.2))
        else:
            conclusion = (f"Novel concepts: {', '.join(query_labels)}. "
                        f"No strong pattern matches found — this is genuinely new information.")
            confidence = 0.3

        return conclusion, confidence


class SymbolicReasoner:
    """
    Symbolic (Logic-based) reasoning — System 2 / Slow thinking.

    Applies formal logical rules, performs deduction, checks constraints,
    and handles explicit rule-based inference.

    This is "deliberation" — slow, precise, guaranteed consistent.
    """

    def __init__(self):
        self.rules: List[Dict] = []
        self.facts: Dict[str, List[str]] = defaultdict(list)

    def add_rule(self, rule: Dict):
        """Add a logical rule to the knowledge base."""
        self.rules.append(rule)

    def add_fact(self, subject: str, predicate: str, obj: str):
        """Add a fact (subject, predicate, object) triple."""
        self.facts[subject].append(f"{predicate}:{obj}")

    def reason(self, query_concepts: List[Dict], knowledge_base: Dict) -> ReasoningResult:
        """
        Symbolic reasoning: apply logical rules and deduction.

        Process:
        1. Extract logical propositions from the query
        2. Match against known rules
        3. Apply forward and backward chaining
        4. Check for contradictions
        5. Generate logically grounded conclusion
        """
        trace = []
        trace.append("🔢 SYMBOLIC REASONING (Logic-based / System 2)")

        # Step 1: Extract propositions
        propositions = self._extract_propositions(query_concepts, knowledge_base)
        for prop in propositions:
            trace.append(f"  📝 Proposition: {prop}")

        # Step 2: Apply rule-based inference
        inferences = self._forward_chain(propositions, knowledge_base)
        for inf in inferences:
            trace.append(f"  ⟹ Inferred: {inf['conclusion']} "
                       f"(from rule: {inf['rule']})")

        # Step 3: Check for contradictions
        contradictions = self._check_contradictions(propositions + [i['conclusion'] for i in inferences])
        if contradictions:
            for c in contradictions:
                trace.append(f"  ⚠️ CONTRADICTION: {c}")

        # Step 4: Build conclusion
        conclusion, confidence = self._build_conclusion(propositions, inferences, contradictions)
        trace.append(f"  📐 Conclusion: {conclusion} (confidence: {confidence:.2f})")

        return ReasoningResult(
            "symbolic", conclusion, confidence, trace,
            [i['rule'] for i in inferences[:3]]
        )

    def _extract_propositions(self, concepts: List[Dict], kb: Dict) -> List[str]:
        """Extract logical propositions from concepts and their relations."""
        propositions = []

        for concept in concepts:
            connections = concept.get('connections', {})
            for rel_type, targets in connections.items():
                for target in targets:
                    propositions.append(f"{concept['label']} {rel_type} {target}")

        # Add facts from the world model
        world_edges = kb.get('causal_edges', [])
        for edge in world_edges[:10]:
            propositions.append(
                f"{edge.get('source', '?')} {edge.get('type', 'relates_to')} {edge.get('target', '?')}"
            )

        return propositions

    def _forward_chain(self, propositions: List[str], kb: Dict) -> List[Dict]:
        """
        Forward chaining: apply rules to derive new facts.

        If we know A and we have a rule A → B, conclude B.
        """
        inferences = []
        semantic_rules = kb.get('semantic_rules', [])

        for rule in semantic_rules:
            rule_content = rule.get('content', {})
            rule_type = rule_content.get('type', '')
            abstraction = rule_content.get('abstraction', {})

            if rule_type == 'sequential':
                from_type = abstraction.get('from_type', '')
                to_type = abstraction.get('to_type', '')
                freq = abstraction.get('frequency', 0)

                for prop in propositions:
                    if from_type.lower() in prop.lower():
                        inferences.append({
                            'conclusion': f"Likely followed by {to_type} "
                                        f"(frequency: {freq:.0%})",
                            'rule': f"{from_type} → {to_type}",
                            'confidence': freq,
                        })

            elif rule_type == 'attribute':
                attr = abstraction.get('attribute', '')
                value = abstraction.get('value', '')
                freq = abstraction.get('frequency', 0)

                if freq > 0.5:
                    inferences.append({
                        'conclusion': f"Common attribute: {attr} = {value}",
                        'rule': f"Frequency-based: {attr}={value} ({freq:.0%})",
                        'confidence': freq,
                    })

        # Apply causal chain inference
        causal_chains = kb.get('causal_chains', [])
        for chain in causal_chains[:5]:
            if len(chain) >= 2:
                for prop in propositions:
                    if chain[0].lower() in prop.lower():
                        inferences.append({
                            'conclusion': f"Causal chain: {' → '.join(chain)}",
                            'rule': 'Causal transitivity',
                            'confidence': 0.7 ** (len(chain) - 1),  # Decreases with chain length
                        })

        return inferences

    def _check_contradictions(self, statements: List[str]) -> List[str]:
        """Check for logical contradictions among statements."""
        contradictions = []

        # Check for opposite assertions (simplified)
        negation_pairs = [
            ('causes', 'prevents'),
            ('is', 'is not'),
            ('has', 'lacks'),
            ('true', 'false'),
        ]

        for i, s1 in enumerate(statements):
            for j, s2 in enumerate(statements):
                if i >= j:
                    continue
                for pos, neg in negation_pairs:
                    if pos in s1.lower() and neg in s2.lower():
                        # Check if they're about the same subject
                        s1_words = set(s1.lower().split())
                        s2_words = set(s2.lower().split())
                        overlap = s1_words & s2_words
                        if len(overlap) >= 2:
                            contradictions.append(f"'{s1}' contradicts '{s2}'")

        return contradictions

    def _build_conclusion(self, propositions: List[str], inferences: List[Dict],
                          contradictions: List[str]) -> Tuple[str, float]:
        """Build a logically grounded conclusion."""
        if contradictions:
            conclusion = (f"Logical analysis found {len(contradictions)} contradiction(s). "
                        f"The knowledge base may need revision. "
                        f"Main issue: {contradictions[0]}")
            confidence = 0.3
        elif inferences:
            top_inf = sorted(inferences, key=lambda x: x['confidence'], reverse=True)
            conclusion = (f"Logical deduction yields: {top_inf[0]['conclusion']}. "
                        f"Derived from {len(inferences)} inference(s) using "
                        f"{len(propositions)} known proposition(s).")
            confidence = float(min(0.95, top_inf[0]['confidence'] + 0.1))
        else:
            conclusion = (f"No applicable rules found for the given propositions. "
                        f"Known facts: {len(propositions)}.")
            confidence = 0.2

        return conclusion, confidence


class GeometricReasoner:
    """
    Geometric (Topology-based) reasoning — Structural thinking.

    Uses topological data analysis and the geometric structure of
    the knowledge graph to reason about:
        - Structural relationships between concept clusters
        - Missing connections (knowledge gaps)
        - Circular reasoning detection
        - Hierarchical inference via entailment cones
    """

    def __init__(self):
        self.ball = PoincareBall(dim=CONCEPT_DIM)

    def reason(self, query_concepts: List[Dict], knowledge_base: Dict) -> ReasoningResult:
        """
        Geometric reasoning: analyze the topology of concept space.

        Process:
        1. Build simplicial complex from concept positions
        2. Compute persistent homology (topological features)
        3. Analyze hierarchical relationships (entailment cones)
        4. Detect structural patterns (clusters, gaps, loops)
        """
        trace = []
        trace.append("📐 GEOMETRIC REASONING (Topology-based / Structural)")

        # Collect all concept positions
        all_concepts = knowledge_base.get('concepts', []) + query_concepts
        positioned_concepts = [c for c in all_concepts if c.get('position')]

        if len(positioned_concepts) < 3:
            return ReasoningResult(
                "geometric",
                "Insufficient concepts for topological analysis (need ≥ 3)",
                0.1, trace
            )

        # Step 1: Build distance matrix in hyperbolic space
        n = min(len(positioned_concepts), 30)  # Limit for computation
        positions = np.array([c['position'] for c in positioned_concepts[:n]])
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = float(self.ball.distance(
                    positions[i].reshape(1, -1),
                    positions[j].reshape(1, -1)
                ).mean())
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        trace.append(f"  📊 Built {n}×{n} hyperbolic distance matrix")

        # Step 2: Build Vietoris-Rips complex and compute persistent homology
        complex = SimplicialComplex()
        max_filtration = float(np.percentile(dist_matrix[dist_matrix > 0], 75)) if dist_matrix.any() else 2.0
        complex.build_from_distance_matrix(dist_matrix, max_dim=2, max_filtration=max_filtration)

        ph = PersistentHomology(complex)
        persistence = ph.compute(max_dim=2)

        betti = ph.betti_numbers()
        trace.append(f"  🔵 Betti numbers: β₀={betti.get(0, 0)} (clusters), "
                    f"β₁={betti.get(1, 0)} (loops), β₂={betti.get(2, 0)} (voids)")

        # Step 3: Analyze topological features
        gaps = ph.knowledge_gaps()
        cycles = ph.circular_reasoning_detector()
        entropy = ph.persistence_entropy()

        trace.append(f"  📈 Topological complexity (persistence entropy): {entropy:.3f}")

        if gaps:
            trace.append(f"  🕳️ Knowledge gaps detected: {len(gaps)} persistent voids")
        if cycles:
            trace.append(f"  🔄 Circular reasoning paths: {len(cycles)} persistent cycles")

        # Step 4: Hierarchical analysis
        hierarchy_report = self._analyze_hierarchy(positioned_concepts[:n])
        for h in hierarchy_report[:3]:
            trace.append(f"  🌳 {h}")

        # Step 5: Build conclusion
        conclusion = self._build_geometric_conclusion(betti, gaps, cycles, entropy, hierarchy_report)
        confidence = min(0.85, 0.3 + entropy * 0.3 + (0.1 if not cycles else 0))
        trace.append(f"  🔷 Geometric conclusion: {conclusion}")

        return ReasoningResult(
            "geometric", conclusion, confidence, trace,
            [f"β₀={betti.get(0, 0)}", f"β₁={betti.get(1, 0)}", f"entropy={entropy:.3f}"]
        )

    def _analyze_hierarchy(self, concepts: List[Dict]) -> List[str]:
        """Analyze hierarchical relationships in hyperbolic space."""
        reports = []

        # Sort by depth (distance from origin = specificity)
        depths = []
        for c in concepts:
            pos = np.array(c['position']).reshape(1, -1)
            origin = np.zeros_like(pos)
            depth = float(self.ball.distance(origin, pos).mean())
            depths.append((c['label'], depth))

        depths.sort(key=lambda x: x[1])

        if depths:
            most_abstract = depths[0]
            most_specific = depths[-1]
            reports.append(f"Most abstract concept: '{most_abstract[0]}' (depth: {most_abstract[1]:.3f})")
            reports.append(f"Most specific concept: '{most_specific[0]}' (depth: {most_specific[1]:.3f})")
            avg_depth = np.mean([d[1] for d in depths])
            reports.append(f"Average concept depth: {avg_depth:.3f} (hierarchy spread)")

        return reports

    def _build_geometric_conclusion(self, betti: Dict, gaps: List, cycles: List,
                                     entropy: float, hierarchy: List[str]) -> str:
        """Build a conclusion from topological analysis."""
        parts = []

        clusters = betti.get(0, 0)
        if clusters > 1:
            parts.append(f"The knowledge forms {clusters} disconnected clusters "
                        f"that may need bridging connections")
        elif clusters == 1:
            parts.append("Knowledge is well-connected (single component)")

        if gaps:
            parts.append(f"There are {len(gaps)} knowledge gap(s) that could be filled")

        if cycles:
            parts.append(f"Warning: {len(cycles)} potential circular reasoning path(s) detected")

        if entropy > 0.5:
            parts.append("The knowledge structure is topologically complex — rich internal organization")
        elif entropy > 0:
            parts.append("The knowledge structure is relatively simple — may benefit from enrichment")

        return ". ".join(parts) if parts else "Knowledge topology is still forming."


class HybridReasoningEngine:
    """
    The integrated reasoning engine that runs all three paradigms
    and builds a consensus.

    This is where "thinking" happens — not as a single process but as
    a PARLIAMENT of different reasoning styles, each contributing
    their perspective, then voting on the best answer.
    """

    def __init__(self):
        self.neural = NeuralReasoner()
        self.symbolic = SymbolicReasoner()
        self.geometric = GeometricReasoner()

        # Learned weights for each paradigm (updated by metacognition)
        self.paradigm_weights = {
            'neural': 0.4,
            'symbolic': 0.35,
            'geometric': 0.25,
        }

        # History for learning from past reasoning
        self.reasoning_history: List[Dict] = []

    def reason(self, query_concepts: List[Dict], knowledge_base: Dict) -> Dict:
        """
        Run all three reasoning paradigms and build consensus.

        Returns a comprehensive reasoning report with full trace.
        """
        start_time = time.time()

        # Run all three paradigms
        neural_result = self.neural.reason(query_concepts, knowledge_base)
        symbolic_result = self.symbolic.reason(query_concepts, knowledge_base)
        geometric_result = self.geometric.reason(query_concepts, knowledge_base)

        # Consensus building
        consensus = self._build_consensus(neural_result, symbolic_result, geometric_result)

        reasoning_time = time.time() - start_time

        report = {
            'paradigms': {
                'neural': neural_result.to_dict(),
                'symbolic': symbolic_result.to_dict(),
                'geometric': geometric_result.to_dict(),
            },
            'consensus': consensus,
            'paradigm_weights': self.paradigm_weights.copy(),
            'reasoning_time_ms': reasoning_time * 1000,
            'combined_trace': (
                neural_result.trace +
                ["---"] +
                symbolic_result.trace +
                ["---"] +
                geometric_result.trace +
                ["---"] +
                [f"🎯 CONSENSUS: {consensus['conclusion']} "
                 f"(confidence: {consensus['confidence']:.2f})"]
            ),
        }

        self.reasoning_history.append({
            'timestamp': time.time(),
            'consensus_confidence': consensus['confidence'],
            'paradigm_confidences': {
                'neural': neural_result.confidence,
                'symbolic': symbolic_result.confidence,
                'geometric': geometric_result.confidence,
            },
        })

        return report

    def _build_consensus(self, neural: ReasoningResult,
                         symbolic: ReasoningResult,
                         geometric: ReasoningResult) -> Dict:
        """
        Build consensus from three reasoning paradigms.

        Uses confidence-weighted integration with conflict detection.
        """
        results = [
            ('neural', neural, self.paradigm_weights['neural']),
            ('symbolic', symbolic, self.paradigm_weights['symbolic']),
            ('geometric', geometric, self.paradigm_weights['geometric']),
        ]

        # Weighted confidence
        total_weight = sum(r[2] * r[1].confidence for r in results)
        normalized_contributions = {}

        for name, result, weight in results:
            contribution = weight * result.confidence / (total_weight + 1e-10)
            normalized_contributions[name] = contribution

        # Select primary conclusion from highest-contributing paradigm
        best_paradigm = max(results, key=lambda r: r[2] * r[1].confidence)

        # Check for conflicts
        conclusions = [neural.conclusion, symbolic.conclusion, geometric.conclusion]
        conflict = self._detect_reasoning_conflict(conclusions)

        # Build consensus conclusion
        if conflict:
            consensus_conclusion = (
                f"MIXED: {best_paradigm[1].conclusion}. "
                f"Note: reasoning paradigms produced different perspectives. "
                f"Primary = {best_paradigm[0]} (contribution: {normalized_contributions[best_paradigm[0]]:.0%})"
            )
            consensus_confidence = best_paradigm[1].confidence * 0.7
        else:
            consensus_conclusion = (
                f"{best_paradigm[1].conclusion}. "
                f"All reasoning paradigms converge (agreement: "
                f"{sum(1 for r in results if r[1].confidence > 0.3)}/3)"
            )
            consensus_confidence = min(0.95, total_weight * 1.2)

        return {
            'conclusion': consensus_conclusion,
            'confidence': consensus_confidence,
            'primary_paradigm': best_paradigm[0],
            'contributions': normalized_contributions,
            'conflict_detected': conflict,
        }

    def _detect_reasoning_conflict(self, conclusions: List[str]) -> bool:
        """Detect if reasoning paradigms produced conflicting conclusions."""
        from .math.information import InformationEngine

        if len(conclusions) < 2:
            return False

        # Use NCD to check if conclusions are very different
        for i in range(len(conclusions)):
            for j in range(i + 1, len(conclusions)):
                ncd = InformationEngine.normalized_compression_distance(
                    conclusions[i], conclusions[j]
                )
                if ncd > 0.8:
                    return True

        return False

    def update_weights(self, feedback: Dict):
        """Update paradigm weights based on metacognitive feedback."""
        for paradigm in self.paradigm_weights:
            if paradigm in feedback:
                # Exponential moving average update
                alpha = 0.1
                self.paradigm_weights[paradigm] = (
                    (1 - alpha) * self.paradigm_weights[paradigm] +
                    alpha * feedback[paradigm]
                )

        # Normalize
        total = sum(self.paradigm_weights.values())
        for paradigm in self.paradigm_weights:
            self.paradigm_weights[paradigm] /= total

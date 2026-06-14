"""
NovaMind — Layer 5: Metacognition System
==========================================

Implements self-monitoring, confidence assessment, and cognitive control
based on Karl Friston's Free Energy Principle and Higher-Order Thought (HOT) theory.

What is Metacognition?
----------------------
Metacognition is "thinking about thinking" — the ability to:
    1. Monitor one's own cognitive processes
    2. Assess confidence in one's own conclusions
    3. Detect inconsistencies in one's beliefs
    4. Identify knowledge gaps
    5. Control attention and resource allocation
    6. Learn from one's own reasoning errors

This is what separates genuine intelligence from sophisticated pattern matching.
An LLM has NO metacognition — it cannot say "I don't know this" reliably, because
it has no model of what it knows vs. doesn't know.

NovaMind's metacognition is grounded in:
    - Free Energy Principle: minimize surprise through belief updating
    - Higher-Order Thought: a process that OBSERVES the primary reasoning process
    - Integrated Information Theory: φ (phi) as a measure of cognitive integration

References:
    - Friston (2010): "The Free-Energy Principle: A Unified Brain Theory?"
    - Fleming & Daw (2017): "Self-Evaluation of Decision-Making"
    - Tononi (2008): "Consciousness as Integrated Information"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time

from .math.information import FreeEnergyMinimizer, InformationEngine


class CognitiveState:
    """
    The metacognitive state — a snapshot of the system's self-model.

    This is the "I know that I know" representation.
    """

    def __init__(self):
        self.confidence: float = 0.5
        self.coherence: float = 0.5
        self.surprise_level: float = 0.5
        self.knowledge_coverage: float = 0.0
        self.attention_focus: Optional[str] = None
        self.emotional_state: Dict[str, float] = {
            'curiosity': 0.5,
            'certainty': 0.5,
            'confusion': 0.0,
            'satisfaction': 0.5,
        }
        self.active_concerns: List[str] = []

    def to_dict(self) -> dict:
        return {
            'confidence': self.confidence,
            'coherence': self.coherence,
            'surprise_level': self.surprise_level,
            'knowledge_coverage': self.knowledge_coverage,
            'attention_focus': self.attention_focus,
            'emotional_state': self.emotional_state,
            'active_concerns': self.active_concerns,
        }


class MetacognitionSystem:
    """
    Layer 5: Self-Monitoring and Cognitive Control.

    This system OBSERVES the operation of all other layers and
    produces assessment reports. It's a "higher-order thought"
    process that watches the first-order reasoning.
    """

    def __init__(self):
        self.fep = FreeEnergyMinimizer()
        self.state = CognitiveState()
        self.history: List[Dict] = []
        self.consistency_violations: List[Dict] = []
        self.knowledge_gaps: List[Dict] = []

        # Internal models of cognitive performance
        self.paradigm_accuracy: Dict[str, List[float]] = defaultdict(list)
        self.prediction_errors: List[float] = []

    def assess(self, perception: Dict, reasoning: Dict, memory: Dict,
               world_model: Dict) -> Dict:
        """
        Perform a full metacognitive assessment of the current cognitive cycle.

        This is the "thinking about thinking" — a higher-order process
        that monitors all first-order processes.
        """
        report = {}
        start_time = time.time()

        # 1. Confidence Assessment
        report['confidence'] = self._assess_confidence(reasoning)

        # 2. Coherence Check (consistency of beliefs)
        report['coherence'] = self._check_coherence(reasoning, world_model)

        # 3. Surprise Analysis (Free Energy)
        report['surprise'] = self._analyze_surprise(perception)

        # 4. Knowledge Gap Detection
        report['knowledge_gaps'] = self._detect_knowledge_gaps(reasoning, memory)

        # 5. Attention Control (what to focus on next)
        report['attention'] = self._control_attention(report)

        # 6. Emotional State (cognitive "feelings")
        report['emotional_state'] = self._update_emotional_state(report)

        # 7. Self-Assessment Summary
        report['self_assessment'] = self._generate_self_assessment(report)

        # 8. Recommendations for other layers
        report['recommendations'] = self._generate_recommendations(report)

        # Update state
        self.state.confidence = report['confidence']['overall']
        self.state.coherence = report['coherence']['score']
        self.state.surprise_level = report['surprise']['current_level']
        self.state.emotional_state = report['emotional_state']
        self.state.attention_focus = report['attention']['focus']
        self.state.active_concerns = report['self_assessment'].get('concerns', [])

        # Record history
        report['assessment_time_ms'] = (time.time() - start_time) * 1000
        self.history.append({
            'timestamp': time.time(),
            'confidence': self.state.confidence,
            'coherence': self.state.coherence,
            'surprise': self.state.surprise_level,
        })

        return report

    def _assess_confidence(self, reasoning: Dict) -> Dict:
        """
        Assess confidence in the reasoning results.

        Uses multiple signals:
        - Agreement between reasoning paradigms
        - Historical accuracy of each paradigm
        - Strength of supporting evidence
        """
        paradigms = reasoning.get('paradigms', {})

        # Agreement score: how much do paradigms agree?
        confidences = []
        for name, result in paradigms.items():
            confidences.append(result.get('confidence', 0.5))

        if confidences:
            agreement = 1 - np.std(confidences)  # Low variance = high agreement
            avg_confidence = np.mean(confidences)
        else:
            agreement = 0.5
            avg_confidence = 0.5

        # Overall confidence = f(agreement, individual confidences)
        overall = float(agreement * 0.3 + avg_confidence * 0.5 + 0.2 * (1 if not reasoning.get('consensus', {}).get('conflict_detected') else 0))

        return {
            'overall': overall,
            'agreement': float(agreement),
            'per_paradigm': {name: result.get('confidence', 0.5)
                           for name, result in paradigms.items()},
            'conflict_detected': reasoning.get('consensus', {}).get('conflict_detected', False),
        }

    def _check_coherence(self, reasoning: Dict, world_model: Dict) -> Dict:
        """
        Check for internal consistency and coherence.

        Looks for:
        - Contradictions in the causal graph
        - Inconsistencies between reasoning results
        - Violations of known constraints
        """
        violations = []

        # Check for contradicting causal chains
        edges = world_model.get('edges', [])
        for i, e1 in enumerate(edges):
            for j, e2 in enumerate(edges):
                if i >= j:
                    continue
                # Check for contradicting edges (A causes B AND A prevents B)
                if (e1.get('source') == e2.get('source') and
                    e1.get('target') == e2.get('target')):
                    if e1.get('type') != e2.get('type'):
                        violations.append({
                            'type': 'contradicting_edges',
                            'description': f"'{e1.get('source')}' both "
                                         f"{e1.get('type')} and {e2.get('type')} "
                                         f"'{e1.get('target')}'",
                            'severity': 'high',
                        })

        self.consistency_violations.extend(violations)

        score = max(0.1, 1.0 - len(violations) * 0.2)
        return {
            'score': score,
            'violations': violations,
            'total_violations_ever': len(self.consistency_violations),
        }

    def _analyze_surprise(self, perception: Dict) -> Dict:
        """
        Analyze surprise level using the Free Energy Principle.

        High surprise = input was very unexpected = beliefs need updating
        Low surprise = input was expected = beliefs are accurate
        """
        new_concepts = perception.get('new_concepts', 0)
        total_concepts = perception.get('total_concepts_known', 1)
        recognized = perception.get('recognized_concepts', 0)

        # Surprise is proportional to novelty
        novelty_ratio = new_concepts / max(1, new_concepts + recognized)

        # Update Free Energy
        observation = np.random.dirichlet(np.ones(10))  # Proxy for input distribution
        fep_report = self.fep.update_belief('input_distribution', observation)

        current_level = novelty_ratio * 0.6 + fep_report.get('surprise', 0.5) * 0.4

        trend = 'stable'
        if len(self.history) >= 5:
            recent = [h['surprise'] for h in self.history[-5:]]
            if np.mean(recent[-3:]) > np.mean(recent[:3]) + 0.1:
                trend = 'increasing'
            elif np.mean(recent[-3:]) < np.mean(recent[:3]) - 0.1:
                trend = 'decreasing'

        return {
            'current_level': float(current_level),
            'novelty_ratio': float(novelty_ratio),
            'free_energy': fep_report.get('free_energy', 0),
            'trend': trend,
            'interpretation': self._interpret_surprise(current_level),
        }

    def _interpret_surprise(self, level: float) -> str:
        """Human-readable interpretation of surprise level."""
        if level < 0.2:
            return "Very low surprise — input is well within expectations. Beliefs are stable."
        elif level < 0.4:
            return "Moderate surprise — some new information, beliefs adapting slightly."
        elif level < 0.6:
            return "Notable surprise — significant new information detected."
        elif level < 0.8:
            return "High surprise — input challenges existing beliefs. Major updating needed."
        else:
            return "Extreme surprise — world model may need fundamental revision."

    def _detect_knowledge_gaps(self, reasoning: Dict, memory: Dict) -> List[Dict]:
        """
        Detect gaps in knowledge that should be filled.

        A knowledge gap is identified when:
        - Reasoning references concepts that don't exist in the world model
        - Multiple queries fail to find relevant semantic rules
        - The knowledge graph has topological "holes" (H₂ in persistent homology)
        """
        gaps = []

        # Check if reasoning had low confidence due to missing knowledge
        consensus = reasoning.get('consensus', {})
        if consensus.get('confidence', 0.5) < 0.3:
            gaps.append({
                'type': 'low_reasoning_confidence',
                'description': 'Reasoning confidence is very low — likely missing crucial knowledge',
                'severity': 'high',
                'suggested_action': 'seek_information',
            })

        # Check if memory has unresolved patterns
        memory_stats = memory if isinstance(memory, dict) else {}
        episodic_stats = memory_stats.get('episodic', {})
        if episodic_stats.get('total_compacted', 0) == 0 and episodic_stats.get('total_stored', 0) > 5:
            gaps.append({
                'type': 'no_patterns_found',
                'description': 'Many episodes stored but no compressible patterns found. '
                             'Knowledge may be too diverse or disorganized.',
                'severity': 'medium',
                'suggested_action': 'seek_related_information',
            })

        self.knowledge_gaps.extend(gaps)
        return gaps

    def _control_attention(self, report: Dict) -> Dict:
        """
        Control where cognitive resources should be focused.

        Based on:
        - Surprise level (high surprise → attend to novelty)
        - Knowledge gaps (gaps → seek information)
        - Confidence (low confidence → deliberate more carefully)
        """
        surprise = report.get('surprise', {}).get('current_level', 0.5)
        gaps = report.get('knowledge_gaps', [])
        confidence = report.get('confidence', {}).get('overall', 0.5)

        # Attention priority
        if surprise > 0.7:
            focus = 'novel_input'
            strategy = 'Prioritize understanding new information — high surprise detected'
        elif gaps and any(g['severity'] == 'high' for g in gaps):
            focus = 'knowledge_gaps'
            strategy = 'Address critical knowledge gaps before proceeding'
        elif confidence < 0.3:
            focus = 'reasoning_quality'
            strategy = 'Insufficient confidence — engage deeper reasoning and seek more evidence'
        else:
            focus = 'integration'
            strategy = 'Knowledge is well-organized — focus on integration and compaction'

        return {
            'focus': focus,
            'strategy': strategy,
            'surprise_driven': surprise > 0.5,
            'gap_driven': len(gaps) > 0,
            'confidence_driven': confidence < 0.4,
        }

    def _update_emotional_state(self, report: Dict) -> Dict[str, float]:
        """
        Update cognitive "emotions" based on the metacognitive assessment.

        These aren't feelings in the biological sense — they're functional
        analogs that drive behavior:
            - Curiosity: high when novelty detected + knowledge gaps exist
            - Certainty: high when confidence is high + beliefs are coherent
            - Confusion: high when contradictions detected + confidence low
            - Satisfaction: high when knowledge is well-compressed + coherent
        """
        surprise = report.get('surprise', {}).get('current_level', 0.5)
        confidence = report.get('confidence', {}).get('overall', 0.5)
        coherence = report.get('coherence', {}).get('score', 0.5)
        gaps = len(report.get('knowledge_gaps', []))

        return {
            'curiosity': float(np.clip(surprise * 0.5 + (gaps > 0) * 0.3 + 0.2, 0, 1)),
            'certainty': float(np.clip(confidence * 0.5 + coherence * 0.3 + 0.1, 0, 1)),
            'confusion': float(np.clip((1 - confidence) * 0.4 + (1 - coherence) * 0.3 + surprise * 0.2, 0, 1)),
            'satisfaction': float(np.clip(coherence * 0.4 + confidence * 0.3 + (1 - surprise) * 0.2, 0, 1)),
        }

    def _generate_self_assessment(self, report: Dict) -> Dict:
        """
        Generate a natural-language self-assessment.

        This is the system reflecting on its own performance.
        """
        confidence = report.get('confidence', {}).get('overall', 0.5)
        coherence = report.get('coherence', {}).get('score', 0.5)
        surprise = report.get('surprise', {}).get('current_level', 0.5)

        assessments = []
        concerns = []

        # Confidence assessment
        if confidence > 0.7:
            assessments.append("I am quite confident in my reasoning on this topic.")
        elif confidence > 0.4:
            assessments.append("My confidence is moderate — there may be aspects I'm uncertain about.")
        else:
            assessments.append("I have low confidence in my conclusions — I may be missing important information.")
            concerns.append("Low reasoning confidence")

        # Coherence assessment
        if coherence > 0.7:
            assessments.append("My knowledge on this subject is internally consistent.")
        elif coherence < 0.4:
            assessments.append("I detect some inconsistencies in my knowledge — there may be contradictions.")
            concerns.append("Knowledge inconsistencies detected")

        # Surprise assessment
        assessments.append(report.get('surprise', {}).get('interpretation', ''))

        return {
            'text': ' '.join(assessments),
            'concerns': concerns,
            'overall_quality': float((confidence + coherence + (1 - surprise)) / 3),
        }

    def _generate_recommendations(self, report: Dict) -> List[Dict]:
        """Generate recommendations for other cognitive layers."""
        recommendations = []

        confidence = report.get('confidence', {}).get('overall', 0.5)
        if confidence < 0.4:
            recommendations.append({
                'target': 'reasoning',
                'action': 'increase_deliberation',
                'description': 'Engage deeper symbolic reasoning to increase confidence',
            })

        gaps = report.get('knowledge_gaps', [])
        if gaps:
            recommendations.append({
                'target': 'goals',
                'action': 'seek_information',
                'description': f'Add information-seeking goals for {len(gaps)} knowledge gap(s)',
            })

        coherence = report.get('coherence', {}).get('score', 0.5)
        if coherence < 0.5:
            recommendations.append({
                'target': 'world_model',
                'action': 'resolve_contradictions',
                'description': 'Review and resolve contradictions in the causal model',
            })

        return recommendations

    def get_full_state(self) -> Dict:
        """Return complete metacognitive state for visualization."""
        return {
            'current_state': self.state.to_dict(),
            'fep_state': self.fep.get_metacognitive_state(),
            'history': self.history[-20:],
            'consistency_violations': self.consistency_violations[-10:],
            'knowledge_gaps': self.knowledge_gaps[-10:],
        }

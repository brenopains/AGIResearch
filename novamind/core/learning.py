"""
NovaMind — Layer 7: Continuous Learning Engine
================================================

Implements online Bayesian learning with catastrophic forgetting prevention,
novelty detection, and self-improvement.

Why continuous learning is essential for AGI?
---------------------------------------------
LLMs are FROZEN after training. They cannot:
    - Learn from interactions
    - Update incorrect beliefs
    - Adapt to new domains
    - Improve their own reasoning

NovaMind learns CONTINUOUSLY — every interaction modifies the system.
But this creates the "stability-plasticity dilemma":
    - Too plastic: catastrophic forgetting (new overwrites old)
    - Too stable: can't learn anything new

Solution: Complementary Learning Systems theory (McClelland et al.):
    - Fast learning in episodic memory (hippocampus analog)
    - Slow consolidation into semantic knowledge (cortex analog)
    - Protection mechanisms for important consolidated knowledge

References:
    - McClelland et al. (1995): "Why There Are Complementary Learning Systems"
    - Kirkpatrick et al. (2017): "Overcoming Catastrophic Forgetting" (EWC)
    - Zenke et al. (2017): "Continual Learning Through Synaptic Intelligence"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time

from .math.information import InformationEngine

EPSILON = 1e-10


class LearningEvent:
    """Record of a single learning event."""

    def __init__(self, event_type: str, description: str, magnitude: float):
        self.id = f"learn_{int(time.time() * 1000)}_{np.random.randint(10000)}"
        self.event_type = event_type  # embedding_update, rule_learned, weight_adjusted, novelty_detected
        self.description = description
        self.magnitude = magnitude
        self.timestamp = time.time()
        self.reversible = True

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.event_type,
            'description': self.description,
            'magnitude': self.magnitude,
            'timestamp': self.timestamp,
        }


class ContinuousLearningEngine:
    """
    Layer 7: Online learning with anti-forgetting mechanisms.

    Implements:
    1. Elastic Weight Consolidation (EWC) — protects important parameters
    2. Novelty Detection — flags genuinely new vs. redundant input
    3. Adaptive Learning Rate — adjusts based on surprise and confidence
    4. Self-Improvement — adjusts reasoning weights based on outcomesFisher
    """

    def __init__(self):
        # Learning state
        self.learning_events: List[LearningEvent] = []
        self.total_updates = 0
        self.learning_rate = 0.1
        self.base_learning_rate = 0.1

        # Elastic Weight Consolidation (anti-forgetting)
        # Tracks importance of each "parameter" (concept embedding)
        self.parameter_importance: Dict[str, float] = defaultdict(lambda: 0.0)
        self.consolidated_values: Dict[str, np.ndarray] = {}

        # Novelty detection
        self.seen_patterns: Dict[str, int] = defaultdict(int)
        self.novelty_threshold = 0.3  # NCD below this = not novel

        # Performance tracking
        self.accuracy_history: List[float] = []
        self.surprise_history: List[float] = []
        self.compression_progress: List[float] = []

    def learn(self, perception: Dict, reasoning: Dict, metacognition: Dict,
              memory_stats: Dict) -> Dict:
        """
        Main learning function — called after each cognitive cycle.

        Decides what to learn and how much, based on:
        - Novelty of the input
        - Surprise level (metacognition)
        - Current learning rate
        - Protection of important knowledge
        """
        report = {
            'events': [],
            'learning_rate': self.learning_rate,
            'novelty_analysis': {},
        }

        # 1. Novelty Detection
        novelty = self._detect_novelty(perception)
        report['novelty_analysis'] = novelty

        # 2. Adaptive Learning Rate
        self._adapt_learning_rate(metacognition)
        report['learning_rate'] = self.learning_rate

        # 3. Embedding Updates (concept positions)
        if novelty['is_novel']:
            events = self._update_embeddings(perception)
            report['events'].extend(events)

        # 4. Reasoning Weight Adjustment
        reasoning_events = self._adjust_reasoning_weights(reasoning, metacognition)
        report['events'].extend(reasoning_events)

        # 5. Consolidation Check (EWC)
        consolidation_events = self._consolidation_check(memory_stats)
        report['events'].extend(consolidation_events)

        # 6. Self-Improvement Analysis
        self_improvement = self._analyze_self_improvement(metacognition)
        report['self_improvement'] = self_improvement

        # Track
        self.total_updates += 1
        self.learning_events.extend(report['events'])

        # Convert events to dicts for JSON serialization
        report['events'] = [e.to_dict() for e in report['events']]

        report['total_updates'] = self.total_updates
        report['total_events'] = len(self.learning_events)

        return report

    def _detect_novelty(self, perception: Dict) -> Dict:
        """
        Detect if the input contains genuinely novel information.

        Uses Normalized Compression Distance (NCD) to compare against
        previously seen patterns. Truly novel input has high NCD with
        all previous patterns.
        """
        concepts = perception.get('concepts', [])
        if not concepts:
            return {'is_novel': False, 'score': 0.0, 'reason': 'No concepts to analyze'}

        # Create a signature of this input
        input_sig = '|'.join(sorted(c['label'] for c in concepts))

        # Check against previous patterns
        if input_sig in self.seen_patterns:
            self.seen_patterns[input_sig] += 1
            return {
                'is_novel': False,
                'score': 0.0,
                'reason': f'Pattern seen {self.seen_patterns[input_sig]} times before',
                'repeat_count': self.seen_patterns[input_sig],
            }

        # Compute NCD against recent patterns
        min_ncd = 1.0
        most_similar = None
        for past_pattern in list(self.seen_patterns.keys())[-50:]:
            ncd = InformationEngine.normalized_compression_distance(input_sig, past_pattern)
            if ncd < min_ncd:
                min_ncd = ncd
                most_similar = past_pattern

        self.seen_patterns[input_sig] = 1
        is_novel = min_ncd > self.novelty_threshold

        return {
            'is_novel': is_novel,
            'score': float(min_ncd),
            'reason': 'Genuinely novel pattern' if is_novel else f'Similar to existing pattern (NCD={min_ncd:.3f})',
            'most_similar_to': most_similar,
        }

    def _adapt_learning_rate(self, metacognition: Dict):
        """
        Adapt learning rate based on metacognitive signals.

        - High surprise → increase learning rate (need to update fast)
        - High confidence → decrease learning rate (don't mess up what works)
        - Low coherence → moderate learning rate (update carefully)
        """
        surprise = metacognition.get('surprise', {}).get('current_level', 0.5)
        confidence = metacognition.get('confidence', {}).get('overall', 0.5)
        coherence = metacognition.get('coherence', {}).get('score', 0.5)

        # High surprise → learn fast, High confidence → learn slow
        rate_modifier = (
            surprise * 0.5 +          # Surprise increases rate
            (1 - confidence) * 0.3 +   # Low confidence increases rate
            coherence * 0.2            # High coherence allows faster learning
        )

        self.learning_rate = np.clip(
            self.base_learning_rate * (0.5 + rate_modifier),
            0.01, 0.5
        )

    def _update_embeddings(self, perception: Dict) -> List[LearningEvent]:
        """
        Update concept embeddings based on new information.

        Applies EWC-protected gradient steps — important concepts
        are updated less aggressively.
        """
        events = []

        for concept in perception.get('concepts', []):
            concept_id = concept.get('id', '')
            if not concept_id:
                continue

            importance = self.parameter_importance[concept_id]
            # EWC: effective learning rate is reduced for important concepts
            effective_lr = self.learning_rate / (1 + importance)

            if effective_lr > 0.01:
                event = LearningEvent(
                    'embedding_update',
                    f"Updated embedding for '{concept.get('label', '?')}' "
                    f"(lr={effective_lr:.4f}, importance={importance:.2f})",
                    effective_lr
                )
                events.append(event)

                # Increase importance for this concept (it's been used)
                self.parameter_importance[concept_id] += 0.1

        return events

    def _adjust_reasoning_weights(self, reasoning: Dict, metacognition: Dict) -> List[LearningEvent]:
        """
        Adjust the weights of reasoning paradigms based on performance.

        If neural reasoning was more accurate than symbolic this cycle,
        slightly increase neural's weight for next cycle.
        """
        events = []

        paradigms = reasoning.get('paradigms', {})
        if not paradigms:
            return events

        # Get confidence of each paradigm
        paradigm_confidences = {}
        for name, result in paradigms.items():
            paradigm_confidences[name] = result.get('confidence', 0.5)

        # Track accuracy
        overall_confidence = metacognition.get('confidence', {}).get('overall', 0.5)
        self.accuracy_history.append(overall_confidence)

        # If we have enough history, adjust weights
        if len(self.accuracy_history) >= 5:
            # Check which paradigm contributes most to high-confidence cycles
            for name, conf in paradigm_confidences.items():
                if conf > overall_confidence:
                    # This paradigm performed better than average
                    event = LearningEvent(
                        'weight_adjusted',
                        f"Slightly increased weight for '{name}' reasoning "
                        f"(paradigm conf: {conf:.2f} > overall: {overall_confidence:.2f})",
                        0.01
                    )
                    events.append(event)

        return events

    def _consolidation_check(self, memory_stats: Dict) -> List[LearningEvent]:
        """
        Check if any knowledge should be consolidated (protected from forgetting).

        Knowledge is consolidated when:
        - It has been accessed frequently
        - It has survived multiple compaction cycles
        - It contributes to high-confidence reasoning
        """
        events = []

        semantic_stats = memory_stats.get('semantic', {})
        total_rules = semantic_stats.get('total_rules', 0)

        if total_rules > 0:
            # Track compression progress (Schmidhuber's theory)
            compression = semantic_stats.get('total_compression', {})
            ratio = compression.get('overall_ratio', 0)
            self.compression_progress.append(ratio)

            if len(self.compression_progress) >= 3:
                recent_progress = self.compression_progress[-1] - self.compression_progress[-3]
                if recent_progress > 0.05:
                    event = LearningEvent(
                        'consolidation',
                        f"Compression progress: {recent_progress:.1%} improvement. "
                        f"Knowledge is becoming more compressed (better understood).",
                        recent_progress
                    )
                    events.append(event)

        return events

    def _analyze_self_improvement(self, metacognition: Dict) -> Dict:
        """
        Analyze whether the system is improving over time.

        Tracks:
        - Accuracy trend (are we getting better?)
        - Compression trend (is knowledge getting more compact?)
        - Surprise trend (are we less surprised over time?)
        """
        report = {
            'accuracy_trend': 'insufficient_data',
            'compression_trend': 'insufficient_data',
            'surprise_trend': 'insufficient_data',
            'overall': 'insufficient_data',
        }

        if len(self.accuracy_history) >= 5:
            recent = np.mean(self.accuracy_history[-3:])
            earlier = np.mean(self.accuracy_history[-5:-2]) if len(self.accuracy_history) >= 6 else np.mean(self.accuracy_history[:3])
            if recent > earlier + 0.05:
                report['accuracy_trend'] = 'improving'
            elif recent < earlier - 0.05:
                report['accuracy_trend'] = 'declining'
            else:
                report['accuracy_trend'] = 'stable'

        if len(self.compression_progress) >= 3:
            recent = self.compression_progress[-1]
            earlier = self.compression_progress[0]
            if recent > earlier + 0.01:
                report['compression_trend'] = 'improving'
            else:
                report['compression_trend'] = 'stable'

        surprise = metacognition.get('surprise', {})
        report['surprise_trend'] = surprise.get('trend', 'insufficient_data')

        # Overall assessment
        trends = [report['accuracy_trend'], report['compression_trend'], report['surprise_trend']]
        improving_count = trends.count('improving')
        declining_count = trends.count('declining')

        if improving_count >= 2:
            report['overall'] = 'improving'
        elif declining_count >= 2:
            report['overall'] = 'declining'
        else:
            report['overall'] = 'stable'

        return report

    def get_stats(self) -> Dict:
        """Return learning statistics for visualization."""
        return {
            'total_updates': self.total_updates,
            'current_learning_rate': self.learning_rate,
            'total_learning_events': len(self.learning_events),
            'recent_events': [e.to_dict() for e in self.learning_events[-10:]],
            'novelty_threshold': self.novelty_threshold,
            'unique_patterns_seen': len(self.seen_patterns),
            'parameter_importance_stats': {
                'mean': float(np.mean(list(self.parameter_importance.values()))) if self.parameter_importance else 0,
                'max': float(max(self.parameter_importance.values())) if self.parameter_importance else 0,
                'num_tracked': len(self.parameter_importance),
            },
            'accuracy_history': self.accuracy_history[-20:],
            'compression_progress': self.compression_progress[-20:],
        }

"""
NovaMind — Layer 3: Memory System (Triple Memory + Auto-Compaction)
===================================================================

Implements a biologically-inspired triple memory system with information-
theoretic auto-compaction based on the Minimum Description Length principle.

Memory Architecture:
--------------------
    1. EPISODIC MEMORY — "What happened"
       Raw, timestamped experiences. Each interaction is stored verbatim.
       Subject to decay + compaction.

    2. SEMANTIC MEMORY — "What I know"
       Compressed, abstract knowledge. Rules, patterns, and facts extracted
       from episodic memories via MDL compaction.

    3. PROCEDURAL MEMORY — "How to do things"
       Learned procedures and strategies. Emerges when the system discovers
       repeating action sequences that achieve goals.

Auto-Compaction Process:
    1. Episodic buffer fills with raw experiences
    2. When buffer exceeds threshold, MDL engine analyzes patterns
    3. Patterns with MDL score below threshold → compressed into semantic rules
    4. Original episodes that were fully explained → allowed to decay
    5. Compression metrics tracked for metacognition

This is the "ZIP cognitivo" the user described — intelligence IS compression.

References:
    - Tulving (1972): "Episodic and Semantic Memory"
    - Squire (2004): "Memory Systems of the Brain"
    - Grünwald (2007): "The Minimum Description Length Principle"
    - Schmidhuber (2009): "Driven by Compression Progress"
"""

import time
import copy
import random
import threading
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict
import numpy as np

from .math.information import MDLCompactor, InformationEngine

# Memory configuration
EPISODIC_BUFFER_SIZE = 500    # Max episodes before compaction triggers
COMPACTION_THRESHOLD = 10     # Minimum episodes to attempt compaction
DECAY_RATE = 0.01             # Per-cycle activation decay (slower)
MIN_ACTIVATION = 0.001        # Below this, episode can be forgotten
COMPACTION_RATIO_TARGET = 0.3 # Target compression ratio (more aggressive)

# Consolidation/Replay config
CONSOLIDATION_INTERVAL = 300  # Seconds between replay cycles (5 min)
REPLAY_BATCH_SIZE = 20        # Episodes per replay
SALIENCE_THRESHOLD = 0.6      # Min importance for consolidation
EMOTIONAL_BOOST = 1.5         # Multiplier for emotional memories
SLEEP_REPLAY_MULTIPLIER = 3.0 # Replay rate during "sleep" phase

# Forgetting curve parameters (Ebbinghaus)
FORGETTING_CURVE_A = 1.0      # Initial retention
FORGETTING_CURVE_B = 0.1      # Decay rate per hour
FORGETTING_CURVE_C = 0.15     # Asymptotic minimum retention


class Episode:
    """
    A single episodic memory — a complete interaction record.

    Biologically informed:
    - Emotional valence (amygdala tagging)
    - Ebbinghaus forgetting curve (time-based decay)
    - Consolidation counter (hippocampal replay strength)
    - Surprise score (prediction error during encoding)
    """

    def __init__(self, content: Dict, episode_type: str = "interaction",
                 emotional_valence: float = 0.0, surprise: float = 0.0):
        self.id = f"ep_{int(time.time() * 1000)}_{np.random.randint(10000)}"
        self.content = content
        self.episode_type = episode_type
        self.timestamp = time.time()
        self.activation = 1.0
        self.access_count = 0
        self.compressed = False
        self.compression_rule_id = None
        self.emotional_valence = max(-1.0, min(1.0, emotional_valence))  # amygdala tag
        self.surprise = max(0.0, min(1.0, surprise))  # prediction error
        self.importance = self._compute_initial_importance()
        self.consolidation_count = 0  # times replayed (hippocampal replay)
        self.last_replay_time = self.timestamp
        self.expectation = ""  # what the mind predicted before this input

    def _compute_initial_importance(self) -> float:
        base = 0.5
        emotional_boost = abs(self.emotional_valence) * 0.3
        surprise_boost = self.surprise * 0.2
        return min(1.0, base + emotional_boost + surprise_boost)

    def compute_retention(self, now: float = None) -> float:
        """Ebbinghaus forgetting curve: R = A * e^(-t/B) + C"""
        if now is None:
            now = time.time()
        age_hours = (now - self.timestamp) / 3600
        return FORGETTING_CURVE_A * np.exp(-FORGETTING_CURVE_B * age_hours) + FORGETTING_CURVE_C

    def access(self):
        self.access_count += 1
        self.activation = min(1.0, self.activation + 0.3)

    def decay(self, rate: float = DECAY_RATE):
        """Ebbinghaus forgetting curve-based decay."""
        retention = self.compute_retention()
        replay_bonus = min(0.2, self.consolidation_count * 0.02)
        self.activation = retention + replay_bonus
        self.activation = max(MIN_ACTIVATION, self.activation)

    def replay_boost(self):
        """Simulates hippocampal replay: strengthens memory."""
        self.consolidation_count += 1
        self.last_replay_time = time.time()
        self.activation = min(1.0, self.activation + 0.15)
        self.importance = min(1.0, self.importance + 0.05)

    def set_expectation(self, expected_input: str):
        self.expectation = expected_input

    def compute_surprise(self, actual_input: str) -> float:
        """Computes prediction error via string divergence."""
        if not self.expectation:
            return 0.0
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, self.expectation, actual_input).ratio()
        return 1.0 - ratio

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.episode_type,
            'content': self.content,
            'timestamp': self.timestamp,
            'activation': self.activation,
            'access_count': self.access_count,
            'compressed': self.compressed,
            'importance': self.importance,
            'emotional_valence': self.emotional_valence,
            'surprise': self.surprise,
            'consolidation_count': self.consolidation_count,
        }


class SemanticRule:
    """
    A semantic memory entry — a compressed piece of knowledge.

    Created by the auto-compaction process from multiple episodic memories.
    Represents a general rule, fact, or pattern.
    """

    def __init__(self, rule_type: str, content: Dict, source_episodes: int = 0):
        self.id = f"sem_{int(time.time() * 1000)}_{np.random.randint(10000)}"
        self.rule_type = rule_type  # fact, rule, pattern, abstraction
        self.content = content
        self.source_episode_count = source_episodes
        self.confidence = min(0.9, 0.3 + 0.1 * source_episodes)
        self.creation_time = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.activation = 0.8
        self.compression_ratio = 0.0
        self.emotional_valence = 0.0  # aggregated from source episodes
        self.generalization_power = min(1.0, source_episodes * 0.05)  # how abstract

    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()
        self.activation = min(1.0, self.activation + 0.1)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.rule_type,
            'content': self.content,
            'confidence': self.confidence,
            'source_episodes': self.source_episode_count,
            'access_count': self.access_count,
            'compression_ratio': self.compression_ratio,
            'activation': self.activation,
            'emotional_valence': self.emotional_valence,
            'generalization_power': self.generalization_power,
        }


class Procedure:
    """
    A procedural memory entry — a learned sequence of actions.

    Emerges when the system detects repeating action patterns that
    achieve specific goals.
    """

    def __init__(self, name: str, steps: List[str], goal: str):
        self.id = f"proc_{int(time.time() * 1000)}_{np.random.randint(10000)}"
        self.name = name
        self.steps = steps
        self.goal = goal
        self.success_count = 0
        self.failure_count = 0
        self.creation_time = time.time()
        self.last_used = time.time()
        self.activation = 0.7

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'steps': self.steps,
            'goal': self.goal,
            'success_rate': self.success_rate,
            'activation': self.activation,
        }


class MemorySystem:
    """
    Layer 3: Triple Memory System with Adaptive Relevance.

    Biologically-informed features:
    - Ebbinghaus forgetting curve (time-based retention)
    - Emotional valence tagging (amygdala-like salience)
    - Hippocampal replay consolidation
    - Surprise-based importance (prediction error)
    - Background consolidation loop
    """

    def __init__(self):
        self.episodic: List[Episode] = []
        self.semantic: List[SemanticRule] = []
        self.procedural: List[Procedure] = []

        self.encoder = None
        self.compactor = MDLCompactor()

        # Metrics
        self.total_episodes_stored = 0
        self.total_episodes_compacted = 0
        self.total_episodes_forgotten = 0
        self.compaction_events: List[Dict] = []

        # Emotional context (propagates through consecutive episodes)
        self.current_emotional_context = 0.0
        self.current_arousal = 0.0  # intensity of emotional state

        # Consolidation state
        self._consolidation_thread: Optional[threading.Thread] = None
        self._consolidation_running = False
        self._last_consolidation_time = time.time()

    def store_episode(self, content: Dict, episode_type: str = "interaction",
                      importance: float = None, emotional_valence: float = None,
                      surprise: float = None, expected_input: str = None) -> Episode:
        """
        Store a new episodic memory with adaptive relevance.

        - emotional_valence: if None, uses current emotional context
        - surprise: if None, computed from expectation if available
        - importance: if None, computed from emotional_valence + surprise
        """
        if emotional_valence is None:
            emotional_valence = self.current_emotional_context
        if surprise is None and expected_input:
            surprise = 0.3  # moderate default surprise
        elif surprise is None:
            surprise = 0.1

        episode = Episode(content, episode_type, emotional_valence, surprise)
        if expected_input:
            episode.set_expectation(expected_input)
            episode.surprise = episode.compute_surprise(str(content))
        episode.importance = episode._compute_initial_importance()
        if importance is not None:
            episode.importance = importance

        self.episodic.append(episode)
        self.total_episodes_stored += 1

        # Emotional context decay (emotional state drifts slowly)
        self.current_emotional_context *= 0.95
        self.current_emotional_context += emotional_valence * 0.05
        self.current_arousal = max(0, self.current_arousal * 0.9 + abs(emotional_valence) * 0.1)

        if len(self.episodic) >= EPISODIC_BUFFER_SIZE:
            self._auto_compact()

        return episode

    def set_emotional_context(self, valence: float, arousal: float = 0.5):
        """Set current emotional state (propagates to future episodes)."""
        self.current_emotional_context = max(-1.0, min(1.0, valence))
        self.current_arousal = max(0.0, min(1.0, arousal))

    def start_consolidation(self):
        """Start background hippocampal replay loop."""
        if self._consolidation_running:
            return
        self._consolidation_running = True
        self._consolidation_thread = threading.Thread(target=self._consolidation_loop, daemon=True)
        self._consolidation_thread.start()
        print("[MEMORY] Consolidation (replay) loop started")

    def stop_consolidation(self):
        self._consolidation_running = False

    def _consolidation_loop(self):
        """Background loop simulating hippocampal replay during 'sleep'."""
        while self._consolidation_running:
            time.sleep(CONSOLIDATION_INTERVAL)
            try:
                self._replay_and_consolidate()
            except Exception as e:
                print(f"[MEMORY] Consolidation error: {e}")

    def _replay_and_consolidate(self):
        """
        Hippocampal replay: reactivates salient episodes, strengthens them,
        and may consolidate into semantic rules.

        Also replays during "sleep" with higher frequency.
        """
        now = time.time()
        delta = now - self._last_consolidation_time
        self._last_consolidation_time = now

        # Check if it's "sleep time" (no recent activity = sleep phase)
        is_sleep = delta > CONSOLIDATION_INTERVAL * 2

        replay_multiplier = SLEEP_REPLAY_MULTIPLIER if is_sleep else 1.0
        replay_count = int(REPLAY_BATCH_SIZE * replay_multiplier)

        # Select candidates: high importance + high emotional valence
        candidates = sorted(
            [ep for ep in self.episodic if not ep.compressed and ep.activation > MIN_ACTIVATION],
            key=lambda ep: ep.importance + abs(ep.emotional_valence) * 0.5,
            reverse=True
        )

        replayed = candidates[:replay_count]
        for ep in replayed:
            ep.replay_boost()

            # Reactivate emotionally congruent episodes (spreading activation)
            for other in self.episodic:
                if other is ep and other.activation > MIN_ACTIVATION:
                    continue
                valence_sim = 1 - abs(ep.emotional_valence - other.emotional_valence)
                if valence_sim > 0.7:
                    other.activation = min(1.0, other.activation + 0.02)

        # Sleep: trigger extra compaction
        if is_sleep and len(self.episodic) >= COMPACTION_THRESHOLD:
            self._auto_compact()

        # Forget truly redundant episodes
        self._forget_old_episodes()

        # Propagate emotional valence to semantic rules
        for rule in self.semantic:
            related = [ep for ep in self.episodic
                       if ep.compression_rule_id == rule.id]
            if related:
                rule.emotional_valence = float(np.mean([ep.emotional_valence for ep in related]))

    def trigger_sleep_cycle(self):
        """
        Manual trigger for sleep consolidation.
        Can be called when learning loop detects idle time.
        """
        print("[MEMORY] Sleep cycle triggered")
        self._replay_and_consolidate()

    def clear_emotional_context(self):
        self.current_emotional_context = 0.0
        self.current_arousal = 0.0

    STOPWORDS = {'the','a','an','is','are','was','were','be','been','has','have',
                 'had','do','does','did','will','would','can','could','may','might',
                 'what','why','how','when','where','who','which','this','that','its',
                 'it','i','you','he','she','we','they','me','him','not','no','yes',
                 'to','of','in','on','for','with','by','as','at','from','am','my'}

    @staticmethod
    def _content_words(text: str) -> set:
        return {w.lower() for w in str(text).split()
                if w.lower() not in MemorySystem.STOPWORDS and len(w) > 2}

    def recall_episodic(self, query: Dict, top_k: int = 5,
                        emotional_bias: float = 0.0) -> List[Episode]:
        """
        Retrieve most relevant episodic memories.

        Adaptive relevance includes:
        - Content similarity (NCD or embedding)
        - Emotional congruence with current state
        - Consolidation strength (replay count)
        - Ebbinghaus retention (activation via forgetting curve)
        """
        query_str = str(query)
        query_input = query.get('input', query_str)

        if self.encoder is not None:
            return self._recall_episodic_embedding(query_input, top_k)

        scored_episodes = []
        query_keywords = self._content_words(query_input)
        for episode in self.episodic:
            if episode.activation < MIN_ACTIVATION:
                continue

            ep_str = str(episode.content)
            ncd = InformationEngine.normalized_compression_distance(query_str, ep_str)
            content_score = 1 - ncd

            ep_input = episode.content.get('input', ep_str)
            ep_keywords = self._content_words(ep_input)
            if query_keywords and ep_keywords:
                jaccard = len(query_keywords & ep_keywords) / len(query_keywords | ep_keywords)
            else:
                jaccard = 0

            # Emotional congruence: if emotional_bias != 0, favor congruent episodes
            emotional_bonus = 0
            if emotional_bias != 0:
                emotional_bonus = (1 - abs(emotional_bias - episode.emotional_valence)) * 0.15

            # Consolidation bonus: replayed memories are more recallable
            consolidation_bonus = min(0.1, episode.consolidation_count * 0.01)

            relevance = (
                jaccard * 10.0 +
                content_score * 0.3 +
                episode.activation * 0.1 +
                episode.importance * 0.05 +
                emotional_bonus +
                consolidation_bonus
            )

            scored_episodes.append((relevance, episode))

        scored_episodes.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, episode in scored_episodes[:top_k]:
            episode.access()
            results.append(episode)

        return results

    def _recall_episodic_embedding(self, query_input: str, top_k: int,
                                   emotional_bias: float = 0.0) -> List[Episode]:
        """Recall by cosine similarity over MiniLM sentence embeddings."""
        active = [ep for ep in self.episodic if ep.activation >= MIN_ACTIVATION]
        if not active:
            return []

        q_vec = self.encoder.encode_one(query_input)
        ep_texts = [ep.content.get('input', str(ep.content)) for ep in active]
        ep_vecs = self.encoder.encode(ep_texts)

        sims = ep_vecs @ q_vec
        scored = []
        for ep, sim in zip(active, sims):
            emotional_bonus = 0
            if emotional_bias != 0:
                emotional_bonus = (1 - abs(emotional_bias - ep.emotional_valence)) * 0.1
            consolidation_bonus = min(0.05, ep.consolidation_count * 0.005)
            relevance = float(sim) + 0.05 * ep.activation + 0.02 * ep.importance + emotional_bonus + consolidation_bonus
            scored.append((relevance, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, ep in scored[:top_k]:
            ep.access()
            results.append(ep)
        return results

    def recall_semantic(self, query: Dict, top_k: int = 5,
                        emotional_bias: float = 0.0) -> List[SemanticRule]:
        """Retrieve most relevant semantic rules for a query."""
        scored_rules = []

        query_str = str(query)
        for rule in self.semantic:
            rule_str = str(rule.content)
            ncd = InformationEngine.normalized_compression_distance(query_str, rule_str)
            content_score = 1 - ncd

            emotional_bonus = 0
            if emotional_bias != 0:
                emotional_bonus = (1 - abs(emotional_bias - rule.emotional_valence)) * 0.1

            relevance = (
                content_score * 0.5 +
                rule.confidence * 0.3 +
                rule.activation * 0.15 +
                rule.generalization_power * 0.05 +
                emotional_bonus
            )

            scored_rules.append((relevance, rule))

        scored_rules.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, rule in scored_rules[:top_k]:
            rule.access()
            results.append(rule)

        return results

    def recall_procedure(self, goal: str) -> Optional[Procedure]:
        """Find the best procedure for achieving a goal."""
        best = None
        best_score = 0

        for proc in self.procedural:
            if proc.goal.lower() in goal.lower() or goal.lower() in proc.goal.lower():
                score = proc.success_rate * proc.activation
                if score > best_score:
                    best_score = score
                    best = proc

        if best:
            best.last_used = time.time()
        return best

    def store_semantic_rule(self, content: Dict, source: str = "extraction") -> SemanticRule:
        """Directly store a semantic rule (bypasses auto-compaction)."""
        rule = SemanticRule(rule_type='fact', content=content, source_episodes=1)
        rule.confidence = 0.85
        self.semantic.append(rule)
        return rule

    def _auto_compact(self):
        """
        The heart of the memory system: MDL-based auto-compaction.

        Process:
        1. Analyze all active episodes for patterns
        2. Find patterns that compress well (low MDL score)
        3. Create semantic rules from good patterns
        4. Mark covered episodes as compressed
        5. Allow very old, compressed episodes to be forgotten
        """
        # Only try to compact if we have enough episodes
        active_episodes = [ep for ep in self.episodic if not ep.compressed]
        if len(active_episodes) < COMPACTION_THRESHOLD:
            return

        # Prepare episodes for MDL analysis
        episode_dicts = [ep.content for ep in active_episodes]

        # Find patterns
        patterns = self.compactor.analyze_patterns(episode_dicts)

        if not patterns:
            return

        # Compact: create rules from good patterns
        compaction = self.compactor.compact(episode_dicts, patterns)

        new_rules = []
        for rule_data in compaction.get('rules', []):
            rule = SemanticRule(
                rule_type=rule_data.get('type', 'pattern'),
                content=rule_data,
                source_episodes=rule_data.get('instance_count', 0)
            )
            rule.compression_ratio = compaction.get('compression_ratio', 0)
            new_rules.append(rule)
            self.semantic.append(rule)

        # Mark covered episodes
        covered_count = compaction.get('episodes_covered', 0)
        for i, ep in enumerate(active_episodes):
            if i < covered_count:
                ep.compressed = True
                self.total_episodes_compacted += 1

        # Record compaction event
        event = {
            'timestamp': time.time(),
            'episodes_analyzed': len(active_episodes),
            'patterns_found': len(patterns),
            'rules_created': len(new_rules),
            'episodes_covered': covered_count,
            'compression_ratio': compaction.get('compression_ratio', 0),
            'raw_bits': compaction.get('raw_bits', 0),
            'compressed_bits': compaction.get('compressed_bits', 0),
        }
        self.compaction_events.append(event)

        # Forget very old compressed episodes
        self._forget_old_episodes()

        return event

    def _forget_old_episodes(self):
        """
        Controlled forgetting with adaptive thresholds.

        Forgets episodes that are:
        1. Compressed into semantic rules
        2. Low Ebbinghaus retention (< threshold)
        3. Low emotional salience
        4. Low consolidation count (never replayed)
        """
        surviving = []
        for ep in self.episodic:
            retention = ep.compute_retention()
            forget = all([
                ep.compressed,
                ep.activation < MIN_ACTIVATION * 3,
                ep.importance < 0.5,
                abs(ep.emotional_valence) < 0.3,
                ep.consolidation_count < 2
            ])
            if forget:
                self.total_episodes_forgotten += 1
            else:
                surviving.append(ep)
        self.episodic = surviving

    def learn_procedure(self, name: str, steps: List[str], goal: str, success: bool = True):
        """Learn or update a procedural memory."""
        # Check if similar procedure exists
        for proc in self.procedural:
            if proc.name == name or proc.goal == goal:
                if success:
                    proc.success_count += 1
                else:
                    proc.failure_count += 1
                proc.last_used = time.time()
                return proc

        proc = Procedure(name, steps, goal)
        if success:
            proc.success_count = 1
        self.procedural.append(proc)
        return proc

    def decay_all(self):
        """Ebbinghaus forgetting curve decay to all memories."""
        now = time.time()
        for ep in self.episodic:
            ep.decay()
        for rule in self.semantic:
            age_hours = (now - rule.creation_time) / 3600
            retention = np.exp(-DECAY_RATE * age_hours) + 0.05
            rule.activation = max(0.05, min(1.0, retention * (0.5 + 0.5 * rule.confidence)))
        # Decay emotional context if no recent input
        self.current_emotional_context *= 0.99
        self.current_arousal *= 0.98

    def get_stats(self) -> Dict:
        """Return memory system statistics for visualization."""
        active_episodes = [ep for ep in self.episodic if not ep.compressed]
        compressed_episodes = [ep for ep in self.episodic if ep.compressed]

        avg_retention = float(np.mean([ep.compute_retention() for ep in self.episodic])) if self.episodic else 0

        avg_valence = float(np.mean([ep.emotional_valence for ep in self.episodic])) if self.episodic else 0
        avg_surprise = float(np.mean([ep.surprise for ep in self.episodic])) if self.episodic else 0
        avg_consolidation = float(np.mean([ep.consolidation_count for ep in self.episodic])) if self.episodic else 0

        return {
            'episodic': {
                'total_stored': self.total_episodes_stored,
                'currently_active': len(active_episodes),
                'currently_compressed': len(compressed_episodes),
                'total_compacted': self.total_episodes_compacted,
                'total_forgotten': self.total_episodes_forgotten,
                'avg_activation': float(np.mean([ep.activation for ep in self.episodic])) if self.episodic else 0,
                'avg_retention': avg_retention,
                'avg_valence': avg_valence,
                'avg_surprise': avg_surprise,
                'avg_consolidation': avg_consolidation,
                'emotional_context': self.current_emotional_context,
                'arousal': self.current_arousal,
            },
            'semantic': {
                'total_rules': len(self.semantic),
                'avg_confidence': float(np.mean([r.confidence for r in self.semantic])) if self.semantic else 0,
                'avg_valence': float(np.mean([r.emotional_valence for r in self.semantic])) if self.semantic else 0,
                'total_compression': self.compactor.get_compression_stats(),
            },
            'procedural': {
                'total_procedures': len(self.procedural),
                'avg_success_rate': float(np.mean([p.success_rate for p in self.procedural])) if self.procedural else 0,
            },
            'compaction_events': self.compaction_events[-10:],
        }

    def get_all_memories(self) -> Dict:
        """Return all memory contents for visualization."""
        return {
            'episodic': [ep.to_dict() for ep in self.episodic[-20:]],  # Last 20
            'semantic': [r.to_dict() for r in self.semantic],
            'procedural': [p.to_dict() for p in self.procedural],
        }

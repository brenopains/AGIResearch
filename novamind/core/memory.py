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
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np

from .math.information import MDLCompactor, InformationEngine

# Memory configuration
EPISODIC_BUFFER_SIZE = 50     # Max episodes before compaction triggers
COMPACTION_THRESHOLD = 3      # Minimum episodes to attempt compaction
DECAY_RATE = 0.02             # Per-cycle activation decay
MIN_ACTIVATION = 0.01         # Below this, episode can be forgotten
COMPACTION_RATIO_TARGET = 0.5 # Target compression ratio


class Episode:
    """
    A single episodic memory — a complete interaction record.

    Contains the full context: input, perception, reasoning trace,
    response, and metadata. Stored verbatim until compacted.
    """

    def __init__(self, content: Dict, episode_type: str = "interaction"):
        self.id = f"ep_{int(time.time() * 1000)}_{np.random.randint(10000)}"
        self.content = content
        self.episode_type = episode_type
        self.timestamp = time.time()
        self.activation = 1.0          # Current relevance (decays over time)
        self.access_count = 0          # Number of times retrieved
        self.compressed = False        # Has this been compacted?
        self.compression_rule_id = None  # Which rule compressed this
        self.emotional_valence = 0.0   # [-1, 1] emotional marking
        self.importance = 0.5          # [0, 1] estimated importance

    def access(self):
        """Record an access (boosts activation)."""
        self.access_count += 1
        self.activation = min(1.0, self.activation + 0.3)

    def decay(self, rate: float = DECAY_RATE):
        """Apply time-based decay to activation."""
        age = time.time() - self.timestamp
        decay_factor = np.exp(-rate * age / 3600)  # Hourly decay
        self.activation *= decay_factor
        self.activation = max(MIN_ACTIVATION, self.activation)

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
        self.compression_ratio = 0.0  # How much was compressed

    def access(self):
        """Record an access."""
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
    Layer 3: Triple Memory System with Auto-Compaction.

    This is where NovaMind fundamentally diverges from LLMs:
    instead of a fixed context window, it has LIVING memory that
    grows, compresses, abstracts, and forgets.
    """

    def __init__(self):
        # Three memory stores
        self.episodic: List[Episode] = []
        self.semantic: List[SemanticRule] = []
        self.procedural: List[Procedure] = []

        # Optional semantic encoder (MiniLM). When attached, recall uses real
        # embedding cosine similarity instead of Jaccard/NCD string overlap.
        # See core.semantic_encoder.attach_encoder().
        self.encoder = None

        # Auto-compaction engine (MDL-based)
        self.compactor = MDLCompactor()

        # Memory metrics
        self.total_episodes_stored = 0
        self.total_episodes_compacted = 0
        self.total_episodes_forgotten = 0
        self.compaction_events: List[Dict] = []

    def store_episode(self, content: Dict, episode_type: str = "interaction",
                      importance: float = 0.5) -> Episode:
        """
        Store a new episodic memory.

        Automatically triggers compaction if the buffer is full.
        """
        episode = Episode(content, episode_type)
        episode.importance = importance
        self.episodic.append(episode)
        self.total_episodes_stored += 1

        # Check if we need to compact
        if len(self.episodic) >= EPISODIC_BUFFER_SIZE:
            self._auto_compact()

        return episode

    STOPWORDS = {'the','a','an','is','are','was','were','be','been','has','have',
                 'had','do','does','did','will','would','can','could','may','might',
                 'what','why','how','when','where','who','which','this','that','its',
                 'it','i','you','he','she','we','they','me','him','not','no','yes',
                 'to','of','in','on','for','with','by','as','at','from','am','my'}

    @staticmethod
    def _content_words(text: str) -> set:
        return {w.lower() for w in str(text).split()
                if w.lower() not in MemorySystem.STOPWORDS and len(w) > 2}

    def recall_episodic(self, query: Dict, top_k: int = 5) -> List[Episode]:
        """
        Retrieve most relevant episodic memories for a query.

        Uses a combination of:
        - Content similarity (NCD-based)
        - Keyword Jaccard overlap (topic matching)
        - Activation level (recency + importance)
        """
        query_str = str(query)
        query_input = query.get('input', query_str)

        # ── Embedding path: real semantic similarity (MiniLM cosine) ──
        if self.encoder is not None:
            return self._recall_episodic_embedding(query_input, top_k)

        # ── Fallback path: string overlap (Jaccard + NCD) ──
        scored_episodes = []
        query_keywords = self._content_words(query_input)
        for episode in self.episodic:
            if episode.activation < MIN_ACTIVATION:
                continue

            ep_str = str(episode.content)
            ncd = InformationEngine.normalized_compression_distance(query_str, ep_str)
            content_score = 1 - ncd

            # Keyword Jaccard overlap on input text only (not full dict representation)
            ep_input = episode.content.get('input', ep_str)
            ep_keywords = self._content_words(ep_input)
            if query_keywords and ep_keywords:
                jaccard = len(query_keywords & ep_keywords) / len(query_keywords | ep_keywords)
            else:
                jaccard = 0

            # Combined relevance: keyword overlap dominates, NCD breaks ties
            relevance = (
                jaccard * 10.0 +
                content_score * 0.3 +
                episode.activation * 0.1 +
                episode.importance * 0.05
            )

            scored_episodes.append((relevance, episode))

        scored_episodes.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, episode in scored_episodes[:top_k]:
            episode.access()
            results.append(episode)

        return results

    def _recall_episodic_embedding(self, query_input: str, top_k: int) -> List[Episode]:
        """Recall by cosine similarity over MiniLM sentence embeddings."""
        active = [ep for ep in self.episodic if ep.activation >= MIN_ACTIVATION]
        if not active:
            return []

        q_vec = self.encoder.encode_one(query_input)
        ep_texts = [ep.content.get('input', str(ep.content)) for ep in active]
        ep_vecs = self.encoder.encode(ep_texts)

        sims = ep_vecs @ q_vec  # both L2-normalized → cosine similarity
        scored = []
        for ep, sim in zip(active, sims):
            # Semantic similarity dominates; activation/importance only break ties.
            relevance = float(sim) + 0.05 * ep.activation + 0.02 * ep.importance
            scored.append((relevance, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, ep in scored[:top_k]:
            ep.access()
            results.append(ep)
        return results

    def recall_semantic(self, query: Dict, top_k: int = 5) -> List[SemanticRule]:
        """Retrieve most relevant semantic rules for a query."""
        scored_rules = []

        query_str = str(query)
        for rule in self.semantic:
            rule_str = str(rule.content)
            ncd = InformationEngine.normalized_compression_distance(query_str, rule_str)
            content_score = 1 - ncd

            relevance = (
                content_score * 0.5 +
                rule.confidence * 0.3 +
                rule.activation * 0.2
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
        Forget old compressed episodes with very low activation.

        This is controlled forgetting — only episodes that have been:
        1. Successfully compressed into semantic rules
        2. Have decayed below activation threshold
        3. Are not marked as high importance
        """
        surviving = []
        for ep in self.episodic:
            if ep.compressed and ep.activation < MIN_ACTIVATION * 2 and ep.importance < 0.7:
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
        """Apply time-based decay to all memories."""
        for ep in self.episodic:
            ep.decay()
        for rule in self.semantic:
            rule.activation = max(0.1, rule.activation - DECAY_RATE * 0.1)

    def get_stats(self) -> Dict:
        """Return memory system statistics for visualization."""
        active_episodes = [ep for ep in self.episodic if not ep.compressed]
        compressed_episodes = [ep for ep in self.episodic if ep.compressed]

        return {
            'episodic': {
                'total_stored': self.total_episodes_stored,
                'currently_active': len(active_episodes),
                'currently_compressed': len(compressed_episodes),
                'total_compacted': self.total_episodes_compacted,
                'total_forgotten': self.total_episodes_forgotten,
                'avg_activation': float(np.mean([ep.activation for ep in self.episodic])) if self.episodic else 0,
            },
            'semantic': {
                'total_rules': len(self.semantic),
                'avg_confidence': float(np.mean([r.confidence for r in self.semantic])) if self.semantic else 0,
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

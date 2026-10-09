"""
brain_persistence.py — Saves and restores the full NovaMind brain state.

Serializes:
  - Perception: concept registry
  - Memory: episodic, semantic, procedural
  - World Model: nodes, edges
  - Goals: active goals
  - Metacognition: emotional state, confidence history
  - Learning: stats, novelty tracking
  - Thought history

Format: single JSON file (brain_state.json) + PyTorch weights (substrate.pt)
"""

import json
import time
import os
import threading


BRAIN_DIR = os.path.join(os.path.dirname(__file__), "brainstate")
BRAIN_FILE = os.path.join(BRAIN_DIR, "brain_state.json")
SAVE_INTERVAL = 60  # Save every 60 seconds


def save_brain(mind) -> str:
    """Serialize the full brain state to disk."""
    os.makedirs(BRAIN_DIR, exist_ok=True)

    state = {
        "saved_at": time.time(),
        "cycle_count": mind.cycle_count,

        # Perception: all concept data
        "concepts": [
            {
                "id": c.id,
                "label": c.label,
                "type": c.concept_type,
                "activation": c.activation,
                "access_count": c.access_count,
                "vector": c.hyperbolic_position.tolist() if c.hyperbolic_position is not None else [],
            }
            for c in mind.perception.concept_registry.values()
        ],

        # Memory
        "episodic": [ep.to_dict() for ep in mind.memory.episodic],
        "semantic": [r.to_dict() for r in mind.memory.semantic],
        "procedural": [p.to_dict() for p in mind.memory.procedural],

        # Memory system state (adaptive relevance)
        "memory_state": {
            "emotional_context": mind.memory.current_emotional_context,
            "arousal": mind.memory.current_arousal,
        },

        # Predictive coding state
        "predictive_coding": mind.predictor.get_state(),

        # Active inference state
        "active_inference": mind.actor.get_state(),

        # Motor cortex state
        "motor_cortex": mind.motor.get_state(),

        # World Model
        "world_model": mind.world_model.get_graph_data(),

        # Goals
        "goals": mind.goals.get_state(),

        # Metacognition
        "metacognition": mind.metacognition.get_full_state(),

        # Learning stats
        "learning": mind.learning.get_stats(),

        # Semantic facts cache
        "semantic_facts": mind.semantic_facts[-500:],

        # Thought history
        "thought_history": mind.thought_history,

        # Unanswered questions (curiosities)
        "unanswered_questions": mind.unanswered_questions,
    }

    with open(BRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, default=str)

    size_kb = os.path.getsize(BRAIN_FILE) / 1024
    print(f"[BRAIN] Saved ({size_kb:.1f} KB, cycle {mind.cycle_count})")
    return BRAIN_FILE


def load_brain(mind) -> bool:
    """Restore brain state from disk. Returns True if successful."""
    if not os.path.exists(BRAIN_FILE):
        print("[BRAIN] No saved state found. Starting fresh.")
        return False

    try:
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Restore cycle count
        mind.cycle_count = state.get("cycle_count", 0)

        # Restore semantic facts
        mind.semantic_facts = state.get("semantic_facts", [])

        # Restore thought history
        mind.thought_history = state.get("thought_history", [])

        # Restore unanswered questions
        mind.unanswered_questions = state.get("unanswered_questions", [])

        # Restore memory system emotional state
        mem_state = state.get("memory_state", {})
        mind.memory.current_emotional_context = mem_state.get("emotional_context", 0.0)
        mind.memory.current_arousal = mem_state.get("arousal", 0.0)

        # Restore concepts
        from core.perception import ConceptNode
        import numpy as np

        mind.perception.concept_registry.clear()
        for c_data in state.get("concepts", []):
            label = c_data.get("label")
            c_type = c_data.get("type", "concept")
            concept = ConceptNode(label, c_type)
            concept.id = c_data.get("id", concept.id)
            concept.activation = c_data.get("activation", 1.0)
            concept.access_count = c_data.get("access_count", 0)
            
            vector_list = c_data.get("vector", [])
            if vector_list:
                h_pos = np.array(vector_list)
                concept.hyperbolic_position = h_pos
                if hasattr(mind.perception, 'algebra') and mind.perception.algebra:
                    concept.clifford_vector = mind.perception.algebra.vector(h_pos)
            
            mind.perception.concept_registry[concept.id] = concept

        # Restore semantic memory (rules/facts the mind learned)
        for rule_data in state.get("semantic", []):
            content = rule_data.get("content", {})
            if content:
                rule = mind.memory.store_semantic_rule(content)
                rule.emotional_valence = rule_data.get("emotional_valence", 0.0)
                rule.generalization_power = rule_data.get("generalization_power", 0.0)

        # Restore episodic memory summaries with all adaptive attributes
        for ep_data in state.get("episodic", []):
            content = ep_data.get("content", {})
            if content:
                ep = mind.memory.store_episode(
                    content=content,
                    episode_type=ep_data.get("type", "interaction"),
                    importance=ep_data.get("importance", 0.5),
                    emotional_valence=ep_data.get("emotional_valence", 0.0),
                )
                ep.surprise = ep_data.get("surprise", 0.0)
                ep.consolidation_count = ep_data.get("consolidation_count", 0)
                ep.activation = ep_data.get("activation", 1.0)

        # Restore predictive coding state
        pc_state = state.get("predictive_coding", {})
        if pc_state:
            mind.predictor.precision = pc_state.get("precision", 1.0)
            mind.predictor.prediction_error = pc_state.get("prediction_error", 0.0)
            mind.predictor.error_history.clear()
            for err in pc_state.get("recent_errors", []):
                mind.predictor.error_history.append(err)
            for s in pc_state.get("surprise_history", []):
                mind.predictor.surprise_history.append(s)

        saved_at = state.get("saved_at", 0)
        age_hours = (time.time() - saved_at) / 3600
        print(f"[BRAIN] Restored! cycle={mind.cycle_count}, "
              f"concepts={len(mind.perception.concept_registry)}, "
              f"episodic={len(state.get('episodic', []))}, "
              f"semantic={len(state.get('semantic', []))}, "
              f"age={age_hours:.1f}h ago")
        return True

    except Exception as e:
        print(f"[BRAIN] Failed to load: {e}")
        return False


def start_autosave(mind, interval: int = SAVE_INTERVAL):
    """Start a background thread that saves the brain periodically."""
    def _loop():
        while True:
            time.sleep(interval)
            try:
                save_brain(mind)
            except Exception as e:
                print(f"[BRAIN] Autosave error: {e}")

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print(f"[BRAIN] Autosave started (every {interval}s)")

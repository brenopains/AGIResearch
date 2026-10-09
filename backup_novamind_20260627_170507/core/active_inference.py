import numpy as np
from typing import Dict, List, Optional, Tuple


class ActiveInferenceEngine:
    """
    Active Inference based on the Free Energy Principle.

    Selects actions that minimize expected free energy (surprise):
    1. Enumerate possible actions from world model causal chains
    2. Predict outcomes for each action
    3. Choose action with lowest expected surprise (highest information gain)
    4. Generate action suggestions for the environment
    """

    def __init__(self, exploration_rate: float = 0.2):
        self.exploration_rate = exploration_rate
        self.current_action: Optional[str] = None
        self.action_history: List[Dict] = []
        self._max_history = 100
        self.pending_actions: List[str] = []

    def suggest_action(self, world_model: object, metacognition: object,
                       memory: object, predictor: object) -> Dict:
        """
        Select the next action using active inference.

        Returns an action recommendation with expected outcomes.
        """
        actions = []

        # 1. Exploit: use world model causal chains
        try:
            chains = world_model.get_causal_chains(min_length=2)
            for chain in chains[:5]:
                current = chain[0]
                next_node = chain[1] if len(chain) > 1 else None
                if next_node:
                    edge = world_model.edges.get((current, next_node))
                    if edge:
                        expected_surprise = 1.0 - (edge.strength * edge.confidence)
                        actions.append({
                            "type": "exploit",
                            "action": f"explorar_{next_node}",
                            "description": f"Investigar '{next_node}' (causal: {edge.link_type}, conf: {edge.confidence:.2f})",
                            "expected_surprise": expected_surprise,
                            "target_concept": next_node,
                            "source_concept": current,
                        })
        except Exception:
            pass

        # 2. Explore: address knowledge gaps from metacognition
        try:
            if metacognition:
                confusion = metacognition.state.emotional_state.get("confusion", 0)
                if confusion > 0.3:
                    gaps = metacognition.knowledge_gaps[-3:]
                    for gap in gaps:
                        actions.append({
                            "type": "explore",
                            "action": f"investigar_{gap.get('topic', 'desconhecido')}",
                            "description": f"Esclarecer dúvida: {gap.get('description', '')[:80]}",
                            "expected_surprise": 0.7,
                        })
        except Exception:
            pass

        # 3. Curiosity-driven: high prediction error areas
        try:
            if predictor:
                error = predictor.prediction_error
                if error > 0.4:
                    actions.append({
                        "type": "curiosity",
                        "action": "revisitar_conceito",
                        "description": f"Revisitar tópico (erro de predição: {error:.2f})",
                        "expected_surprise": error,
                    })
        except Exception:
            pass

        # Select best action: lowest expected free energy
        if actions:
            # Mix exploration: sometimes pick randomly
            if np.random.random() < self.exploration_rate:
                best = np.random.choice(actions)
            else:
                actions.sort(key=lambda a: a.get("expected_surprise", 1.0))
                best = actions[0]

            best["selected_by"] = "active_inference"
            self.current_action = best["action"]
            self.action_history.append({
                "time": __import__("time").time(),
                "action": best["action"],
                "type": best["type"],
                "expected_surprise": best.get("expected_surprise", 0.5),
            })
            if len(self.action_history) > self._max_history:
                self.action_history = self.action_history[-self._max_history:]
            return best

        return {
            "type": "idle",
            "action": "observar",
            "description": "Aguardando novos estímulos",
            "expected_surprise": 0.5,
        }

    def get_state(self) -> Dict:
        return {
            "current_action": self.current_action,
            "exploration_rate": self.exploration_rate,
            "recent_actions": self.action_history[-5:],
            "pending_actions": self.pending_actions[-3:],
        }

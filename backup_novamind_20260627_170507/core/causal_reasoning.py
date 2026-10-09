import numpy as np
import time
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque


class CausalReasoningEngine:
    def __init__(self, max_horizon: int = 3):
        self.max_horizon = max_horizon
        self.counterfactual_traces: deque = deque(maxlen=50)
        self.planning_cache: Dict[str, List[Dict]] = {}

    def simulate_counterfactual(self, world_model, concept_a: str, concept_b: str,
                                 hypothetical_value: float = 0.9) -> Dict:
        result = world_model.query_counterfactual(
            target=concept_b,
            actual={concept_a: 1.0 - hypothetical_value},
            hypothetical={concept_a: hypothetical_value}
        )
        trace = {
            "timestamp": time.time(),
            "type": "counterfactual",
            "concept_a": concept_a,
            "concept_b": concept_b,
            "hypothetical_value": hypothetical_value,
            "actual": result.get("actual_outcome", 0.5),
            "counterfactual": result.get("counterfactual_outcome", 0.5),
            "attribution": result.get("causal_attribution", 0),
            "confidence": result.get("confidence", 0),
        }
        self.counterfactual_traces.append(trace)
        return trace

    def plan(self, world_model, current_concepts: List[str],
             goal_concept: str, horizon: int = None) -> List[str]:
        if horizon is None:
            horizon = self.max_horizon
        cache_key = f"{','.join(sorted(current_concepts))}|{goal_concept}|{horizon}"
        if cache_key in self.planning_cache:
            return self.planning_cache[cache_key]

        chains = world_model.get_causal_chains(min_length=2)
        scored_plans = []

        for chain in chains:
            if chain[0] in current_concepts and goal_concept in chain:
                plan = chain[1:chain.index(goal_concept) + 1]
                if len(plan) <= horizon:
                    score = 0
                    for i in range(len(plan) - 1):
                        edge = world_model.edges.get((plan[i], plan[i + 1]))
                        if edge:
                            score += edge.strength * edge.confidence
                    scored_plans.append((score / max(1, len(plan)), plan))

        scored_plans.sort(key=lambda x: x[0], reverse=True)
        best_plan = scored_plans[0][1] if scored_plans else []
        self.planning_cache[cache_key] = best_plan
        if len(self.planning_cache) > 100:
            for k in list(self.planning_cache)[:50]:
                del self.planning_cache[k]
        return best_plan

    def what_if(self, world_model, intervention: Dict[str, float],
                target: str) -> Dict:
        return world_model.query_intervention(target, intervention)

    def causal_explain(self, world_model, effect: str,
                       candidates: List[str]) -> List[Dict]:
        explanations = []
        for cause in candidates:
            try:
                result = world_model.query_association(effect, {cause: 1.0})
                explanations.append({
                    "cause": cause,
                    "effect": effect,
                    "probability": result.get("probability", 0.5),
                    "confidence": result.get("confidence", 0),
                    "path": result.get("path", []),
                })
            except Exception:
                continue
        explanations.sort(key=lambda x: x["probability"] * x["confidence"], reverse=True)
        return explanations[:5]

    def get_state(self) -> Dict:
        return {
            "counterfactual_traces": len(self.counterfactual_traces),
            "planning_cache_size": len(self.planning_cache),
            "max_horizon": self.max_horizon,
            "last_counterfactual": self.counterfactual_traces[-1] if self.counterfactual_traces else None,
        }

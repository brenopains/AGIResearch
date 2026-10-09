"""
GoalEmergence — Metas que Emergem do Conhecimento

Nao sao goals pre-definidos.
Sao objetivos que brotam de:
  - Vazios estruturais no UnknownSpace (conceitos proximos sem aresta)
  - Paradigmas que contradizem worldviews
  - Teorias com baixo poder explicativo
  - Dominios com alta incerteza

Cada goal tem:
  - descricao
  - origem (que observacao gerou)
  - potencial (quao promissor)
  - status (active, completed, abandoned)
"""
import time, uuid, hashlib, math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class EmergentGoal:
    id: str = ""
    description: str = ""
    origin: str = ""            # "unknown_space", "paradigm_conflict", "low_explanatory", "high_uncertainty"
    source_concept: str = ""
    target_concept: str = ""
    potential: float = 0.5      # quao promissor
    urgency: float = 0.3        # quao urgente
    status: str = "active"      # active, completed, abandoned
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "description": self.description[:80],
                "origin": self.origin, "potential": round(self.potential, 2),
                "urgency": round(self.urgency, 2), "status": self.status}


class GoalEmergence:
    """Faz with que o UnknownSpace e outras fontes gerem objetivos."""

    def __init__(self, unknown_space=None, cge=None):
        self.unknown = unknown_space
        self.cge = cge
        self.goals: Dict[str, EmergentGoal] = {}
        self._history: List[Dict] = []
        self._max_goals: int = 20

    def tick(self):
        """Ciclo de emergencia de metas."""
        new_goals = []

        # 1. Metas do UnknownSpace (vazios estruturais)
        if self.unknown:
            frontier = self.unknown.get_frontier(5)
            for region in frontier:
                a, b = region.get("concept_a", "?"), region.get("concept_b", "?")
                gid = f"goal_{hashlib.md5((a+b).encode()).hexdigest()[:8]}"
                if gid not in self.goals and len(self.goals) < self._max_goals:
                    goal = EmergentGoal(
                        id=gid,
                        description=f"Conectar {a} com {b}",
                        origin="unknown_space",
                        source_concept=a, target_concept=b,
                        potential=region.get("potential", 0.5),
                        urgency=0.3 + region.get("potential", 0.5) * 0.4,
                    )
                    self.goals[gid] = goal
                    new_goals.append(goal)

        # 2. Verificar se algum goal foi alcancado (aresta criada no CGE)
        self._check_completed()

        return new_goals

    def _check_completed(self):
        """Verifica se goals do UnknownSpace foram alcancados (aresta criada)."""
        if self.cge is None:
            return
        for gid, goal in list(self.goals.items()):
            if goal.status != "active":
                continue
            if goal.origin == "unknown_space":
                # Verificar se aresta entre source e target foi criada
                for edge in self.cge.edges:
                    if (edge["from"] == goal.source_concept and edge["to"] == goal.target_concept) or \
                       (edge["from"] == goal.target_concept and edge["to"] == goal.source_concept):
                        goal.status = "completed"
                        goal.completed_at = time.time()
                        self._history.append({
                            "time": time.time(), "goal": goal.description[:60],
                            "status": "completed", "origin": goal.origin,
                        })
                        break

    def get_active_goals(self) -> List[EmergentGoal]:
        return [g for g in self.goals.values() if g.status == "active"]

    def get_most_urgent(self) -> Optional[EmergentGoal]:
        active = self.get_active_goals()
        if not active:
            return None
        return max(active, key=lambda g: g.urgency * g.potential)

    def get_state(self) -> Dict:
        active = self.get_active_goals()
        return {
            "total": len(self.goals),
            "active": len(active),
            "completed": sum(1 for g in self.goals.values() if g.status == "completed"),
            "most_urgent": self.get_most_urgent().to_dict() if self.get_most_urgent() else None,
            "recent_history": self._history[-3:] if self._history else [],
        }

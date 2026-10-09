"""
TeachingEngine — Aprende a Ensinar

Nao apenas responde.
APRENDE como cada perfil de pessoa aprende melhor cada conceito.

Exemplo de conhecimento descoberto:
  "Para explicar Relatividade, 90% entendem melhor com analogia do lencol"
  "Para explicar JavaScript, programadores preferem exemplos de codigo"
  "Para explicar gravidade, criancas entendem melhor com desenhos"
"""
import time, hashlib, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TeachingStrategy:
    id: str = ""
    concept: str = ""
    profile_id: str = "leigo"
    format: str = "analogy"      # "analogy", "code", "visual", "text", "video", "story"
    template: str = ""            # como explicar
    effectiveness: float = 0.5   # 0..1, medida de sucesso
    n_trials: int = 0
    confidence: float = 0.3
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"concept": self.concept, "profile": self.profile_id,
                "format": self.format, "effectiveness": round(self.effectiveness, 2)}


class TeachingEngine:
    """Motor didatico.

    Aprende a melhor forma de explicar cada conceito para cada perfil.
    """

    def __init__(self):
        self.strategies: Dict[str, TeachingStrategy] = {}
        self._history: List[Dict] = []

    def recommend(self, concept: str, profile_id: str = "leigo") -> TeachingStrategy:
        """Recomenda a melhor estrategia para explicar um conceito."""
        candidates = [s for s in self.strategies.values()
                     if s.concept == concept and s.profile_id == profile_id]
        if not candidates:
            # Fallback: estrategia generica
            return TeachingStrategy(
                id="default", concept=concept, profile_id=profile_id,
                format="analogy", effectiveness=0.3)

        return max(candidates, key=lambda s: s.effectiveness * s.confidence)

    def record_outcome(self, concept: str, profile_id: str, format: str,
                       success: bool):
        """Registra o resultado de uma explicacao."""
        sid = f"teach_{hashlib.md5(f'{concept}_{profile_id}_{format}'.encode()).hexdigest()[:8]}"
        if sid not in self.strategies:
            self.strategies[sid] = TeachingStrategy(
                id=sid, concept=concept, profile_id=profile_id, format=format)
        s = self.strategies[sid]
        s.n_trials += 1
        if success:
            s.effectiveness = (s.effectiveness * (s.n_trials - 1) + 1.0) / s.n_trials
        else:
            s.effectiveness = (s.effectiveness * (s.n_trials - 1)) / s.n_trials
        s.confidence = min(1.0, s.confidence + 0.1 if success else s.confidence - 0.05)
        self._history.append({"time": time.time(), "concept": concept,
                              "profile": profile_id, "format": format, "success": success})

    def get_state(self) -> Dict:
        return {"n_strategies": len(self.strategies),
                "recent": self._history[-3:] if self._history else []}

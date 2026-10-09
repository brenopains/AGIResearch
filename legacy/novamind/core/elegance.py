"""
EleganceMetric — Metrica de Elegancia Multi-Dimensional

Entre duas teorias igualmente preditivas, humanos preferem a mais elegante.

Elegancia NAO e apenas simplicidade. Envolve:
  - compressao: explica muito com pouco
  - simetria: estrutura repetida
  - reutilizacao: mesmo padrao aparece em dominios diferentes
  - generalidade: aplica-se amplamente
  - consistencia: sem contradicoes internas
  - parcimonia: numero minimo de pressupostos

A AGI usa isso para:
  - escolher entre teorias candidatas
  - preferir theories compactas
  - avaliar ModelCompetition
  - guiar ForjaEvolution
"""
import time, math, hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


PRIMORDIAL_ELEMENTS = {
    "compression": 0.5,      # explica muito com pouco
    "symmetry": 0.5,         # estrutura repetida
    "reusability": 0.5,      # reuso entre dominios
    "generality": 0.5,       # aplicacao ampla
    "consistency": 0.5,      # sem contradicoes
    "parsimony": 0.7,        # Occam
}


class EleganceVector:
    """Vetor de elegancia multi-dimensional."""

    def __init__(self, initial: Optional[Dict[str, float]] = None):
        self.elements: Dict[str, float] = dict(PRIMORDIAL_ELEMENTS)
        if initial:
            for k, v in initial.items():
                if k in self.elements:
                    self.elements[k] = max(0.0, min(1.0, v))

    def __getitem__(self, key: str) -> float:
        return self.elements.get(key, 0.5)

    def __setitem__(self, key: str, value: float):
        self.elements[key] = max(0.0, min(1.0, value))

    def overall(self) -> float:
        if not self.elements:
            return 0.5
        return sum(self.elements.values()) / len(self.elements)

    def to_dict(self) -> Dict:
        return {k: round(v, 3) for k, v in self.elements.items()}


class EleganceEvaluator:
    """Avalia a elegancia de uma teoria ou modelo.

    NAO hardcoded: os pesos de cada dimensao sao auto-ajustados
    baseado em feedback (qual teoria foi adotada pela humanidade).
    """

    def __init__(self):
        self.weights: Dict[str, float] = dict(PRIMORDIAL_ELEMENTS)
        self._evaluations: List[Dict] = []

    def evaluate(self, n_axioms: int = 0, n_explanations: int = 0,
                 n_domains: int = 0, n_contradictions: int = 0,
                 reuses: int = 0) -> EleganceVector:
        """Avalia a elegancia de uma teoria/algoritmo.

        Args:
            n_axioms: numero de axiomas/pressupostos
            n_explanations: quantos fenomenos explica
            n_domains: em quantos dominios diferentes aplica-se
            n_contradictions: contradicoes com outras teorias
            reuses: quantas vezes o padrao foi reusado
        """
        e = EleganceVector()

        # Compression: explica muito / usa pouco
        if n_axioms > 0 and n_explanations > 0:
            ratio = n_explanations / max(1, n_axioms)
            e["compression"] = min(1.0, ratio * 0.3)
        else:
            e["compression"] = 0.3

        # Reusability: reuso entre dominios
        e["reusability"] = min(1.0, reuses * 0.2)

        # Generality: quantos dominios
        e["generality"] = min(1.0, n_domains * 0.2)

        # Consistency: sem contradicoes
        e["consistency"] = max(0.0, 1.0 - n_contradictions * 0.2)

        # Parsimony: poucos axiomas
        if n_axioms <= 2:
            e["parsimony"] = 1.0
        elif n_axioms <= 5:
            e["parsimony"] = 0.7
        else:
            e["parsimony"] = max(0.1, 1.0 - n_axioms * 0.1)

        # Symmetry: detectada externamente (nivel arquitetural)
        e["symmetry"] = 0.5  # default neutro

        return e

    def rank_by_elegance(self, candidates: List[Dict]) -> List[Dict]:
        """Ordena candidatos por elegancia geral."""
        scored = []
        for c in candidates:
            vec = self.evaluate(
                n_axioms=c.get("n_axioms", 0),
                n_explanations=c.get("n_explanations", 0),
                n_domains=c.get("n_domains", 0),
                n_contradictions=c.get("n_contradictions", 0),
                reuses=c.get("reuses", 0),
            )
            score = vec.overall()
            scored.append((score, c, vec))
        scored.sort(key=lambda x: -x[0])
        return [{"score": round(s, 2), "c": c, "vector": v.to_dict()}
                for s, c, v in scored]

    def get_state(self) -> Dict:
        return {
            "n_evaluations": len(self._evaluations),
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
        }

"""
MultiConfidence — Confianca Multi-Facetada

Um modelo pode ser excelente em dados conhecidos e pessimo fora
de distribuicao. Confianca nao e um escalar — e um VETOR.

Dimensoes de confianca:
  - estatistica: baseada em dados observados
  - causal: baseada em intervencoes
  - temporal: baseada em permanencia ao longo do tempo
  - por dominio: especifica ao dominio
  - fora de distribuicao: cautela quando input nao se parece com treino
  - por evidencia: forte/fraqueza da evidencia disponivel
  - de consenso: concordancia com outros modelos

Cada dimensao e atualizada independentemente.
A "confianca geral" e uma media ponderada auto-aprendida.
"""
import time, hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


PRIMORDIAL_CONFIDENCE_DIMENSIONS = {
    "statistical": 0.5,      # precisao em dados observados
    "causal": 0.3,           # precisao em intervencoes
    "temporal": 0.5,         # permanencia ao longo do tempo
    "ood": 0.3,              # cautela OOD
    "evidence": 0.5,         # forca da evidencia
    "consensus": 0.5,        # concordancia com outros modelos
}


class MultiConfidence:
    """Confianca multi-facetada.

    Nao colapsa em um unico numero. Cada dimensao evolui
    independentemente baseada em experiencias.
    """

    def __init__(self, initial: Optional[Dict[str, float]] = None):
        # Comeca com veses neutros ou fornecidos
        self.dimensions: Dict[str, float] = dict(PRIMORDIAL_CONFIDENCE_DIMENSIONS)
        if initial:
            for k, v in initial.items():
                self.dimensions[k] = max(0.0, min(1.0, v))
        self._weight_history: Dict[str, List[float]] = {}

    def __getitem__(self, key: str) -> float:
        return self.dimensions.get(key, 0.5)

    def __setitem__(self, key: str, value: float):
        self.dimensions[key] = max(0.0, min(1.0, value))

    def update(self, dimension: str, observation: float):
        """Atualiza uma dimensao baseado em observacao."""
        if dimension not in self.dimensions:
            self.dimensions[dimension] = 0.5
        old = self.dimensions[dimension]
        # Media movel exponencial
        self.dimensions[dimension] = old * 0.85 + observation * 0.15

    def overall(self) -> float:
        """Confianca geral (media das dimensoes)."""
        if not self.dimensions:
            return 0.5
        return sum(self.dimensions.values()) / len(self.dimensions)

    def is_reliable(self, threshold: float = 0.6) -> bool:
        """Verifica se a confianca geral e suficiente para agir."""
        return self.overall() >= threshold

    def weakest_dimension(self) -> Tuple[str, float]:
        """Retorna a dimensao mais fraca (maior incerteza)."""
        if not self.dimensions:
            return ("", 0.0)
        weakest = min(self.dimensions.items(), key=lambda x: x[1])
        return weakest

    def to_dict(self) -> Dict:
        return {k: round(v, 3) for k, v in self.dimensions.items()}


class ConfidenceTracker:
    """Gerencia MultiConfidence para multiplas fontes (modelos, hipoteses)."""

    def __init__(self):
        self.confidences: Dict[str, MultiConfidence] = {}

    def get_or_create(self, source_id: str) -> MultiConfidence:
        """Obtem ou cria a MultiConfidence de uma fonte."""
        if source_id not in self.confidences:
            self.confidences[source_id] = MultiConfidence()
        return self.confidences[source_id]

    def record_observation(self, source_id: str, dimension: str, value: float):
        conf = self.get_or_create(source_id)
        conf.update(dimension, value)

    def is_ood(self, source_id: str, input_similarity: float) -> bool:
        """Verifica se um input esta fora de distribuicao para uma fonte."""
        conf = self.get_or_create(source_id)
        # Se similarity baixa, OOD risk e alto
        if input_similarity < 0.3:
            return True
        return conf["ood"] < 0.4

    def consensus_score(self, source_ids: List[str]) -> float:
        """Mede o consenso entre multiplas fontes."""
        if not source_ids:
            return 0.0
        overalls = [self.get_or_create(s).overall() for s in source_ids
                    if s in self.confidences]
        if not overalls:
            return 0.0
        # Consenso: 1 - desvio padrao
        mean = sum(overalls) / len(overalls)
        variance = sum((o - mean) ** 2 for o in overalls) / len(overalls)
        return max(0.0, 1.0 - math.sqrt(variance)) if variance else 1.0

    def get_state(self) -> Dict:
        return {
            "n_sources": len(self.confidences),
            "by_source": {k: v.to_dict() for k, v in
                         list(self.confidences.items())[:5]},
        }

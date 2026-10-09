"""
CFEBridge — Conecta o CFE ao ProgramSynthesizer.

O CFE observa input/output, descobre invariantes, classifica o tipo
de transformacao, e INFORMA o synthesizer sobre quais primitives
sao mais viaveis — em vez de busca cega.

Isso transforma:
  Busca: 12^5 = 248.832 programas
  Guiada: conceito + 3^5 = 243 programas (1000x menos)
"""
from typing import Optional, Set, Dict, List
import numpy as np

from .invariant_discovery import InvariantDiscovery
from .concept_formation import ConceptFormation


# Mapeamento: invariante -> tipos de primitiva preferidos
CONCEPT_TO_PRIMITIVES: Dict[str, Set[str]] = {
    "expansion": {
        "transform", "composite",
        "tile", "copy", "resize", "scale", "expand",
        "add",
    },
    "contraction": {
        "crop", "compress", "extract_bbox", "pad",
        "remove",
    },
    "reflection": {
        "flip_horizontal", "flip_vertical",
        "rotate_90", "rotate_180", "rotate_270",
        "transpose",
    },
    "symmetry": {
        "flip_horizontal", "flip_vertical",
        "rotate_90", "rotate_180",
        "identity",
        "move",
    },
    "causal_chain": {
        "move", "recolor", "remove", "add",
        "transform", "composite",
    },
    "spatial_transformation": {
        "move", "rotate_90", "rotate_180", "rotate_270",
        "flip_horizontal", "flip_vertical", "transpose",
    },
    "branching": {
        "tile", "copy", "composite", "transform",
        "add",
    },
}

# Ordem de prioridade dos conceitos (mais especifico primeiro)
CONCEPT_PRIORITY = [
    "reflection", "symmetry", "expansion", "contraction",
    "branching", "spatial_transformation", "causal_chain",
]


class CFEBridge:
    """Ponte entre CFE e ProgramSynthesizer.

    Uso:
        bridge = CFEBridge()
        filter_set = bridge.get_primitive_filter(inp, out)
        programs = synthesizer.synthesize(inp, out, primitive_filter=filter_set)
    """

    def __init__(self, discovery: Optional[InvariantDiscovery] = None,
                 formation: Optional[ConceptFormation] = None):
        self.discovery = discovery or InvariantDiscovery()
        self.formation = formation or ConceptFormation()

    def get_concept_hint(self, input_grid: np.ndarray,
                         output_grid: np.ndarray) -> Optional[str]:
        """Classifica o tipo de transformacao entre input e output.

        Returns: nome do conceito (ex: "expansion", "reflection"), ou None.
        """
        # Observa a transformacao
        self.discovery.observe_transformation(input_grid, output_grid,
                                               "query", domain="arc")

        # Busca invariantes que se aplicam
        invariants = self.discovery.get_invariants(min_confidence=0.1)

        if not invariants:
            return None

        # Pega o invariante com a assinatura mais especifica
        # (priorizando os primeiros na lista CONCEPT_PRIORITY)
        best = None
        best_priority = 999

        for inv in invariants:
            try:
                idx = CONCEPT_PRIORITY.index(inv.structure_type)
                if idx < best_priority:
                    best_priority = idx
                    best = inv.structure_type
            except ValueError:
                pass

        return best

    def get_primitive_filter(self, input_grid: np.ndarray,
                             output_grid: np.ndarray) -> Optional[Set[str]]:
        """Retorna filtro de primitives para o ProgramSynthesizer.

        Returns: set de structure_types para filtrar, ou None (busca completa).
        """
        concept = self.get_concept_hint(input_grid, output_grid)
        if concept is None:
            return None

        primitives = CONCEPT_TO_PRIMITIVES.get(concept, set())
        if not primitives:
            return None

        return primitives

    def get_concept_confidence(self, input_grid: np.ndarray,
                               output_grid: np.ndarray,
                               concept_name: str) -> float:
        """Quanto um conceito especifico se aplica a este par."""
        inv = self.discovery.get_by_type(concept_name)
        if inv is None:
            return 0.0
        return inv.confidence

    def get_stats(self) -> dict:
        return {
            "discovery": self.discovery.get_stats(),
            "concepts": list(CONCEPT_TO_PRIMITIVES.keys()),
            "total_mappings": sum(len(v) for v in CONCEPT_TO_PRIMITIVES.values()),
        }

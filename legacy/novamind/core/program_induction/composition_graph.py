"""
CompositionGraph — Aprende pesos de composicao entre primitives da experiencia.

Nao e uma lista fixa. Cada vez que o sistema encontra uma composicao
valida (ex: recolor -> move resolveu uma task), o peso da borda
recolor->move aumenta.

Durante a busca, o beam prefere bordas com peso alto, podando
bordas com peso baixo ou zero (nunca funcionaram juntas).

O grafo e armazenado de forma compativel com o CGE:
  - nodes = structure_types
  - edges = composicoes com peso e historico
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class CompositionEdge:
    """Uma borda no grafo de composicao: A -> B."""
    from_type: str
    to_type: str
    weight: float = 0.0  # learned transition probability
    success_count: int = 0
    attempt_count: int = 0
    last_success: float = 0.0

    @property
    def probability(self) -> float:
        return self.success_count / max(1, self.attempt_count)

    def record_success(self):
        self.success_count += 1
        self.attempt_count += 1
        self.last_success = time.time()
        self.weight = self.probability

    def record_attempt(self, succeeded: bool):
        self.attempt_count += 1
        if succeeded:
            self.success_count += 1
            self.last_success = time.time()
        self.weight = self.probability

    def to_dict(self) -> dict:
        return {
            "from": self.from_type,
            "to": self.to_type,
            "weight": round(self.weight, 3),
            "success_rate": round(self.probability, 3),
            "attempts": self.attempt_count,
        }


class CompositionGraph:
    """Grafo de composicoes entre primitives, aprendido da experiencia.

    Cresce organicamente. Nenhuma aresta pre-definida.

    Uso:
        graph = CompositionGraph()
        graph.record_composition("recolor", "move", succeeded=True)
        best_next = graph.get_best_next("recolor")  # -> "move"
    """

    def __init__(self):
        self.edges: Dict[Tuple[str, str], CompositionEdge] = {}
        self.node_types: set = set()

    def record_composition(self, from_type: str, to_type: str, succeeded: bool = True):
        """Registra que from_type -> to_type foi tentado."""
        self.node_types.add(from_type)
        self.node_types.add(to_type)
        key = (from_type, to_type)
        if key not in self.edges:
            self.edges[key] = CompositionEdge(from_type=from_type, to_type=to_type)
        self.edges[key].record_attempt(succeeded=succeeded)

    def record_program(self, structure_types: List[str], succeeded: bool = True):
        """Registra um programa inteiro (sequencia de tipos)."""
        for i in range(len(structure_types) - 1):
            self.record_composition(structure_types[i], structure_types[i + 1], succeeded)

    def get_weight(self, from_type: str, to_type: str) -> float:
        """Retorna peso aprendido da borda (0 se nunca tentada)."""
        key = (from_type, to_type)
        if key in self.edges:
            return self.edges[key].weight
        return 0.0

    def get_best_next(self, from_type: str, min_weight: float = 0.0) -> List[str]:
        """Retorna os melhores proximos tipos ordenados por peso."""
        candidates = []
        for (frm, to), edge in self.edges.items():
            if frm == from_type and edge.weight >= min_weight:
                candidates.append((to, edge.weight))
        candidates.sort(key=lambda x: -x[1])
        return [t for t, w in candidates]

    def get_predecessors(self, to_type: str) -> List[str]:
        """Retorna tipos que costumam preceder este tipo."""
        return [
            frm for (frm, to) in self.edges if to == to_type
        ]

    def get_stats(self) -> dict:
        """Estatisticas do grafo."""
        total_attempts = sum(e.attempt_count for e in self.edges.values())
        total_successes = sum(e.success_count for e in self.edges.values())
        return {
            "n_edges": len(self.edges),
            "n_nodes": len(self.node_types),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "avg_weight": round(total_successes / max(1, total_attempts), 3) if total_attempts > 0 else 0,
        }

    def get_edges(self, min_weight: float = 0.0) -> List[dict]:
        return [
            e.to_dict() for e in sorted(self.edges.values(), key=lambda x: -x.weight)
            if e.weight >= min_weight
        ]

    def to_cge_compatible(self) -> dict:
        """Exporta para formato compativel com CGE."""
        return {
            "graph_type": "composition_graph",
            "nodes": list(self.node_types),
            "edges": [
                {
                    "from": e.from_type,
                    "to": e.to_type,
                    "weight": e.weight,
                    "type": "composes_to",
                }
                for e in self.edges.values() if e.weight > 0.1
            ],
        }

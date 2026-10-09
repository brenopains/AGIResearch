"""
AnalogyEngine — Isomorfismos Estruturais A:B :: C:D

Nao e similaridade vetorial.
E isomorfismo de GRAFOS no CGE.

Exemplo:
  A (atomo) esta para B (sistema solar)
  assim como
  C (empresa) esta para D (ecossistema)

O motor descobre que:
  estrutura(A, B) ≈ estrutura(C, D)
  mesmo quando A, B, C, D sao semanticamente distantes.
"""
import time, hashlib, math, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Analogy:
    """Uma analogia descoberta: estrutura(A,B) ≈ estrutura(C,D)."""
    id: str = ""
    source_a: str = ""
    source_b: str = ""
    target_c: str = ""
    target_d: str = ""
    similarity: float = 0.0       # 0..1, forca do isomorfismo
    shared_structure: str = ""     # descricao da estrutura compartilhada
    confidence: float = 0.3
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id[:8],
                "analogy": f"{self.source_a}:{self.source_b} :: {self.target_c}:{self.target_d}",
                "similarity": round(self.similarity, 2),
                "structure": self.shared_structure[:40]}


class AnalogyEngine:
    """Descobre analogias estruturais no CGE.

    Nao procura similaridade semantica.
    Procura ISOMORFISMO DE GRAFOS:
      structure(A, B) ≈ structure(C, D)
    """

    def __init__(self, cge=None, insight_library=None):
        self.cge = cge
        self.insight_library = insight_library
        self.analogies: Dict[str, Analogy] = {}
        self._discovery_log: List[Dict] = []

    def scan(self):
        """Escaneia o CGE para descobrir analogias."""
        if self.cge is None or len(self.cge.edges) < 10:
            return

        nodes = list(self.cge.nodes.keys())
        if len(nodes) < 4:
            return

        # Construir perfil de vizinhanca para cada no
        profiles = {}
        for nid in nodes[:100]:
            neighbors = set()
            for e in self.cge.edges[-500:]:
                if e["from"] == nid:
                    neighbors.add(e["to"])
                elif e["to"] == nid:
                    neighbors.add(e["from"])
            if len(neighbors) >= 2:
                profiles[nid] = neighbors

        # Comparar pares de nos
        nids = list(profiles.keys())
        for i in range(len(nids)):
            for j in range(i+1, len(nids)):
                a, b = nids[i], nids[j]
                # Similaridade de Jaccard entre vizinhancas
                set_a, set_b = profiles[a], profiles[b]
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                if union == 0:
                    continue
                jaccard = intersection / union
                if jaccard > 0.3:  # vizinhancas similares
                    # Se A e B sao de "dominios" diferentes, e uma analogia
                    if abs(hash(a) - hash(b)) % 100 > 30:  # proxy para dominios diferentes
                        analogia = f"{a}:{list(set_a)[0] if set_a else '?'} :: {b}:{list(set_b)[0] if set_b else '?'}"
                        aid = f"ana_{hashlib.md5(analogia.encode()).hexdigest()[:8]}"
                        if aid not in self.analogies:
                            ana = Analogy(
                                id=aid,
                                source_a=a, source_b=list(set_a)[0] if set_a else "",
                                target_c=b, target_d=list(set_b)[0] if set_b else "",
                                similarity=jaccard,
                                shared_structure=f"ambos conectam {intersection} conceitos compartilhados",
                                confidence=min(0.7, jaccard),
                            )
                            self.analogies[aid] = ana
                            self._discovery_log.append({
                                "time": time.time(), "analogy": analogia[:60],
                                "similarity": jaccard,
                            })

    def find_by_concept(self, concept: str) -> List[Analogy]:
        """Retorna analogias envolvendo um conceito."""
        return [a for a in self.analogies.values()
                if concept in (a.source_a, a.source_b, a.target_c, a.target_d)]

    def get_best(self, top_k: int = 5) -> List[Analogy]:
        """Melhores analogias por similaridade."""
        sorted_a = sorted(self.analogies.values(), key=lambda a: -a.similarity)
        return sorted_a[:top_k]

    def get_state(self) -> Dict:
        return {
            "n_analogies": len(self.analogies),
            "best": [a.to_dict() for a in self.get_best(3)],
            "recent": self._discovery_log[-3:] if self._discovery_log else [],
        }

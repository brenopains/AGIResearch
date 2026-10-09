"""
CognitiveTension — Pressao Interna por Incoerencia (≠ Curiosidade)

Curiosidade: "o que e isso?" (novidade)
Tensao: "isso nao faz sentido" (incoerencia)

Diferenca crucial:
  - Erro de predicao: "esperava X, aconteceu Y" (superficial)
  - Tensao cognitiva: acumulo de micro-incoerencias que formam
    uma pressao interna ate que a AGI precise resolver.

Exemplos de tensao:
  "Teoria A explica X, teoria B explica Y, mas ambas se contradizem"
  "Esse padrao aparece em 3 dominios com nomes diferentes"
  "Prevejo corretamente mas nao sei explicar por que"

Acumulo de tensao:
  - Baixa: operacao normal
  - Media: Maestro prioriza resolucao
  - Alta: domina atencao, forcado a refletir
  - Critica: disruptura, abandona paradigma
"""
import time, math, random, hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TensionSource:
    """Uma fonte de tensao cognitiva."""
    id: str = ""
    description: str = ""
    source_type: str = ""       # "contradiction", "incomplete", "incoherent", "unexplained"
    strength: float = 0.0       # 0..1, magnitude desta tensao
    domain: str = "general"
    concepts_involved: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0

    @property
    def age_weighted_strength(self) -> float:
        age = (time.time() - self.created_at) / 3600  # horas
        decay = math.exp(-age * 0.1)
        return self.strength * decay


class CognitiveTension:
    """Sistema de tensao cognitiva.

    Acumula micro-incoerencias. Quando passa do threshold,
    o Maestro e forcado a mudar de modo para resolver.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.sources: Dict[str, TensionSource] = {}
        self._history: List[Dict] = []

    def add_tension(self, description: str, source_type: str = "contradiction",
                    strength: float = 0.3, domain: str = "general",
                    concepts=None) -> str:
        """Adiciona uma fonte de tensao."""
        tid = f"tension_{hashlib.md5(description.encode()).hexdigest()[:8]}"
        if tid in self.sources:
            self.sources[tid].strength = min(1.0, self.sources[tid].strength + strength * 0.3)
            return tid
        source = TensionSource(
            id=tid, description=description, source_type=source_type,
            strength=min(1.0, strength), domain=domain,
            concepts_involved=concepts or [],
        )
        self.sources[tid] = source
        self._history.append({"action": "add", "type": source_type,
                              "desc": description[:40], "strength": strength,
                              "time": time.time()})
        return tid

    def detect_from_cge(self):
        """Escaneia o CGE para detectar fontes de tensao."""
        if self.cge is None:
            return
        # 1. Contradicoes: conceitos com multiplas edges causais conflitantes
        node_edges = defaultdict(list)
        for e in self.cge.edges[-200:]:
            f, t = e.get("from", ""), e.get("to", "")
            if f and t:
                node_edges[f].append(t)
                node_edges[t].append(f)
        for node, neighbors in node_edges.items():
            if len(neighbors) > 5:
                # Muitas conexoes sem organizacao = tensao potencial
                self.add_tension(
                    f"{node} tem {len(neighbors)} conexoes sem hierarquia clara",
                    source_type="incoherent", strength=0.2,
                    domain=node[:20], concepts=[node] + neighbors[:3])

    def get_total_tension(self) -> float:
        """Tensao total acumulada (0..1)."""
        if not self.sources:
            return 0.0
        raw = sum(s.strength for s in self.sources.values() if s.resolved_at == 0)
        return min(1.0, raw * 0.1)

    def get_pressing_tensions(self, top_k: int = 3) -> List[TensionSource]:
        """Tensoes mais urgentes."""
        unresolved = [s for s in self.sources.values() if s.resolved_at == 0]
        unresolved.sort(key=lambda s: -s.age_weighted_strength)
        return unresolved[:top_k]

    def resolve(self, tension_id: str):
        """Resolve uma tensao."""
        s = self.sources.get(tension_id)
        if s:
            s.resolved_at = time.time()
            self._history.append({"action": "resolve", "id": tension_id,
                                  "desc": s.description[:40], "time": time.time()})

    def get_state(self) -> Dict:
        return {
            "total_tension": round(self.get_total_tension(), 2),
            "n_sources": len(self.sources),
            "pressing": [{"desc": s.description[:40], "strength": round(s.strength, 2),
                          "type": s.source_type} for s in self.get_pressing_tensions(3)],
        }

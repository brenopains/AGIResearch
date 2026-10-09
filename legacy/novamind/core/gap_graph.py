"""
GapGraph — Grafo Unificado de Lacunas Cognitivas

UnknownSpace, ExplainabilitySpace, CognitiveTension e FailedTrace
trabalham sobre LACUNAS. Este modulo unifica todos.

Cada Gap e uma lacuna com atributos:
  - tipo: "unknown", "unexplained", "tension", "failure"
  - descricao
  - dominio
  - importancia
  - urgencia
  - resolvido ou nao
  - conexoes com conceitos no CGE

Modulos especializados sao apenas projecoes do GapGraph.
"""
import time, uuid, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Gap:
    """Uma lacuna no conhecimento da AGI."""
    id: str = ""
    type: str = "unknown"       # "unknown", "unexplained", "tension", "failure"
    description: str = ""
    domain: str = "general"
    importance: float = 0.5     # 0..1 quao importante
    urgency: float = 0.3        # 0..1 quao urgente
    resolved: bool = False
    resolution_cycle: int = 0
    concepts_involved: List[str] = field(default_factory=list)
    source_module: str = ""     # que modulo gerou ("unknown_space", "tension", etc.)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "type": self.type,
                "description": self.description[:60],
                "importance": round(self.importance, 2),
                "urgency": round(self.urgency, 2),
                "resolved": self.resolved}


class GapGraph:
    """Grafo unificado de lacunas.

    NAO substitui os modulos especializados. Ele e a fonte de dados
    comum que todos eles compartilham.

    Cada Gap e um no. Gaps podem conectar-se entre si (A.gap → B.gap).
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.gaps: Dict[str, Gap] = {}
        self._edges: List[Tuple[str, str]] = []  # (gap_id, related_gap_id)
        self._history: List[Dict] = []

    def add(self, type: str, description: str, domain: str = "general",
            importance: float = 0.5, urgency: float = 0.3,
            concepts: Optional[List[str]] = None,
            source: str = "") -> str:
        """Adiciona uma lacuna (ou atualiza se ja existe)."""
        gid = f"gap_{hashlib.md5((type+description).encode()).hexdigest()[:8]}"
        if gid in self.gaps:
            existing = self.gaps[gid]
            existing.importance = max(existing.importance, importance)
            existing.urgency = max(existing.urgency, urgency)
            return gid
        gap = Gap(id=gid, type=type, description=description,
                  domain=domain, importance=importance, urgency=urgency,
                  concepts_involved=concepts or [], source_module=source)
        self.gaps[gid] = gap
        self._history.append({"action": "add", "type": type,
                              "desc": description[:40], "time": time.time()})
        return gid

    def resolve(self, gap_id: str, cycle: int = 0):
        """Marca uma lacuna como resolvida."""
        gap = self.gaps.get(gap_id)
        if gap and not gap.resolved:
            gap.resolved = True
            gap.resolution_cycle = cycle
            self._history.append({"action": "resolve", "id": gap_id,
                                  "time": time.time()})

    def connect(self, gap_a_id: str, gap_b_id: str):
        """Conecta duas lacunas."""
        if gap_a_id in self.gaps and gap_b_id in self.gaps:
            self._edges.append((gap_a_id, gap_b_id))

    def get_open(self, type: Optional[str] = None, domain: Optional[str] = None,
                 top_k: int = 10) -> List[Gap]:
        """Retorna as lacunas mais importantes nao resolvidas."""
        candidates = [g for g in self.gaps.values() if not g.resolved]
        if type:
            candidates = [g for g in candidates if g.type == type]
        if domain:
            candidates = [g for g in candidates if g.domain == domain]
        # Ordena por importancia * urgencia
        candidates.sort(key=lambda g: -g.importance * (0.5 + g.urgency))
        return candidates[:top_k]

    def total_unresolved(self) -> int:
        return sum(1 for g in self.gaps.values() if not g.resolved)

    def get_state(self) -> Dict:
        return {
            "total": len(self.gaps),
            "open": self.total_unresolved(),
            "by_type": {t: sum(1 for g in self.gaps.values() if g.type == t and not g.resolved)
                       for t in ["unknown", "unexplained", "tension", "failure"]},
            "top": [g.to_dict() for g in self.get_open(top_k=3)],
        }

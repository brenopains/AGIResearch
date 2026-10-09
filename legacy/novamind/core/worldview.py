"""
Worldview — Vieses Epistemologicos Aprendidos Acima de Paradigmas

Enquanto paradigmas dizem "como construir teorias neste dominio",
worldviews dizem "que tipo de explicacao eu prefiro, ponto final".

Exemplos de worldviews:
  "Prefiro causalidade a correlacao"
  "Prefiro continuidade a discretizacao"
  "Prefiro simetria a assimetria"
  "Prefiro agentes a campos"
  "Prefiro estatistica a geometria"

Cada worldview e um conjunto de vieses que influencia TODOS os dominios.
Worldviews competem: a AGI descobre qual funciona melhor globalmente.
"""
import time, hashlib, random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Worldview:
    id: str = ""
    name: str = ""
    description: str = ""

    # Vieses epistemologicos (0..1, quanto prefere esta abordagem)
    bias_causality: float = 0.5     # prefere causalidade a correlacao
    bias_continuity: float = 0.5    # prefere continuo a discreto
    bias_symmetry: float = 0.5      # prefere simetria a assimetria
    bias_agents: float = 0.5        # prefere agentes a campos
    bias_statistics: float = 0.5    # prefere estatistica a geometria
    bias_simplicity: float = 0.7    # prefere simplicidade (navalha)
    bias_emergence: float = 0.5     # prefere emergencia a reducao

    # Metricas
    confidence: float = 0.5
    n_applications: int = 0
    success_rate: float = 0.0
    paradigm_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "name": self.name,
                "biases": {k: round(v, 2) for k, v in self.__dict__.items()
                          if k.startswith("bias_")},
                "confidence": round(self.confidence, 2)}


PRIMORDIAL_WORLDVIEWS = [
    Worldview(name="Reducionista", description="Tudo se explica por partes menores",
              bias_causality=0.8, bias_continuity=0.3, bias_symmetry=0.4,
              bias_agents=0.3, bias_statistics=0.3, bias_simplicity=0.9,
              bias_emergence=0.2, confidence=0.85),
    Worldview(name="Sistêmico", description="Tudo se explica por interacoes",
              bias_causality=0.5, bias_continuity=0.6, bias_symmetry=0.6,
              bias_agents=0.5, bias_statistics=0.5, bias_simplicity=0.5,
              bias_emergence=0.9, confidence=0.75),
    Worldview(name="Matematico", description="Prefiro estruturas formais e simetrias",
              bias_causality=0.4, bias_continuity=0.7, bias_symmetry=0.9,
              bias_agents=0.1, bias_statistics=0.6, bias_simplicity=0.7,
              bias_emergence=0.3, confidence=0.8),
    Worldview(name="Empirista", description="Dados primeiro, teoria depois",
              bias_causality=0.6, bias_continuity=0.4, bias_symmetry=0.3,
              bias_agents=0.4, bias_statistics=0.8, bias_simplicity=0.4,
              bias_emergence=0.5, confidence=0.8),
    Worldview(name="Pragmatico", description="O que funciona, funciona",
              bias_causality=0.5, bias_continuity=0.5, bias_symmetry=0.5,
              bias_agents=0.5, bias_statistics=0.5, bias_simplicity=0.8,
              bias_emergence=0.6, confidence=0.85),
]


class WorldviewLibrary:
    def __init__(self, cge=None):
        self.cge = cge
        self.worldviews: Dict[str, Worldview] = {}
        self.active_worldview_id: Optional[str] = None
        self._history: List[Dict] = []

        for w in PRIMORDIAL_WORLDVIEWS:
            wid = f"wv_{hashlib.md5(w.name.encode()).hexdigest()[:8]}"
            w.id = wid
            self.worldviews[wid] = w

        self.active_worldview_id = list(self.worldviews.keys())[0]

    def select_for_context(self, domain: str = "general") -> Worldview:
        """Seleciona a worldview mais adequada para um dominio."""
        if not self.worldviews:
            return Worldview()
        # Por enquanto, retorna a de maior confianca
        best = max(self.worldviews.values(), key=lambda w: w.confidence * (0.5 + w.success_rate))
        if best.id != self.active_worldview_id:
            self.active_worldview_id = best.id
            self._history.append({"time": time.time(), "worldview": best.name, "domain": domain})
        return best

    def get_active(self) -> Optional[Worldview]:
        return self.worldviews.get(self.active_worldview_id) if self.active_worldview_id else None

    def discover_from_paradigms(self, paradigms: List[Any]) -> List[Worldview]:
        """Descobre novas worldviews observando paradigmas."""
        new = []
        if len(paradigms) < 3:
            return new
        # Extrair vieses compartilhados entre paradigmas
        approach_counts = defaultdict(int)
        for p in paradigms:
            data = p.to_dict() if hasattr(p, 'to_dict') else (p if isinstance(p, dict) else {})
            approach = data.get("approach", "") if isinstance(data, dict) else ""
            if approach:
                approach_counts[approach] += 1
        # Se uma abordagem domina, pode ser uma worldview
        for approach, count in approach_counts.items():
            if count >= 2:
                wid = f"wv_{hashlib.md5(('discovered_'+approach).encode()).hexdigest()[:8]}"
                if wid not in self.worldviews:
                    wv = Worldview(id=wid, name=f"Worldview {approach[:15]}",
                                   description=f"Emergente de {count} paradigmas {approach}",
                                   confidence=min(0.6, count * 0.15), n_applications=count)
                    self.worldviews[wid] = wv
                    new.append(wv)
        return new

    def get_state(self) -> Dict:
        active = self.get_active()
        return {"n_worldviews": len(self.worldviews),
                "active": active.name if active else None,
                "active_biases": {k: round(v, 2) for k, v in (active.__dict__.items() if active else {})
                                 if k.startswith("bias_")},
                "history": self._history[-3:]}

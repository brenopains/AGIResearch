"""
ExplainabilitySpace — O Gap entre "Prever" e "Explicar"

Diferenca do UnknownSpace:
  UnknownSpace: "o que sei que nao sei" (vazios estruturais)
  ExplainabilitySpace: "o que consigo prever mas nao consigo explicar"

Exemplos de entradas:
  - AGI preve que objeto cai quando solto, mas nao explica gravidade
  - AGI preve que 2+2=4, mas nao explica adicao
  - AGI preve que pessoas compram mais no frio, mas nao explica comportamento

Cada entrada e um gap entre:
  - Modelo preditivo (funciona)
  - Modelo explicativo (nao existe ainda)

Esse gap e uma forca motriz para descobrir teorias mais profundas.
"""
import time, uuid, hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ExplainabilityGap:
    """Um gap entre predicao e explicacao."""
    id: str = ""
    phenomenon: str = ""       # o que a AGI consegue prever
    domain: str = "general"
    prediction_confidence: float = 0.0  # quao bem preve
    explanation_exists: bool = False      # tem explicacao?
    explanation_depth: float = 0.0       # 0=nenhuma, 1=completa
    gap_size: float = 0.0                # prediction_confidence - explanation_depth
    created_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "phenomenon": self.phenomenon[:60],
                "domain": self.domain, "gap": round(self.gap_size, 2),
                "prediction": round(self.prediction_confidence, 2)}


class ExplainabilitySpace:
    """Espaco de explicabilidade.

    Monitora o CGE e o WorldModel para identificar gaps entre
    o que a AGI preve e o que ela explica.
    """

    def __init__(self, cge=None, insight_library=None):
        self.cge = cge
        self.insight_library = insight_library
        self.gaps: Dict[str, ExplainabilityGap] = {}
        self._resolved: List[str] = []

    def scan(self):
        """Escaneia o CGE para identificar gaps preve-mas-nao-explica."""
        if self.cge is None:
            return
        # Para cada par de conceitos com aresta, verificar se ha explicacao
        for edge in self.cge.edges[-200:]:
            f, t = edge.get("from", ""), edge.get("to", "")
            if not f or not t:
                continue
            # Tem predicao (aresta existe)
            prediction_conf = edge.get("weight", 0.5)

            # Verificar se ha um no de teoria/explicacao conectado
            has_explanation = False
            explanation_depth = 0.0
            for e2 in self.cge.edges:
                if e2.get("from", "").startswith("_theory") or \
                   e2.get("from", "").startswith("theory"):
                    if e2.get("to", "") in (f, t):
                        has_explanation = True
                        explanation_depth = e2.get("weight", 0.5)
                        break

            gap = prediction_conf - explanation_depth
            if gap > 0.3:  # gap significativo
                key = f"{f}-{t}"
                if key not in self.gaps:
                    self.gaps[key] = ExplainabilityGap(
                        id=f"exp_{hashlib.md5(key.encode()).hexdigest()[:8]}",
                        phenomenon=f"{f} -> {t}",
                        domain=f[:20] if len(f) > 5 else "general",
                        prediction_confidence=prediction_conf,
                        explanation_exists=has_explanation,
                        explanation_depth=explanation_depth,
                        gap_size=gap,
                    )

    def resolve(self, gap_id: str):
        """Marca um gap como resolvido (teoria encontrada)."""
        gap = self.gaps.get(gap_id)
        if gap:
            gap.resolved_at = time.time()
            gap.explanation_exists = True
            gap.explanation_depth = 0.8
            gap.gap_size = 0.0
            self._resolved.append(gap_id)

    def get_open_gaps(self, top_k: int = 5) -> List[ExplainabilityGap]:
        """Retorna os maiores gaps nao resolvidos."""
        open_gaps = [g for g in self.gaps.values() if g.gap_size > 0]
        open_gaps.sort(key=lambda g: -g.gap_size)
        return open_gaps[:top_k]

    def get_state(self) -> Dict:
        return {
            "n_gaps": len(self.gaps),
            "resolved": len(self._resolved),
            "biggest_gap": self.get_open_gaps(1)[0].to_dict() if self.get_open_gaps() else None,
        }

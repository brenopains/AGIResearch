"""
LatentWorkspace — Estado Intermediario Instavel entre Percepcao e Semantica

Antes de um conceito virar HDV consolidado no CGE, ele passa por
um estado LATENTE: nebuloso, instavel, multifacetado.

Diferenca do SemanticWorkspace:
  SemanticWorkspace: conceitos definidos competindo por saliencia
  LatentWorkspace: proto-conceitos, fragmentos, "quase ideias"

Exemplos de estados latentes:
  "█████▒▒▒▒" — quase formando um conceito
  "A que se parece com B mas nao sei o que e B"
  "Sinto que tem algo aqui" (intuicao)

Nao vai para o CGE ate se consolidar.
"""
import time, uuid, math, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LatentFragment:
    """Um fragmento de pensamento ainda nao consolidado."""
    id: str = ""
    description: str = ""        # descricao textual do proto-conceito
    source: str = ""             # "perception", "intuition", "analogy", "tension"
    coherence: float = 0.1       # 0..1, quao coeso (baixo = fragmentado)
    novelty: float = 0.5         # 0..1, quao novo
    emotional_valence: float = 0.0  # -1..1
    hdv_proto: Optional[Any] = None  # proto-HDV (nao consolidado)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "description": self.description[:60],
                "coherence": round(self.coherence, 2),
                "novelty": round(self.novelty, 2)}


class LatentWorkspace:
    """Espaco de trabalho latente.

    Fragmentos entram aqui antes de virarem conceitos no CGE.
    Quando a coerencia ultrapassa um threshold, o fragmento
    pode ser consolidado e movido para o SemanticWorkspace/CGE.
    """

    def __init__(self, capacity: int = 30, consolidation_threshold: float = 0.7):
        self.fragments: Dict[str, LatentFragment] = {}
        self.capacity = capacity
        self.consolidation_threshold = consolidation_threshold
        self._history: List[Dict] = []

    def post(self, description: str, source: str = "perception",
             coherence: float = 0.1, novelty: float = 0.5,
             hdv_proto=None) -> str:
        """Adiciona um fragmento latente."""
        if len(self.fragments) >= self.capacity:
            self._prune()
        fid = f"lat_{uuid.uuid4().hex[:8]}"
        fragment = LatentFragment(
            id=fid, description=description, source=source,
            coherence=coherence, novelty=novelty, hdv_proto=hdv_proto,
        )
        self.fragments[fid] = fragment
        return fid

    def refine(self, fragment_id: str, delta_coherence: float = 0.1):
        """Refina um fragmento (aumenta coerencia)."""
        f = self.fragments.get(fragment_id)
        if f:
            f.coherence = min(1.0, f.coherence + delta_coherence)
            if f.coherence >= self.consolidation_threshold:
                return self._consolidate(fragment_id)
        return None

    def _consolidate(self, fragment_id: str) -> Optional[Dict]:
        """Consolida um fragmento em proto-conceito (pronto para CGE)."""
        f = self.fragments.pop(fragment_id, None)
        if f is None:
            return None
        result = {
            "description": f.description,
            "source": f.source,
            "coherence": f.coherence,
            "novelty": f.novelty,
            "hdv_proto": f.hdv_proto,
        }
        self._history.append({"action": "consolidate", "fragment": f.description[:40],
                              "coherence": f.coherence, "time": time.time()})
        return result

    def get_most_coherent(self, top_k: int = 3) -> List[LatentFragment]:
        """Retorna fragmentos mais proximos da consolidacao."""
        sorted_frags = sorted(self.fragments.values(),
                              key=lambda f: -f.coherence)
        return sorted_frags[:top_k]

    def get_intuitions(self, threshold: float = 0.3) -> List[LatentFragment]:
        """Retorna fragmentos com baixa coerencia mas alta novidade (intuicoes)."""
        return [f for f in self.fragments.values()
                if f.coherence < threshold and f.novelty > 0.5]

    def tick(self, dt: float = 0.2):
        """Decaimento natural. Fragmentos muito incoerentes evaporam."""
        to_remove = []
        for fid, f in self.fragments.items():
            if f.coherence < 0.05:
                to_remove.append(fid)
            # Fragmentos antigos perdem coerencia se nao refinados
            age = time.time() - f.created_at
            if age > 120 and f.coherence < 0.3:
                f.coherence *= (1.0 - 0.01 * dt)
        for fid in to_remove:
            self._history.append({"action": "evaporate", "fragment": self.fragments[fid].description[:40],
                                  "time": time.time()})
            del self.fragments[fid]

    def _prune(self):
        """Remove fragmentos com menor coerencia."""
        sorted_f = sorted(self.fragments.items(), key=lambda x: -x[1].coherence)
        keep = sorted_f[:self.capacity // 2]
        self.fragments = dict(keep)

    def get_state(self) -> Dict:
        return {
            "n_fragments": len(self.fragments),
            "most_coherent": [f.to_dict() for f in self.get_most_coherent(3)],
            "intuitions": len(self.get_intuitions()),
        }

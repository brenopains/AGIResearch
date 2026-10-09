"""
SemanticWorkspace — Memoria de Trabalho Global

Espaco temporario onde hipoteses, conceitos, previsoes, objetivos
e resultados de diferentes subsistemas competem e se combinam 
antes das decisoes do Maestro.

Nao substitui CGE nem World Model. Funciona como espaco de trabalho
dinamico entre eles e o Maestro.
"""
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import deque


@dataclass
class WorkspaceEntry:
    """Uma entrada no espaco de trabalho."""
    content: str  # descricao textual
    source: str  # que subsistema gerou ("perception", "prediction", "hypothesis", "goal", "error", "memory", "simulation", "self")
    weight: float = 0.5  # 0..1 importancia
    confidence: float = 0.5  # 0..1 certeza
    novelty: float = 0.0  # 0..1 quao novo
    urgency: float = 0.0  # 0..1 quao urgente
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def salience(self) -> float:
        """Saliencia = peso * confianca * (1 + urgencia) * (1 + novidade)."""
        return self.weight * (0.5 + self.confidence) * (1 + self.urgency) * (1 + self.novelty * 0.5)

    def decay(self, dt: float = 1.0):
        """Decaimento natural."""
        self.weight *= (1.0 - 0.05 * dt)
        self.urgency *= (1.0 - 0.1 * dt)
        if self.weight < 0.01:
            self.weight = 0.0


class SemanticWorkspace:
    """Espaco de trabalho global.

    O Maestro consulta o workspace a cada ciclo para decidir:
      - O que esta "consciente" agora (entradas com maior saliencia)
      - Qual modo cognitivo ativar baseado no estado do workspace
      - Que informacao passar para a Forja / Self Model / Simulation Engine
    """

    def __init__(self, capacity: int = 50):
        self.entries: Dict[str, WorkspaceEntry] = {}
        self.capacity = capacity
        self.history: List[Dict] = []
        self._global_broadcast: Optional[WorkspaceEntry] = None  # "conteudo consciente" atual

    def post(self, content: str, source: str, weight: float = 0.5,
             confidence: float = 0.5, novelty: float = 0.0,
             urgency: float = 0.0, tags: Optional[List[str]] = None) -> str:
        """Um subsistema publica informacao no workspace.
        
        Se o mesmo conteudo ja existe, apenas atualiza peso/confianca.
        """
        if content in self.entries:
            entry = self.entries[content]
            entry.weight = min(1.0, entry.weight + weight * 0.3)
            entry.confidence = (entry.confidence + confidence) / 2
            entry.novelty = novelty
            entry.urgency = max(entry.urgency, urgency)
            entry.last_accessed = time.time()
            if tags:
                entry.tags = list(set(entry.tags + tags))
            return content

        if len(self.entries) >= self.capacity:
            self._prune()

        entry = WorkspaceEntry(
            content=content, source=source, weight=weight,
            confidence=confidence, novelty=novelty, urgency=urgency,
            tags=tags or []
        )
        self.entries[content] = entry

        self.history.append({
            "time": time.time(),
            "content": content[:60],
            "source": source,
            "salience": entry.salience,
        })
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return content

    def get_conscious(self, n: int = 5) -> List[WorkspaceEntry]:
        """Retorna as N entradas mais salientes (conteudo 'consciente')."""
        active = [e for e in self.entries.values() if e.weight > 0.05]
        active.sort(key=lambda e: -e.salience)
        return active[:n]

    def get_by_source(self, source: str) -> List[WorkspaceEntry]:
        """Retorna entradas de uma fonte especifica."""
        return [e for e in self.entries.values() if e.source == source]

    def broadcast(self, content: str, source: str = "maestro"):
        """O Maestro define o 'conteudo consciente' atual."""
        entry = self.post(content, source, weight=1.0, urgency=0.8)
        self._global_broadcast = self.entries.get(content)
        return entry

    def tick(self, dt: float = 0.2):
        """Decaimento de todas as entradas."""
        to_remove = []
        for key, entry in self.entries.items():
            entry.decay(dt)
            if entry.weight <= 0:
                to_remove.append(key)
        for key in to_remove:
            del self.entries[key]

    def _prune(self):
        """Remove entradas com menor saliencia."""
        sorted_e = sorted(self.entries.items(), key=lambda x: -x[1].salience)
        keep = sorted_e[:self.capacity // 2]
        self.entries = dict(keep)

    def get_state(self) -> Dict:
        conscious = self.get_conscious(3)
        return {
            "n_entries": len(self.entries),
            "conscious": [{"content": e.content[:40], "source": e.source,
                           "salience": round(e.salience, 2)} for e in conscious],
            "sources": list(set(e.source for e in self.entries.values())),
        }

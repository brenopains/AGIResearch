"""
MacroRegistry — Registry separado para macros (conhecimento compilado).

Diferenca do PrimitiveRegistry:
  - Macros sao sequencias comprimidas, nao operacoes fundamentais
  - Macros tem metadata de origem (quais tasks, quais primitives)
  - Macros sao priorizadas por generalidade (funcionam em muitas tasks)
  - Macros NAO entram no beam search como primitives iguais
  - O planner tenta macros primeiro, depois fallback para primitives

Isso evita que o espaco de busca seja enviesado por macros muito especificas.
"""
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class Macro:
    """Uma macro — sequencia comprimida, nao operacao fundamental."""
    name: str
    primitive_sequence: Tuple[str, ...]
    description: str = ""
    frequency: int = 0
    n_tasks: int = 0          # em quantas tasks DIFERENTES apareceu
    success_rate: float = 0.0
    confidence: float = 0.0
    compression_ratio: float = 1.0  # len(sequence) / 1
    domains: Set[str] = field(default_factory=set)
    discovered_at: float = field(default_factory=time.time)
    source_tasks: Set[str] = field(default_factory=set)

    @property
    def generality(self) -> float:
        """Quao geral e a macro (0-1). 1 = funciona em todas as tasks."""
        return min(1.0, self.n_tasks / max(1, self.frequency)) if self.frequency > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "macro": self.name,
            "sequence": " -> ".join(self.primitive_sequence),
            "frequency": self.frequency,
            "n_tasks": self.n_tasks,
            "success_rate": round(self.success_rate, 3),
            "confidence": round(self.confidence, 3),
            "generality": round(self.generality, 3),
            "compression": round(self.compression_ratio, 1),
        }


class MacroRegistry:
    """Registry separado para macros.

    Uso:
        macros = MacroRegistry()
        macros.register("flip_move", ("flip_horizontal", "move"),
                        frequency=83, n_tasks=12)
        best = macros.get_best_for_context("ARC")
    """

    def __init__(self):
        self.macros: Dict[str, Macro] = {}

    def register(self, name: str, sequence: Tuple[str, ...],
                 frequency: int = 1, n_tasks: int = 1,
                 success_rate: float = 1.0, domain: str = "",
                 source_task: str = ""):
        """Registra ou atualiza uma macro."""
        if name in self.macros:
            m = self.macros[name]
            m.frequency += frequency
            if source_task:
                m.source_tasks.add(source_task)
            m.n_tasks = len(m.source_tasks)
            m.success_rate = (m.success_rate + success_rate) / 2
            m.confidence = min(1.0, m.frequency * 0.02)
            if domain:
                m.domains.add(domain)
        else:
            self.macros[name] = Macro(
                name=name,
                primitive_sequence=sequence,
                description=f"macro: {' -> '.join(sequence)}",
                frequency=frequency,
                n_tasks=n_tasks,
                success_rate=success_rate,
                confidence=min(1.0, frequency * 0.02),
                compression_ratio=len(sequence),
                domains={domain} if domain else set(),
                source_tasks={source_task} if source_task else set(),
            )

    def get_best(self, n: int = 10, min_confidence: float = 0.0,
                 domain: str = "") -> List[Macro]:
        """Retorna as melhores macros, ordenadas por prioridade.

        Prioridade = confidence * generality.
        """
        candidates = [m for m in self.macros.values()
                     if m.confidence >= min_confidence]
        if domain:
            candidates = [m for m in candidates if domain in m.domains]
        candidates.sort(key=lambda m: -m.confidence * m.generality)
        return candidates[:n]

    def get_for_sequence(self, seq: Tuple[str, ...]) -> Optional[Macro]:
        """Retorna macro para uma sequencia, se existir."""
        for m in self.macros.values():
            if m.primitive_sequence == seq:
                return m
        return None

    def get_stats(self) -> dict:
        if not self.macros:
            return {"total": 0}
        confs = [m.confidence for m in self.macros.values()]
        gens = [m.generality for m in self.macros.values()]
        ratios = [m.compression_ratio for m in self.macros.values()]
        return {
            "total": len(self.macros),
            "avg_confidence": round(np.mean(confs), 3),
            "avg_generality": round(np.mean(gens), 3),
            "avg_compression": round(np.mean(ratios), 1),
            "total_frequency": sum(m.frequency for m in self.macros.values()),
            "total_tasks": len(set.union(*[m.source_tasks for m in self.macros.values()])) if self.macros else 0,
        }

    def clear(self):
        self.macros.clear()

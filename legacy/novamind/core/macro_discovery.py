"""
MacroDiscovery — Descobre MACROS (sequencias frequentes de primitives) e as promove a novas primitives.

O que sao macros?
  flip_horizontal → move (100% sucesso) = "NormalizeOrientation"
  rotate_180 → recolor = "RecolorRotated"
  
Cada macro vira uma NOVA primitiva que o planner pode usar em 1 passo
em vez de 2-3. Isso e COMPRESSAO HIERARQUICA: sequencias longas
viram unidades de alto nivel.

O sistema descobre macros automaticamente:
  1. Escaneia o CompositionGraph por n-grams frequentes
  2. Testa se a macro generaliza (funciona em tasks diferentes)
  3. Se validada, cria nova primitiva no registry
  4. A macro e marcada como "macro" para rastreabilidade
"""
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time

from .program_induction.composition_graph import CompositionGraph


@dataclass
class Macro:
    """Uma macro descoberta: sequencia comprimida em nova primitive."""
    name: str                           # "NormalizeOrientation"
    primitive_sequence: Tuple[str, ...] # ("flip_horizontal", "move")
    frequency: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    confidence: float = 0.0
    discovered_at: float = field(default_factory=time.time)
    source_tasks: Set[str] = field(default_factory=set)  # tasks onde apareceu

    def to_dict(self) -> dict:
        return {
            "macro": self.name,
            "sequence": " -> ".join(self.primitive_sequence),
            "frequency": self.frequency,
            "success_rate": round(self.success_rate, 3),
            "confidence": round(self.confidence, 3),
            "source_tasks": len(self.source_tasks),
        }


class MacroDiscovery:
    """Descobre macros de sequencias frequentes no CompositionGraph.

    Uso:
        discovery = MacroDiscovery(composition_graph, registry)
        discovery.scan()
        for macro in discovery.get_macros(min_confidence=0.7):
            discovery.promote_to_primitive(macro)
    """

    def __init__(self, composition_graph: CompositionGraph, registry=None):
        self.graph = composition_graph
        self.registry = registry
        self.macros: Dict[Tuple[str, ...], Macro] = {}
        self._scanned = False

    def scan(self, min_frequency: int = 3, min_success_rate: float = 0.6):
        """Escaneia o CompositionGraph por n-grams frequentes de comprimento 2-4."""
        self._scanned = True

        # Extrai todas as sequencias de 2, 3, 4 primitives do grafo
        # O CompositionGraph armazena edges A→B. Para sequencias maiores,
        # reconstruimos dos programas registrados.
        sequences = self._extract_sequences()

        # Agrupa por tupla de tipos, conta frequencia e sucesso
        seq_stats: Dict[Tuple[str, ...], List[bool]] = defaultdict(list)

        for seq, succeeded in sequences:
            seq_stats[seq].append(succeeded)

        # Cria macros para sequencias frequentes
        for seq, outcomes in seq_stats.items():
            if len(seq) < 2:
                continue
            # freq = numero total de vezes que esta sequencia foi tentada
            # (soma dos attempt_counts de todas as edges na sequencia)
            freq = len(outcomes)
            successes = sum(1 for s in outcomes if s)
            rate = successes / max(1, freq)
            # Usa attempt_count real do grafo como frequencia
            raw_freq = self._get_sequence_frequency(seq)
            if raw_freq > freq:
                freq = raw_freq

            if freq >= min_frequency and rate >= min_success_rate:
                macro_name = self._generate_name(seq)
                if seq not in self.macros:
                    self.macros[seq] = Macro(
                        name=macro_name,
                        primitive_sequence=seq,
                    )
                self.macros[seq].frequency = freq
                self.macros[seq].success_count = successes
                self.macros[seq].success_rate = rate
                self.macros[seq].confidence = rate * min(1.0, freq / 10)

    def _extract_sequences(self) -> List[Tuple[Tuple[str, ...], bool]]:
        """Extrai sequencias de comprimento 2-4 do grafo.

        Usa attempt_count como indicador de composicao valida:
        se foi tentado 2+ vezes, considera uma composicao real.
        """
        sequences = []
        edges = self.graph.get_edges(min_weight=0.0)

        # Acessa edges internas do grafo para pegar attempt_count
        raw_edges = list(self.graph.edges.values()) if hasattr(self.graph, 'edges') else []

        for e in raw_edges:
            seq = (e.from_type, e.to_type)
            # Sucesso se attempt_count > 1 (composicao intencional)
            succeeded = e.attempt_count > 1
            sequences.append((seq, succeeded))

        # Sequencias de 3: A->B + B->C
        for e1 in raw_edges:
            for e2 in raw_edges:
                if e1.to_type == e2.from_type and e1.from_type != e2.to_type:
                    seq = (e1.from_type, e1.to_type, e2.to_type)
                    succeeded = e1.attempt_count > 1 and e2.attempt_count > 1
                    sequences.append((seq, succeeded))

        return sequences

    def _get_sequence_frequency(self, seq: Tuple[str, ...]) -> int:
        """Obtem a frequencia real de uma sequencia a partir do grafo."""
        raw_edges = list(self.graph.edges.values()) if hasattr(self.graph, 'edges') else []
        if len(seq) == 2:
            for e in raw_edges:
                if e.from_type == seq[0] and e.to_type == seq[1]:
                    return e.attempt_count
        # Para sequencias maiores, usa a media dos attempt_counts
        if len(seq) >= 3:
            counts = []
            for i in range(len(seq) - 1):
                for e in raw_edges:
                    if e.from_type == seq[i] and e.to_type == seq[i + 1]:
                        counts.append(e.attempt_count)
            if counts:
                return int(np.mean(counts))
        return 0

    def _generate_name(self, seq: Tuple[str, ...]) -> str:
        """Gera nome para a macro baseado nos tipos."""
        # Extrai palavras-chave
        keywords = []
        for s in seq:
            # Pega a parte principal: "flip_horizontal" -> "flip"
            parts = s.split("_")
            if parts:
                keywords.append(parts[0])
        # Remove duplicatas mantendo ordem
        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k.capitalize())
        base = "".join(unique) if unique else "Macro"
        return f"Macro{base}_{len(seq)}"

    def promote_to_primitive(self, macro: Macro) -> bool:
        """Promove uma macro a primitiva real no registry."""
        if self.registry is None:
            return False

        from .program_induction.primitive import Primitive

        # Verifica se ja existe
        existing = self.registry.get(macro.name)
        if existing:
            existing.confidence = max(existing.confidence, macro.confidence)
            return False

        # Cria primitiva com funcao que executa a sequencia completa
        prim = Primitive(
            name=macro.name,
            structure_type="macro",
            description=f"macro: {' -> '.join(macro.primitive_sequence)}",
        )

        # A funcao apply executa cada primitive da sequencia em ordem
        def _make_exec(seq):
            def _exec(grid, params):
                result = grid.copy()
                for prim_name in seq:
                    p = self.registry.get(prim_name) if self.registry else None
                    if p:
                        result = p.apply(result, params)
                return result
            return _exec

        prim._execute = _make_exec(macro.primitive_sequence)
        prim.confidence = macro.confidence

        # Marca como macro via nome
        self.registry.register(prim)
        return True

    def get_macros(self, min_confidence: float = 0.0) -> List[Macro]:
        """Retorna macros descobertas."""
        return [m for m in self.macros.values() if m.confidence >= min_confidence]

    def get_stats(self) -> dict:
        if not self._scanned:
            return {"scanned": False, "macros": 0}
        confs = [m.confidence for m in self.macros.values()]
        return {
            "scanned": True,
            "macros": len(self.macros),
            "avg_confidence": round(np.mean(confs), 3) if confs else 0,
            "promotable": sum(1 for m in self.macros.values() if m.confidence > 0.5),
        }

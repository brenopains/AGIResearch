"""
GapDetector — Detecta quando um invariante NAO tem primitiva correspondente.

Quando o CFE identifica um invariante (ex: "reflection"), mas o registry
nao tem nenhuma primitiva que produza consistentemente esse invariante,
o GapDetector registra uma "lacuna" que o PrimitiveInventionEngine
deve preencher.
"""
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class PrimitiveGap:
    """Uma lacuna: invariante detectado mas sem primitiva que o produza."""
    invariant_type: str                 # "reflection", "expansion", "branching", etc
    domain: str = "arc"
    n_observations: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    priority: float = 0.3               # 0-1, aumenta com frequencia
    examples: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "invariant": self.invariant_type,
            "domain": self.domain,
            "observations": self.n_observations,
            "priority": round(self.priority, 2),
            "examples": len(self.examples),
        }


class GapDetector:
    """Detecta lacunas entre invariantes e primitives disponiveis.

    Uso:
        detector = GapDetector(registry)
        gaps = detector.find_gaps(invariants, input_grid, output_grid)
        for gap in gaps:
            print(f"Missing primitive for: {gap.invariant_type}")
    """

    def __init__(self, registry=None):
        self.registry = registry
        self.gaps: Dict[str, PrimitiveGap] = {}

    def find_gaps(self, invariants: list, input_grid: np.ndarray,
                  output_grid: np.ndarray) -> List[PrimitiveGap]:
        """Encontra lacunas entre invariantes e primitives.

        Para cada invariante, verifica se alguma primitiva no registry
        consegue produzir consistentemente a transformacao observada.

        Returns: lista de lacunas (priorizadas por frequencia).
        """
        found_gaps = []

        for inv in invariants:
            inv_type = inv.structure_type

            # Verifica se alguma primitiva ja cobre este invariante
            has_primitive = self._invariant_covered(inv_type, input_grid, output_grid)

            if not has_primitive:
                # Lacuna detectada
                gap = self._register_gap(inv_type, inv.domain if hasattr(inv, 'domain') else "general",
                                          input_grid, output_grid)
                found_gaps.append(gap)

        # Ordena por prioridade
        found_gaps.sort(key=lambda g: -g.priority)
        return found_gaps

    def _invariant_covered(self, inv_type: str, inp: np.ndarray,
                           out: np.ndarray) -> bool:
        """Verifica se alguma primitiva no registry cobre este invariante."""
        if self.registry is None:
            return False

        for prim in self.registry.get_all():
            # Tenta aplicar a primitiva e ver se produz o output
            if hasattr(prim, '_execute') and prim._execute is not None:
                try:
                    result = prim._execute(inp, {})
                    if result.shape == out.shape and np.array_equal(result, out):
                        return True
                except Exception:
                    pass

            # Tenta generate_hypothesis
            if hasattr(prim, 'generate_hypothesis'):
                try:
                    inst = prim.generate_hypothesis(inp, out)
                    if inst is not None:
                        result = inst.apply(inp)
                        if result.shape == out.shape and np.array_equal(result, out):
                            return True
                except Exception:
                    pass

        return False

    def _register_gap(self, inv_type: str, domain: str,
                       inp: np.ndarray, out: np.ndarray) -> PrimitiveGap:
        """Registra ou atualiza uma lacuna."""
        if inv_type in self.gaps:
            gap = self.gaps[inv_type]
            gap.n_observations += 1
            gap.last_seen = time.time()
            gap.priority = min(1.0, gap.n_observations * 0.05)
            if len(gap.examples) < 5:
                gap.examples.append((inp.copy(), out.copy()))
        else:
            gap = PrimitiveGap(
                invariant_type=inv_type,
                domain=domain,
                n_observations=1,
                examples=[(inp.copy(), out.copy())],
            )
            self.gaps[inv_type] = gap

        return gap

    def get_gaps(self, min_priority: float = 0.0) -> List[PrimitiveGap]:
        return [g for g in self.gaps.values() if g.priority >= min_priority]

    def get_stats(self) -> dict:
        return {
            "total_gaps": len(self.gaps),
            "gaps": [g.to_dict() for g in sorted(self.gaps.values(), key=lambda x: -x.priority)],
        }

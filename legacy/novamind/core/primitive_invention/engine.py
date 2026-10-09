"""
PrimitiveInventionEngine — Engine completa que INVENTA novas primitives.

Integra:
  1. GapDetector: encontra lacunas entre invariantes e primitives
  2. ProgramHypothesizer: gera programas candidatos
  3. PrimitiveVerifier: valida os candidatos
  4. PrimitiveRegistrar: salva os validados

Ciclo:
  Observacao -> invariante -> gap -> hipotese -> validacao -> registro -> nova primitiva
"""
from typing import List, Optional, Tuple
import numpy as np

from .gap_detector import GapDetector, PrimitiveGap
from .hypothesizer import ProgramHypothesizer, CandidateProgram
from .verifier import PrimitiveVerifier
from .registrar import PrimitiveRegistrar


class PrimitiveInventionEngine:
    """Engine que inventa novas primitives.

    Uso:
        engine = PrimitiveInventionEngine(registry)
        engine.observe(inp, out, invariants)
        new_prims = engine.invent()  # retorna as novas primitives criadas
    """

    def __init__(self, registry=None, min_gap_priority: float = 0.2,
                 use_atomic: bool = False):
        self.registry = registry
        self.use_atomic = use_atomic
        self.gap_detector = GapDetector(registry=registry)
        self.hypothesizer = ProgramHypothesizer(max_depth=3, use_atomic=use_atomic)
        self.verifier = PrimitiveVerifier(min_examples=2, use_atomic=use_atomic)
        self.registrar = PrimitiveRegistrar(registry=registry, use_atomic=use_atomic)
        self.min_gap_priority = min_gap_priority
        self.invention_history: List[dict] = []

    def observe(self, input_grid: np.ndarray, output_grid: np.ndarray,
                invariants: list):
        """Observa uma transformacao e detecta gaps."""
        self.gap_detector.find_gaps(invariants, input_grid, output_grid)

    def invent(self, domain: str = "arc") -> List[str]:
        """Tenta inventar novas primitives para preencher gaps.

        Returns: lista de nomes das novas primitives criadas.
        """
        gaps = self.gap_detector.get_gaps(min_priority=self.min_gap_priority)
        created = []

        for gap in gaps:
            if len(gap.examples) < 3:
                continue

            # Gera candidatos
            candidates = self.hypothesizer.hypothesize(gap.examples)
            best = candidates[0] if candidates else None

            if best is None or best.score < 0.5:
                continue

            # Separa treino/teste (prioriza treino)
            n = len(gap.examples)
            n_train = max(2, n - 2)
            train = gap.examples[:n_train]
            test = gap.examples[n_train:] if n_train < n else None

            # Verifica
            if not self.verifier.verify(best, train, test):
                continue

            # Nome descritivo
            inv_type = gap.invariant_type[:20].replace(" ", "_")
            op_names = "_".join(best.op_sequence[:3])
            prim_name = f"invented_{inv_type}_{op_names}"

            # Registra
            success = self.registrar.register(best, name=prim_name)

            if success:
                self.invention_history.append({
                    "name": prim_name,
                    "gap": gap.invariant_type,
                    "program": best.name,
                    "score": best.score,
                    "domain": domain,
                })
                created.append(prim_name)

        return created

    def get_inventions(self) -> List[dict]:
        return self.invention_history

    def get_gaps(self) -> List[PrimitiveGap]:
        return self.gap_detector.get_gaps()

    def get_stats(self) -> dict:
        return {
            "total_gaps": len(self.gap_detector.gaps),
            "total_inventions": len(self.invention_history),
            "inventions": self.invention_history[-5:],
            "registrar": self.registrar.get_stats(),
        }

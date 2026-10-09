"""
CFEBridgeSearch — Integra CFE (conceitos) ao ProgramSynthesizer com soft guidance.

Em vez de filtro duro (só usa primitivas X), aplica PESOS:
  - conceitos do CFE viram boost_factor nas primitivas preferidas
  - se conceito-guided search achar solucao exata: retorna com atribuicao
  - se nao achar: fallback para busca completa
  - stats comparam: conceito ajudou ou atrapalhou?
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
import time

from core.conceptual_discovery.cfe_bridge import CFEBridge
from core.program_induction.synthesis import ProgramSynthesizer, Program


# Boost padrao quando um conceito e detectado
CONCEPT_BOOST = 2.5
FALLBACK_BOOST = 1.0


class CFEBridgeSearch:
    """Integra CFE ao ProgramSynthesizer com soft guidance.

    Uso:
        bridge_search = CFEBridgeSearch(cfe_bridge, synthesizer)
        result = bridge_search.search(inp, out)
        print(bridge_search.get_stats())
    """

    def __init__(self, cfe_bridge: CFEBridge,
                 synthesizer: ProgramSynthesizer,
                 concept_boost: float = CONCEPT_BOOST):
        self.cfe = cfe_bridge
        self.synthesizer = synthesizer
        self.concept_boost = concept_boost

        # Stats
        self.total_searches: int = 0
        self.concept_guided: int = 0
        self.exact_with_concept: int = 0
        self.exact_without_concept: int = 0
        self.fallback_needed: int = 0
        self.concept_miss: int = 0
        self.search_histories: List[dict] = []

    def search(self, input_grid: np.ndarray,
               output_grid: np.ndarray) -> List[Program]:
        """Busca com soft guidance do CFE.

        Returns: programas ordenados por score.
        """
        self.total_searches += 1
        start = time.time()

        # 1. CFE detecta conceito
        concept = self.cfe.get_concept_hint(input_grid, output_grid)
        filter_set = self.cfe.get_primitive_filter(input_grid, output_grid)

        if concept and filter_set:
            self.concept_guided += 1
            concept_detected = True

            # Cria pesos: boost para primitivas do conceito
            weights = {p: self.concept_boost for p in filter_set}

            # 2. Tenta com guidance
            results = self.synthesizer.synthesize(
                input_grid, output_grid,
                primitive_weights=weights,
            )

            # 3. Se achou solucao exata → retorna
            if results and results[0].validated and results[0].score >= 1.0:
                self.exact_with_concept += 1
                elapsed = time.time() - start
                self.search_histories.append({
                    "concept": concept,
                    "n_guided": len(filter_set),
                    "exact": True,
                    "fallback": False,
                    "elapsed": round(elapsed, 3),
                })
                return results

            # 4. Fallback: busca completa sem pesos
            self.fallback_needed += 1
            fallback_results = self.synthesizer.synthesize(
                input_grid, output_grid,
            )

            # Compara: conceito ajudou ou atrapalhou?
            best_guided = results[0].score if results else 0.0
            best_fallback = fallback_results[0].score if fallback_results else 0.0

            elapsed = time.time() - start
            self.search_histories.append({
                "concept": concept,
                "n_guided": len(filter_set),
                "exact": False,
                "fallback": True,
                "guided_score": round(best_guided, 3),
                "fallback_score": round(best_fallback, 3),
                "concept_helped": best_guided > best_fallback,
                "elapsed": round(elapsed, 3),
            })

            if not (fallback_results and fallback_results[0].validated):
                self.concept_miss += 1

            return fallback_results

        # 5. Sem conceito: busca normal
        concept_detected = False
        results = self.synthesizer.synthesize(input_grid, output_grid)

        if results and results[0].validated and results[0].score >= 1.0:
            self.exact_without_concept += 1

        elapsed = time.time() - start
        self.search_histories.append({
            "concept": None,
            "n_guided": 0,
            "exact": bool(results and results[0].validated),
            "fallback": False,
            "elapsed": round(elapsed, 3),
        })

        return results

    def get_concept_help_rate(self) -> float:
        """Taxa de acerto do CFE: quantas vezes o conceito ajudou vs atrapalhou."""
        if self.concept_guided == 0:
            return 0.0
        helped = sum(1 for h in self.search_histories
                     if h.get("concept_helped"))
        return helped / max(1, self.concept_guided)

    def get_stats(self) -> dict:
        return {
            "total_searches": self.total_searches,
            "concept_guided": self.concept_guided,
            "exact_with_concept": self.exact_with_concept,
            "exact_without_concept": self.exact_without_concept,
            "fallback_needed": self.fallback_needed,
            "concept_miss": self.concept_miss,
            "concept_help_rate": round(self.get_concept_help_rate(), 3),
            "concept_success_rate": round(
                self.exact_with_concept / max(1, self.concept_guided), 3
            ) if self.concept_guided else 0.0,
        }

    def get_recent_searches(self, n: int = 5) -> List[dict]:
        return self.search_histories[-n:]

    def search_with_attribution(self, input_grid: np.ndarray,
                                 output_grid: np.ndarray) -> dict:
        """Busca + atribuicao: qual conceito guiou, funcionou, etc."""
        programs = self.search(input_grid, output_grid)
        last = self.search_histories[-1] if self.search_histories else {}

        return {
            "programs": programs,
            "concept": last.get("concept"),
            "exact": bool(programs and programs[0].validated),
            "concept_helped": last.get("concept_helped", False),
            "fallback_used": last.get("fallback", False),
            "guided_score": last.get("guided_score", 0.0),
            "fallback_score": last.get("fallback_score", 0.0),
            "n_programs": len(programs),
            "best_score": round(programs[0].score, 3) if programs else 0.0,
        }

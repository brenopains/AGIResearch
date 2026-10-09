"""
LawGuidedSynthesizer — Ponte entre WorldLaws e ProgramSynthesizer.

FECHA O CICLO:
  Leis → guiam busca → observacao → confirmacao/refutacao → refinamento

Quando o synthesizer vai buscar uma solucao:
  1. Consulta quais leis se aplicam ao dominio
  2. Leis fornecem primitive_weights (boost para primitivas alinhadas)
  3. Apos busca, verifica se resultado confirma ou contradiz leis
  4. Se contradiz → alimenta ConflictResolver
"""
from typing import Optional, List, Dict
import numpy as np

from core.program_induction.synthesis import ProgramSynthesizer, Program


class LawGuidedSynthesizer:
    """Wraps ProgramSynthesizer com guidance de leis do mundo.

    Uso:
        law_syn = LawGuidedSynthesizer(synthesizer, law_registry, conflict_resolver)
        programs = law_syn.synthesize(inp, out, domain="geometry")
    """

    def __init__(self, synthesizer: ProgramSynthesizer,
                 law_registry=None,
                 conflict_resolver=None,
                 world_model=None):
        self.synthesizer = synthesizer
        self.law_registry = law_registry
        self.conflict_resolver = conflict_resolver
        self.world_model = world_model

        # Stats
        self.total_searches: int = 0
        self.law_confirmed: int = 0
        self.law_violated: int = 0

    def synthesize(self, input_grid: np.ndarray,
                   output_grid: np.ndarray,
                   domain: str = "general",
                   primitive_weights: Optional[Dict[str, float]] = None) -> List[Program]:
        """Sintetiza programas com guidance das leis do dominio.

        1. Consulta leis do dominio
        2. Leis sugerem primitive_weights
        3. Roda synthesizer com esses pesos
        4. Verifica se resultado confirma ou viola leis
        5. Se violou → alimenta ConflictResolver
        """
        self.total_searches += 1

        # 1. Consulta leis
        law_weights = {}
        if self.law_registry:
            domain_laws = self.law_registry.get_laws_for_domain(domain)
            for law in domain_laws:
                # Cada lei sugere boost de 2.0 para conceitos que explica
                for concept in law.explains:
                    law_weights[concept] = max(law_weights.get(concept, 1.0), 2.0)

        # Merge com weights externos (se houver)
        if primitive_weights:
            for k, v in primitive_weights.items():
                law_weights[k] = max(law_weights.get(k, 1.0), v)

        # 2. Roda synthesizer
        if law_weights:
            results = self.synthesizer.synthesize(
                input_grid, output_grid,
                primitive_weights=law_weights,
            )
        else:
            results = self.synthesizer.synthesize(input_grid, output_grid)

        # 3. Verifica leis (se temos resultado e world model)
        if results and results[0].validated and self.world_model and self.law_registry:
            self._check_laws(results[0], input_grid, output_grid, domain)

        return results

    def _check_laws(self, program: Program,
                    inp: np.ndarray, out: np.ndarray,
                    domain: str):
        """Verifica se o programa confirma ou viola leis do dominio."""
        if not self.law_registry:
            return

        domain_laws = self.law_registry.get_laws_for_domain(domain)
        for law in domain_laws:
            # Executa programa e ve se resultado faz sentido para a lei
            try:
                result = program.execute(inp)
                if result.shape != out.shape:
                    continue

                # Metricas simples por tipo de lei
                if law.name == "conservation":
                    preserved = np.array_equal(result, inp) if out.shape == inp.shape else False
                    if preserved:
                        self.law_confirmed += 1
                    else:
                        self.law_violated += 1
                        if self.conflict_resolver:
                            self.conflict_resolver.detect_conflict(
                                law_name=law.name,
                                domain=domain,
                                observation={"output": "changed", "expected": "conserved"},
                                prediction={"conserved": True},
                                description=f"conservation violated in {domain}",
                            )
                elif law.name == "propagation":
                    n_out = int((out > 0).sum()) if hasattr(out, 'sum') else 0
                    n_inp = int((inp > 0).sum()) if hasattr(inp, 'sum') else 0
                    if n_out > n_inp:
                        self.law_confirmed += 1
                    else:
                        self.law_violated += 1
            except:
                pass

    def get_stats(self) -> dict:
        return {
            "total_searches": self.total_searches,
            "law_confirmed": self.law_confirmed,
            "law_violated": self.law_violated,
            "law_consistency": round(
                self.law_confirmed / max(1, self.law_confirmed + self.law_violated), 3
            ) if (self.law_confirmed + self.law_violated) > 0 else 0,
        }

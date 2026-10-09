"""
AbstractProgramSynthesis — Sintese de cadeias de raciocinio abstrato.

Assim como o ProgramGraphSearch compoe primitives visuais
(recolor→move) para transformar grids, este modulo compoe
primitives abstratas (infer→analogize→generalize) para
TRANSFORMAR CONHECIMENTO NO CGE.

O fluxo:
  1. CognitiveDeltaExtractor detecta o que mudou no CGE
  2. Abstract solvers inferem quais operacoes causaram a mudanca
  3. O synthesis compoe operacoes em cadeias de raciocinio
  4. A cadeia e registrada como um CognitiveProgram no CGE
"""
from typing import List, Optional, Dict
import time

from .cognitive_delta import CognitiveDeltaExtractor, CognitiveDelta, CGEState
from .abstract_primitives import (
    get_abstract_solver, ABSTRACT_SCHEMAS,
    _solve_infer, _solve_analogize, _solve_generalize,
    _solve_contradict, _solve_specialize,
)


class AbstractReasoningChain:
    """Uma cadeia de raciocinio abstrato composta.

    Equivalente ao Program para o dominio cognitivo.
    """

    def __init__(self, name: str = "", steps: List[dict] = None,
                 score: float = 0.0, validated: bool = False):
        self.name = name
        self.steps = steps or []
        self.score = score
        self.validated = validated
        self.created_at = time.time()

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_steps": self.n_steps,
            "score": round(self.score, 3),
            "validated": self.validated,
            "steps": self.steps[:5],
        }


class AbstractProgramSynthesizer:
    """Sintetizador de cadeias de raciocinio abstrato.

    Observa deltas no CGE e infere as operacoes cognitivas
    que transformaram um estado em outro.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.delta_extractor = CognitiveDeltaExtractor()
        self._history: List[AbstractReasoningChain] = []

    def observe(self, before: CGEState, after: CGEState) -> Optional[AbstractReasoningChain]:
        """Observa uma transicao de estado cognitivo e tenta sintetizar a cadeia.

        Args:
            before: estado CGE antes
            after: estado CGE depois

        Returns: AbstractReasoningChain se conseguiu sintetizar, None caso contrario.
        """
        if self.cge is None:
            return None

        delta = self.delta_extractor.extract(before, after)
        if not delta.has_changes:
            return None

        # Tenta cada solver para o tipo de delta
        solver = get_abstract_solver(delta.transformation_type)
        steps = []

        if solver:
            params = solver(delta, self.cge)
            if params:
                schema = ABSTRACT_SCHEMAS.get(delta.transformation_type, {})
                step_name = delta.transformation_type
                steps.append({
                    "operation": step_name,
                    "params": params,
                    "description": schema.get("description", step_name),
                })

        # Tenta solvers adicionais
        for extra_solver in [_solve_analogize, _solve_contradict, _solve_specialize]:
            params = extra_solver(delta, self.cge)
            if params:
                op_name = extra_solver.__name__.replace("_solve_", "")
                steps.append({
                    "operation": op_name,
                    "params": params,
                    "description": ABSTRACT_SCHEMAS.get(op_name, {}).get("description", op_name),
                })

        if not steps:
            return None

        # Constroi nome da cadeia
        chain_name = " -> ".join(s["operation"] for s in steps)

        chain = AbstractReasoningChain(
            name=chain_name,
            steps=steps,
            score=min(1.0, len(steps) * 0.3),
            validated=delta.transformation_type != "unknown",
        )
        self._history.append(chain)
        return chain

    def get_history(self, n_last: int = 20) -> List[AbstractReasoningChain]:
        return self._history[-n_last:]

    def get_stats(self) -> dict:
        if not self._history:
            return {"total": 0, "unique_chains": 0}
        chains = set(c.name for c in self._history)
        validated = sum(1 for c in self._history if c.validated)
        return {
            "total_observations": len(self._history),
            "unique_chains": len(chains),
            "validated": validated,
            "avg_steps": sum(c.n_steps for c in self._history) / max(1, len(self._history)),
        }

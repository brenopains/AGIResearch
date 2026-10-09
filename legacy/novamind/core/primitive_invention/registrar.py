"""
PrimitiveRegistrar — Salva primitiva validada no registry.

Converte um CandidateProgram em uma Primitive executavel,
com nome descritivo, executor, solver_fn, e confianca.
"""
from typing import Optional
from .hypothesizer import CandidateProgram
from .low_level_ops import get_ops
from .atomic_ops import get_atomic_programs


class PrimitiveRegistrar:
    """Registra primitivas inventadas no registry.

    Uso:
        registrar = PrimitiveRegistrar(registry)
        registrar.register(candidate, "rotate_90_invented")
    """

    def __init__(self, registry=None, use_atomic: bool = False):
        self.registry = registry
        self.use_atomic = use_atomic
        self._counter = 0

    def register(self, candidate: CandidateProgram,
                 name: Optional[str] = None) -> bool:
        """Registra um candidato validado como primitiva."""
        if self.registry is None:
            return False

        if name is None:
            self._counter += 1
            name = f"invented_{self._counter}"

        if self.registry.get(name):
            return False

        from ..program_induction.primitive import Primitive

        prim = Primitive(
            name=name,
            structure_type="invented",
            description=f"invented primitive: {' -> '.join(candidate.op_sequence)}",
            solver_fn=lambda i, o: {},
        )

        # Construtor de executor: suporta atomic e low-level
        if self.use_atomic:
            atomic_dict = {n: fn for n, fn, _ in get_atomic_programs()}
        else:
            atomic_dict = {n: fn for n, fn, _ in get_ops()}

        def _make_executor(ops_seq):
            def _exec(grid, params):
                result = grid.copy()
                for op_name in ops_seq:
                    fn = atomic_dict.get(op_name)
                    if fn:
                        try:
                            result = fn(result)
                        except Exception:
                            pass
                return result
            return _exec

        prim._execute = _make_executor(candidate.op_sequence)
        prim.confidence = candidate.score

        self.registry.register(prim)
        return True

    def get_stats(self) -> dict:
        return {"counter": self._counter}

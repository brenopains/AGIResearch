"""
PrimitiveVerifier — Valida candidatos a primitiva em multiplos exemplos.
"""
from typing import List, Tuple, Optional
import numpy as np

from .hypothesizer import CandidateProgram
from .low_level_ops import get_ops
from .atomic_ops import get_atomic_programs


class PrimitiveVerifier:
    def __init__(self, min_examples: int = 2, min_accuracy: float = 1.0,
                 use_atomic: bool = False):
        self.min_examples = min_examples
        self.min_accuracy = min_accuracy
        self.use_atomic = use_atomic

    def verify(self, candidate: CandidateProgram,
               train_examples: List[Tuple[np.ndarray, np.ndarray]],
               test_examples: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
               ) -> bool:
        if len(train_examples) < self.min_examples:
            return False

        train_ok = self._check_batch(candidate, train_examples)
        if not train_ok:
            return False

        if test_examples:
            test_ok = self._check_batch(candidate, test_examples)
            if not test_ok:
                return False

        inp0 = train_examples[0][0]
        out0 = train_examples[0][1]
        if np.array_equal(inp0, out0):
            return False

        return True

    def _check_batch(self, candidate: CandidateProgram,
                     examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        correct = 0
        for inp, out in examples:
            try:
                result = self._execute_candidate(candidate, inp)
                if result.shape == out.shape and np.array_equal(result, out):
                    correct += 1
            except Exception:
                pass
        return correct / max(1, len(examples)) >= self.min_accuracy

    def _execute_candidate(self, candidate: CandidateProgram,
                           grid: np.ndarray) -> np.ndarray:
        """Tenta atomic ops primeiro, depois low-level ops."""
        from .low_level_ops import get_ops as get_ll_ops
        from .atomic_ops import get_atomic_programs as get_at_ops

        # Tenta atomic ops primeiro
        atomic_dict = {name: fn for name, fn, _ in get_at_ops()}
        low_dict = {name: fn for name, fn, _ in get_ll_ops()}

        result = grid.copy()
        for op_name in candidate.op_sequence:
            fn = atomic_dict.get(op_name) or low_dict.get(op_name)
            if fn is None:
                raise ValueError(f"Unknown op: {op_name}")
            result = fn(result)
        return result

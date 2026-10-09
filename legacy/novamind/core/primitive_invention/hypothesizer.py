"""
ProgramHypothesizer — Gera PROGRAMAS CANDIDATOS a partir de low-level ops.

Dado um gap (invariante sem primitiva), o hypothesizer:
  1. Tenta TODAS as low-level ops no input
  2. Ve qual delas aproxima o output
  3. Se nenhuma funciona, tenta COMPOSICOES de 2 ops
  4. Se ainda nao, tenta composicoes de 3 ops
  5. Retorna os candidatos que mais reduzem o gap

Cada candidato e uma sequencia de low-level ops que pode virar primitiva.
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import time

from .low_level_ops import get_ops
from .atomic_ops import get_atomic_programs


@dataclass
class CandidateProgram:
    """Um programa candidato (sequencia de low-level ops) que pode virar primitiva."""
    op_sequence: List[str]         # nomes das ops na sequencia
    score: float = 0.0             # 0-1, quao bem funciona nos exemplos
    n_correct: int = 0
    n_tested: int = 0
    created_at: float = field(default_factory=time.time)
    validated: bool = False

    @property
    def name(self) -> str:
        return " -> ".join(self.op_sequence)

    def to_dict(self) -> dict:
        return {
            "program": self.name,
            "score": round(self.score, 3),
            "correct": f"{self.n_correct}/{self.n_tested}",
            "validated": self.validated,
        }


class ProgramHypothesizer:
    """Gera candidatos a programa para preencher uma lacuna.

    Busca gulosa por profundidade:
      depth=1: tenta cada low-level op isoladamente
      depth=2: tenta composicoes de 2 ops
      depth=3: tenta composicoes de 3 ops (so se depth<2 nao resolveu)
    """

    def __init__(self, max_depth: int = 3, use_atomic: bool = False):
        self.max_depth = max_depth
        self.use_atomic = use_atomic
        if use_atomic:
            self.ops = get_atomic_programs()
        else:
            self.ops = get_ops()

    def hypothesize(self, examples: List[Tuple[np.ndarray, np.ndarray]]
                    ) -> List[CandidateProgram]:
        """Gera programas candidatos para transformar input em output.

        Args:
            examples: lista de (input_grid, output_grid)

        Returns: lista de CandidateProgram ordenados por score.
        """
        if not examples:
            return []

        all_candidates: List[CandidateProgram] = []

        for depth in range(1, self.max_depth + 1):
            candidates = self._search_at_depth(examples, depth)
            all_candidates.extend(candidates)

            # Se achou candidato com score > 0.8, nao precisa buscar mais fundo
            if any(c.score > 0.8 for c in candidates):
                break

        all_candidates.sort(key=lambda c: (-c.score, -c.n_correct))
        return all_candidates[:10]

    def _search_at_depth(self, examples: List[Tuple[np.ndarray, np.ndarray]],
                          depth: int) -> List[CandidateProgram]:
        """Busca todos os programas de comprimento 'depth'."""
        inp0, out0 = examples[0]
        candidates = []

        if depth == 1:
            # Tenta cada operacao individual
            for op_name, op_fn, _ in self.ops:
                score, n_correct, n_tested = self._evaluate(op_fn, examples)
                if score > 0.05:
                    cand = CandidateProgram(
                        op_sequence=[op_name],
                        score=score,
                        n_correct=n_correct,
                        n_tested=n_tested,
                    )
                    # Se funciona em TODOS os exemplos, valida
                    if n_correct == n_tested and n_tested >= 2:
                        cand.validated = True
                    candidates.append(cand)

        elif depth == 2:
            # Tenta composicoes de 2 operacoes
            for name1, fn1, _ in self.ops:
                for name2, fn2, _ in self.ops:
                    if name1 == name2:
                        continue
                    def _compose(g, f1=fn1, f2=fn2):
                        try: return f2(f1(g))
                        except: return g.copy()
                    score, n_correct, n_tested = self._evaluate(_compose, examples)
                    if score > 0.1:
                        cand = CandidateProgram(
                            op_sequence=[name1, name2],
                            score=score,
                            n_correct=n_correct,
                            n_tested=n_tested,
                        )
                        if n_correct == n_tested and n_tested >= 2:
                            cand.validated = True
                        candidates.append(cand)

        elif depth == 3:
            # Tenta composicoes de 3 operacoes (amostra limitada)
            for name1, fn1, _ in self.ops[:8]:
                for name2, fn2, _ in self.ops[:8]:
                    for name3, fn3, _ in self.ops[:8]:
                        if len({name1, name2, name3}) < 2:
                            continue
                        def _compose3(g, f1=fn1, f2=fn2, f3=fn3):
                            try: return f3(f2(f1(g)))
                            except: return g.copy()
                        score, n_correct, n_tested = self._evaluate(_compose3, examples)
                        if score > 0.5:
                            cand = CandidateProgram(
                                op_sequence=[name1, name2, name3],
                                score=score,
                                n_correct=n_correct,
                                n_tested=n_tested,
                            )
                            if n_correct == n_tested and n_tested >= 2:
                                cand.validated = True
                            candidates.append(cand)

        return candidates

    def _evaluate(self, fn, examples) -> Tuple[float, int, int]:
        """Avalia uma funcao em todos os exemplos."""
        n_correct = 0
        n_tested = 0

        for inp, out in examples:
            n_tested += 1
            try:
                result = fn(inp)
                if result.shape == out.shape and np.array_equal(result, out):
                    n_correct += 1
            except Exception:
                pass

        score = n_correct / max(1, n_tested)
        return score, n_correct, n_tested

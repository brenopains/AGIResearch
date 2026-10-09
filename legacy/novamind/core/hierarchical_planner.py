"""
HierarchicalPlanner v2 — Planejador por iteracao de subgoals com reducao de gap.

Em vez de decompor o target em sub-targets artificiais, o planner:

  1. Identifica os tipos de gap (shape, objects, colors, position)
  2. Para cada tipo, TENTA primitives relevantes no estado ATUAL
  3. Escolhe a que MAIS REDUZ o gap total (nao a que melhor resolve o subgoal)
  4. Aplica, atualiza estado
  5. Proximo tipo de gap

Isso e "planejamento guloso por dimensao de gap".
"""
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import time

from .goal_driven_engine import GoalState, Gap, GoalComparator


@dataclass
class SubGoal:
    dimension: str
    priority: int
    description: str = ""
    solved: bool = False
    program: Any = None
    gap_before: float = 0.0
    gap_after: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "description": self.description,
            "solved": self.solved,
            "gap_reduction": round(self.gap_before - self.gap_after, 4),
        }


# Mapeia dimensao -> tipos de primitiva relevantes
DIMENSION_PRIMITIVES = {
    "shape": {"rotate_90", "rotate_180", "rotate_270",
              "flip_horizontal", "flip_vertical", "transpose",
              "transform", "composite"},
    "object_count": {"remove", "add", "transform", "composite",
                     "delete_color", "copy_object"},
    "color": {"recolor", "composite"},
    "position": {"move", "rotate_90", "rotate_180", "rotate_270",
                 "flip_horizontal", "flip_vertical", "transpose",
                 "composite"},
}

DIMENSION_ORDER = ["shape", "object_count", "color", "position"]


class HierarchicalPlanner:
    """Planejador que itera por dimensoes de gap, testando primitives relevantes."""

    def __init__(self, synthesizer=None):
        self.synthesizer = synthesizer
        self.comparator = GoalComparator()
        self._history: List[Dict] = []

    def plan(self, current: np.ndarray, target: np.ndarray,
             max_steps: int = 6) -> List[Any]:
        """Planeja sequencia de programas.

        Usa decomposicao por dimensao + beam search para combinar.
        """
        t_start = time.time()
        state = current.copy()
        programs: List[Any] = []
        tried_types = set()

        for dimension in DIMENSION_ORDER:
            if np.array_equal(state, target):
                break
            if len(programs) >= max_steps:
                break

            relevant = DIMENSION_PRIMITIVES.get(dimension, set()) - tried_types
            if not relevant:
                continue

            # Tenta sintetizar com filtro (inclui composicao se disponivel)
            best_prog = self._try_primitives(state, target, relevant | {"composite"}, tried_types)
            if best_prog is not None:
                new_state = best_prog.execute(state)
                gap_before = self.comparator.compute(state, GoalState(target=target))
                gap_after = self.comparator.compute(new_state, GoalState(target=target))

                if gap_after.total < gap_before.total:
                    programs.append(best_prog)
                    state = new_state
                    if hasattr(best_prog, 'instances'):
                        for inst in best_prog.instances:
                            tried_types.add(inst.primitive.structure_type)

        # Se o plano basico nao resolveu, tenta beam search a partir do estado ATUAL
        if not np.array_equal(state, target) and self.synthesizer is not None:
            full_search = self.synthesizer.synthesize(state, target)
            if full_search:
                best_beam = full_search[0]
                beam_state = best_beam.execute(state)
                beam_gap = self.comparator.compute(beam_state, GoalState(target=target))
                plan_gap = self.comparator.compute(state, GoalState(target=target))
                if beam_gap.total < plan_gap.total:
                    programs = [best_beam]
                    state = beam_state

        elapsed = time.time() - t_start
        self._history.append({
            "time": elapsed,
            "programs": len(programs),
            "final_gap": self.comparator.compute(state, GoalState(target=target)).total,
        })

        return programs

    def _try_primitives(self, state: np.ndarray, target: np.ndarray,
                        relevant_types: set,
                        tried_types: set) -> Optional[Any]:
        """Tenta primitives relevantes, retorna a que mais reduz gap."""
        if self.synthesizer is None:
            return None

        # So tenta tipos NAO testados ainda
        types_to_try = relevant_types - tried_types
        if not types_to_try:
            return None

        programs = self.synthesizer.synthesize(state, target,
                                                primitive_filter=types_to_try)
        if not programs:
            return None

        # Entre os candidatos, escolhe o que MAIS reduz o gap
        best = None
        best_reduction = 0

        for prog in programs[:5]:
            try:
                new_state = prog.execute(state)
                gap_before = self.comparator.compute(state, GoalState(target=target))
                gap_after = self.comparator.compute(new_state, GoalState(target=target))
                reduction = gap_before.total - gap_after.total

                if reduction > best_reduction:
                    best_reduction = reduction
                    best = prog
            except Exception:
                pass

        return best

    def get_stats(self) -> dict:
        if not self._history:
            return {"n_plans": 0}
        avg_p = np.mean([h.get("programs", 0) for h in self._history])
        avg_gap = np.mean([h["final_gap"] for h in self._history])
        return {
            "n_plans": len(self._history),
            "avg_programs": round(avg_p, 2),
            "avg_final_gap": round(avg_gap, 4),
        }


def compute_distance_to_solution(found_programs: list,
                                  target: np.ndarray,
                                  final_state: np.ndarray) -> dict:
    """Distancia ate a solucao."""
    if final_state.shape != target.shape:
        return {"program_distance": -1, "pixel_gap": 1.0, "structural_coverage": 0.0}

    total = int(target.size)
    wrong = int((final_state != target).sum())
    pixel_gap = wrong / max(1, total)

    dims_solved = 0
    dims_total = 4  # shape, count, colors, position

    if final_state.shape == target.shape:
        dims_solved += 1
    if (final_state > 0).sum() == (target > 0).sum():
        dims_solved += 1
    if (set(np.unique(final_state)) - {0}) == (set(np.unique(target)) - {0}):
        dims_solved += 1
    if np.array_equal(final_state, target):
        dims_solved += 1

    return {
        "program_distance": len(found_programs),
        "pixel_gap": round(pixel_gap, 4),
        "structural_coverage": round(dims_solved / dims_total, 4),
    }

"""
ProgramGraphSearch — Busca composicional com grafo aprendido e pruning estrutural.

Diferencas do beam search ingênuo:
  1. CompositionGraph: pesos de borda APRENDIDOS da experiencia
  2. Pruning estrutural: 90% das composicoes sao invalidas e jamais tentadas
  3. Profundidade DINAMICA: para quando score estagna
  4. Largura ADAPTATIVA: mais largura no inicio, menos conforme converge

O CompositionGraph e o pruning tornam a busca:
  - Nao cega (edges ponderados por historico)
  - Nao uniforme (profundidade/largura adaptativas)
  - Estrutural (constraints gerais de grid)
"""
from typing import List, Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass, field

from .primitive import PrimitiveRegistry, PrimitiveInstance
from .composition_graph import CompositionGraph
from .pruning import check_pruning
from .operator_grounding_bridge import OperatorGroundingBridge, OperatorProposal


@dataclass
class ProgramCandidate:
    instances: List[PrimitiveInstance] = field(default_factory=list)
    score: float = 0.0
    validated: bool = False
    score_history: List[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        if not self.instances:
            return "empty"
        return " -> ".join(p.name for p in self.instances)

    @property
    def n_steps(self) -> int:
        return len(self.instances)

    @property
    def last_improvement(self) -> float:
        if len(self.score_history) < 2:
            return self.score
        return self.score - self.score_history[-2]

    @property
    def stagnated(self) -> bool:
        if len(self.score_history) < 3:
            return False
        return all(self.score_history[-i] - self.score_history[-i-1] < 0.02 for i in range(1, 3))

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for inst in self.instances:
            result = inst.apply(result)
        return result

    def to_dict(self) -> dict:
        return {
            "program": self.name,
            "score": round(self.score, 4),
            "validated": self.validated,
            "n_steps": self.n_steps,
        }


class ProgramGraphSearch:
    """Busca composicional com grafo aprendido e pruning estrutural.

    Args:
        registry: PrimitiveRegistry
        composition_graph: CompositionGraph (aprende pesos da experiencia)
        max_depth: maximo de passos
        initial_beam: largura inicial do beam
        stagnation_patience: quantas iteracoes sem melhora antes de podar
    """

    def __init__(self, registry: PrimitiveRegistry,
                 composition_graph: Optional[CompositionGraph] = None,
                 grounding_bridge: Optional[OperatorGroundingBridge] = None,
                 max_depth: int = 6,
                 initial_beam: int = 8,
                 stagnation_patience: int = 2):
        self.registry = registry
        self.graph = composition_graph or CompositionGraph()
        self.bridge = grounding_bridge
        self.max_depth = max_depth
        self.initial_beam = initial_beam
        self.stagnation_patience = stagnation_patience

    def search(self, input_grid: np.ndarray,
               output_grid: np.ndarray) -> List[ProgramCandidate]:
        """Busca composicional com pruning e grafo aprendido."""

        beam: List[Tuple[ProgramCandidate, np.ndarray, List[Dict]]] = [
            (ProgramCandidate(), input_grid.copy(), [])
        ]

        best_score_ever = 0.0
        steps_without_improvement = 0
        all_candidates: List[ProgramCandidate] = []

        for depth in range(self.max_depth):
            if not beam:
                break

            # Largura adaptativa: mais no inicio, menos conforme converge
            adaptive_beam = max(2, self.initial_beam - depth)

            next_beam: List[Tuple[ProgramCandidate, np.ndarray, List[Dict]]] = []
            best_at_depth = 0.0

            for candidate, current_grid, history in beam:
                # Poda por estagnacao
                if candidate.stagnated and depth > 2:
                    continue

                # Pruning: pula se score estagnou ha muito tempo
                if (best_score_ever > 0 and
                    candidate.score < best_score_ever * 0.3 and
                        depth > 2):
                    continue

                # 0. OPERATOR PROPOSALS da bridge (CGE -> candidatos)
                proposals_to_try = []
                if self.bridge and depth == 0:
                    proposals_to_try = self.bridge.generate_proposals(
                        current_grid, output_grid
                    )
                for prop in proposals_to_try[:5]:
                    new_grid = prop.execute(current_grid)
                    if np.array_equal(new_grid, current_grid):
                        continue
                    if new_grid.shape != output_grid.shape:
                        continue
                    score = _compute_score(new_grid, output_grid)
                    if score < 0.05:
                        continue

                    # Se tiver exec_fn, vira primitive instance
                    from .primitive import Primitive
                    temp_prim = Primitive(
                        name=f"proposal_{prop.name}",
                        structure_type=prop.name,
                        description=prop.description,
                    )
                    if prop.execute_fn:
                        def _make_exec(fn):
                            def _exec(g, p):
                                try: return fn(g)
                                except: return g.copy()
                            return _exec
                        temp_prim._execute = _make_exec(prop.execute_fn)

                    temp_inst = PrimitiveInstance(primitive=temp_prim, params={})
                    new_history = history + [{"type": prop.name, "params": {}}]
                    new_candidate = ProgramCandidate(
                        instances=candidate.instances + [temp_inst],
                        score=score,
                        score_history=candidate.score_history + [score],
                    )
                    all_candidates.append(new_candidate)

                    if np.array_equal(new_grid, output_grid):
                        new_candidate.validated = True
                        # Promove proposta a primitiva real
                        if self.bridge:
                            self.bridge.promote_if_validated(prop, current_grid, output_grid)
                        return _rank_candidates(all_candidates)

                    next_beam.append((new_candidate, new_grid, new_history))

                for prim in self.registry.get_all(min_confidence=0.0):
                    ptype = prim.structure_type

                    # 1. PRUNING ESTRUTURAL: invalida 90% antes de tentar
                    instance = prim.generate_hypothesis(current_grid, output_grid)
                    if instance is None:
                        continue
                    prune_reasons = check_pruning(
                        history, current_grid, ptype, instance.params
                    )
                    if prune_reasons:
                        continue

                    # 2. Aplica e verifica se mudou algo
                    new_grid = instance.apply(current_grid)
                    if np.array_equal(new_grid, current_grid):
                        continue

                    # 3. Score
                    score = _compute_score(new_grid, output_grid)
                    if score < 0.05:
                        continue

                    new_history = history + [{"type": ptype, "params": instance.params}]
                    new_candidate = ProgramCandidate(
                        instances=candidate.instances + [instance],
                        score=score,
                        score_history=candidate.score_history + [score],
                    )

                    if score > best_score_ever:
                        best_score_ever = score
                        steps_without_improvement = 0
                    else:
                        steps_without_improvement += 1

                    all_candidates.append(new_candidate)

                    # 4. Exact match?
                    if np.array_equal(new_grid, output_grid):
                        new_candidate.validated = True
                        # Registra a composicao no grafo
                        types = [inst.primitive.structure_type for inst in new_candidate.instances]
                        self.graph.record_program(types, succeeded=True)
                        return _rank_candidates(all_candidates)

                    next_beam.append((new_candidate, new_grid, new_history))

                    if score > best_at_depth:
                        best_at_depth = score

            # Poda por estagnacao global
            if steps_without_improvement > self.stagnation_patience * 3:
                break

            # Ordena e mantem adaptive_beam
            next_beam.sort(key=lambda x: -x[0].score)
            beam = next_beam[:adaptive_beam]

            # Se ninguem melhorou, reduz beam
            if best_at_depth <= best_score_ever * 0.9 and depth > 1:
                beam = beam[:max(1, adaptive_beam // 2)]

        # Registra tentativas no grafo (mesmo as falhas)
        for cand in all_candidates[:10]:
            types = [inst.primitive.structure_type for inst in cand.instances]
            self.graph.record_program(types, succeeded=False)

        return _rank_candidates(all_candidates)


def _compute_score(result: np.ndarray, target: np.ndarray) -> float:
    """Score com penalidade por diferenca de cores e objetos."""
    if result.shape != target.shape:
        return 0.0
    total = int(target.size)
    if total == 0:
        return 1.0
    wrong = int((result != target).sum())
    pixel_acc = 1.0 - wrong / total
    r_objs = (result > 0).sum()
    t_objs = (target > 0).sum()
    obj_sim = min(r_objs, t_objs) / max(1, max(r_objs, t_objs))
    r_colors = set(int(c) for c in np.unique(result) if c != 0)
    t_colors = set(int(c) for c in np.unique(target) if c != 0)
    color_sim = len(r_colors & t_colors) / max(1, len(t_colors)) if t_colors else 1.0
    return round(pixel_acc * 0.5 + obj_sim * 0.25 + color_sim * 0.25, 4)


def _rank_candidates(candidates: List[ProgramCandidate]) -> List[ProgramCandidate]:
    """Ordena: validated primeiro, depois score."""
    validated = [c for c in candidates if c.validated]
    not_validated = [c for c in candidates if not c.validated]
    validated.sort(key=lambda c: (-c.score, c.n_steps))
    not_validated.sort(key=lambda c: (-c.score, c.n_steps))
    result = validated + not_validated
    seen = set()
    unique = []
    for c in result:
        key = tuple(p.primitive.structure_type for p in c.instances)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:20]

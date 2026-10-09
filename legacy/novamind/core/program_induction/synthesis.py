"""
ProgramSynthesizer — Sintese por CONSTRAINT SOLVING + COMPOSITIONAL BEAM SEARCH.

Fluxo:
  1. Tenta primitiva unica (rapido)
  2. Se nao achou match exato → beam search composicional
  3. Cada passo do beam: solver INFERE parametros para o grid ATUAL
  4. A constraint propaga: solver do passo N ve output do passo N-1
  5. Nao ha enumeracao — cada solver usa constraint satisfaction
"""
from typing import List, Optional, Dict
import numpy as np

from .primitive import PrimitiveRegistry, PrimitiveInstance
from .delta_extractor import DeltaExtractor
from .graph_search import ProgramGraphSearch, ProgramCandidate
from .composition_graph import CompositionGraph
from .operator_grounding_bridge import OperatorGroundingBridge


def _program_from_candidate(cand: ProgramCandidate) -> 'Program':
    return Program(
        primitives=cand.instances,
        score=cand.score,
        validated=cand.validated,
    )


class ProgramSynthesizer:
    """Sintetizador que combina single-query + beam search composicional."""

    def __init__(self, registry: PrimitiveRegistry,
                 composition_graph: CompositionGraph = None,
                 grounding_bridge: OperatorGroundingBridge = None,
                 beam_width: int = 5, max_depth: int = 4,
                 min_score_threshold: float = 0.1):
        self.registry = registry
        self.composition_graph = composition_graph or CompositionGraph()
        self.grounding_bridge = grounding_bridge or OperatorGroundingBridge(
            registry=registry, composition_graph=self.composition_graph
        )
        self.graph_search = ProgramGraphSearch(
            registry=registry,
            composition_graph=self.composition_graph,
            grounding_bridge=self.grounding_bridge,
            max_depth=max_depth,
            initial_beam=beam_width,
        )
        self._beam_width = beam_width
        self._max_depth = max_depth

    def set_search_params(self, beam_width: int = None, max_depth: int = None):
        """Ajusta parametros de busca dinamicamente (ex: por modo cognitivo)."""
        if beam_width is not None:
            self._beam_width = beam_width
            self.graph_search.initial_beam = beam_width
        if max_depth is not None:
            self._max_depth = max_depth
            self.graph_search.max_depth = max_depth

    def synthesize(self, input_grid: np.ndarray,
                   output_grid: np.ndarray,
                   primitive_filter: set = None,
                   primitive_weights: Optional[Dict[str, float]] = None) -> List['Program']:
        """Sintetiza programas candidatos.

        Args:
            primitive_filter: se fornecido, so usa primitives destes tipos.
            primitive_weights: dict {structure_type: boost_factor} para
                               guiar busca (soft guidance). Ex: {"rotate_90": 2.0}

        Returns lista ordenada por score.
        """
        # Guarda estado original do filtro
        if primitive_filter:
            self._filter = primitive_filter
        else:
            self._filter = None

        self._weights = primitive_weights or {}

        # 1. Tenta primitiva unica primeiro
        single_results = self._try_single(input_grid, output_grid)
        if self._has_exact(single_results):
            self._filter = None
            self._weights = {}
            return single_results

        # 2. Beam search composicional
        beam_results = self.graph_search.search(input_grid, output_grid)

        self._filter = None
        self._weights = {}

        seen = set()
        combined = []

        for prog in single_results + beam_results:
            key = tuple(p.primitive.structure_type for p in prog.instances) if hasattr(prog, 'instances') else tuple()
            if key not in seen:
                seen.add(key)
                combined.append(prog)

        combined.sort(key=lambda p: (p.validated, p.score), reverse=True)
        return combined[:20]

    def _try_single(self, inp: np.ndarray, out: np.ndarray) -> List['Program']:
        """Tenta cada primitiva individualmente, com pesos opcionais."""
        results = []
        primitives = self.registry.get_all(min_confidence=0.0)

        # Aplica filtro se definido
        if hasattr(self, '_filter') and self._filter is not None:
            primitives = [p for p in primitives if p.structure_type in self._filter]

        # Ordena por peso (soft guidance): primitivas com maior boost primeiro
        weights = getattr(self, '_weights', {})
        primitives = sorted(primitives, key=lambda p: -weights.get(p.structure_type, 1.0))

        for prim in primitives:
            instance = prim.generate_hypothesis(inp, out)
            if instance is None:
                continue
            result = instance.apply(inp)
            if result.shape != out.shape:
                continue

            score = _compute_score(result, out)
            score = _apply_weights(score, prim.structure_type, weights)
            prog = Program(primitives=[instance], score=score)
            if np.array_equal(result, out):
                prog.validated = True
                prog.score = 1.0
                prim.record_attempt(success=True)
            else:
                prim.record_attempt(success=False)

            if score > 0.3:
                results.append(prog)
        return results

    def _has_exact(self, programs: List['Program']) -> bool:
        return any(p.validated and p.score >= 1.0 for p in programs)


def _apply_weights(score: float, structure_type: str,
                    weights: dict = None) -> float:
    """Aplica peso de boost ao score (soft guidance)."""
    if weights is None:
        return score
    boost = weights.get(structure_type, 1.0)
    return min(1.0, score * boost)


def _compute_score(result: np.ndarray, target: np.ndarray) -> float:
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


class Program:
    """Um programa composto por instancias de primitives com parametros resolvidos."""

    def __init__(self, primitives: List[PrimitiveInstance] = None,
                 score: float = 0.0, validated: bool = False):
        self.instances = primitives or []
        self.score = score
        self.validated = validated

    @property
    def primitives(self) -> List[PrimitiveInstance]:
        return self.instances

    @property
    def name(self) -> str:
        if not self.instances:
            return "empty"
        return " -> ".join(p.name for p in self.instances)

    @property
    def n_steps(self) -> int:
        return len(self.instances)

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for p in self.instances:
            result = p.apply(result)
        return result

    def to_dict(self) -> dict:
        return {
            "program": self.name,
            "score": round(self.score, 4),
            "validated": self.validated,
            "n_steps": self.n_steps,
        }

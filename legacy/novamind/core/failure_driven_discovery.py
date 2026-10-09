"""
FailureDrivenDiscovery — Descobre NOVOS OPERADORES a partir de falhas.

O ciclo:
  1. Sistema tenta um goal e FALHA (gap residual > 0)
  2. FailureAnalyzer classifica o TIPO da falha
  3. FailureCluster agrupa falhas similares
  4. OperatorDiscovery INFERE um operador candidato do padrao
  5. OperatorValidator testa se reduz gap em falhas similares
  6. Se validado, o operador e ADICIONADO AO REGISTRY

Isso transforma fracasso em EXPANSAO DE CAPACIDADE.
"""
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import time
import random

from .goal_driven_engine import GoalState, Gap, GoalComparator


# ─── Failure Types ─────────────────────────────────────────

FAILURE_TYPES = [
    "NO_OPERATOR_MATCH",       # Nenhuma primitiva se aplica
    "WRONG_OBJECT_COUNT",      # Numero de objetos diferente
    "SPATIAL_MISMATCH",        # Objetos em posicoes erradas
    "COLOR_MISMATCH",          # Estrutura certa, cores erradas
    "PARTIAL_MATCH",           # Parcialmente certo
    "COMPOSITION_REQUIRED",    # Precisa de multiplos passos
    "SHAPE_MISMATCH",          # Grid tem dimensoes diferentes
    "PATTERN_REQUIRED",        # Precisa replicar padrao
]


# ─── FailureSignature — a "impressao digital" de uma falha ─

@dataclass
class FailureSignature:
    """Assinatura estrutural de uma falha."""
    task_id: str = ""
    failure_type: str = ""
    input_objects: int = 0
    output_objects: int = 0
    input_colors: set = field(default_factory=set)
    output_colors: set = field(default_factory=set)
    input_shape: tuple = (0, 0)
    output_shape: tuple = (0, 0)
    gap_before: float = 0.0
    gap_after: float = 0.0
    gap_structural: float = 0.0
    gap_semantic: float = 0.0
    n_actions_tried: int = 0
    primitives_tried: List[str] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "failure_type": self.failure_type,
            "input_objects": self.input_objects,
            "output_objects": self.output_objects,
            "gap_before": round(self.gap_before, 3),
            "gap_after": round(self.gap_after, 3),
            "details": self.details,
        }


# ─── FailureAnalyzer — classifica a falha ─────────────────

class FailureAnalyzer:
    """Analisa uma falha (gap residual) e classifica seu tipo estrutural."""

    def __init__(self):
        self.comparator = GoalComparator()

    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray,
                current_state: np.ndarray, gap: Gap,
                task_id: str = "",
                primitives_tried: list = None) -> FailureSignature:
        """Analisa a falha e retorna uma assinatura."""
        # Extrai caracteristicas
        in_objs = (input_grid > 0).sum()
        out_objs = (output_grid > 0).sum()
        in_colors = set(int(c) for c in np.unique(input_grid) if c != 0)
        out_colors = set(int(c) for c in np.unique(output_grid) if c != 0)

        # Calcula quantos objetos ja estao corretos
        current_nonzero = set(zip(*np.where(current_state > 0))) if np.any(current_state > 0) else set()
        target_nonzero = set(zip(*np.where(output_grid > 0))) if np.any(output_grid > 0) else set()
        overlap = len(current_nonzero & target_nonzero) if current_nonzero and target_nonzero else 0

        # Classifica o tipo de falha
        ftype = self._classify(
            input_grid, output_grid, current_state, gap,
            in_objs, out_objs, in_colors, out_colors,
        )

        return FailureSignature(
            task_id=task_id,
            failure_type=ftype,
            input_objects=in_objs,
            output_objects=out_objs,
            input_colors=in_colors,
            output_colors=out_colors,
            input_shape=input_grid.shape,
            output_shape=output_grid.shape,
            gap_before=gap.total,
            gap_after=gap.total,  # mesmo valor antes/depois da falha
            gap_structural=gap.structural_gap,
            gap_semantic=gap.semantic_gap,
            primitives_tried=primitives_tried or [],
            details=self._detail(ftype, in_objs, out_objs, in_colors, out_colors, overlap),
        )

    def _classify(self, inp, out, current, gap,
                  in_objs, out_objs, in_cols, out_cols) -> str:
        """Classifica o tipo estrutural da falha."""
        if inp.shape != out.shape:
            return "SHAPE_MISMATCH"

        if gap.structural_gap > 0.7:
            return "NO_OPERATOR_MATCH"

        if in_objs != out_objs:
            return "WRONG_OBJECT_COUNT"

        if in_cols != out_cols:
            return "COLOR_MISMATCH"

        if gap.structural_gap < 0.3 and gap.structural_gap > 0.05:
            return "PARTIAL_MATCH"

        return "SPATIAL_MISMATCH"

    def _detail(self, ftype, in_objs, out_objs, in_cols, out_cols, overlap) -> str:
        """Gera descricao textual do tipo de falha."""
        if ftype == "SHAPE_MISMATCH":
            return f"grid dimensions differ"
        if ftype == "NO_OPERATOR_MATCH":
            return f"no primitive can transform this structure"
        if ftype == "WRONG_OBJECT_COUNT":
            return f"objects: {in_objs} -> {out_objs}"
        if ftype == "COLOR_MISMATCH":
            return f"colors differ: in={in_cols}, out={out_cols}"
        if ftype == "PARTIAL_MATCH":
            return f"{overlap} pixels correct out of {max(out_objs,1)}"
        return f"spatial arrangement differs"


# ─── FailureCluster — agrupa falhas similares ─────────────

@dataclass
class FailureCluster:
    """Um grupo de falhas com a mesma assinatura estrutural."""
    failure_type: str
    signatures: List[FailureSignature] = field(default_factory=list)
    pattern_description: str = ""
    discovered_at: float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        return len(self.signatures)

    @property
    def avg_gap(self) -> float:
        return sum(s.gap_after for s in self.signatures) / max(1, self.count)

    def infer_pattern(self) -> str:
        """Tenta inferir um padrao comum entre as falhas."""
        if self.count < 2:
            return "insufficient_data"

        sigs = self.signatures

        # WRONG_OBJECT_COUNT: qual a relacao entre in/out?
        if self.failure_type == "WRONG_OBJECT_COUNT":
            ratios = []
            for s in sigs:
                if s.input_objects > 0:
                    ratios.append(s.output_objects / s.input_objects)
            if ratios:
                avg_r = sum(ratios) / len(ratios)
                if abs(avg_r - round(avg_r)) < 0.1:
                    return f"objects multiply by ~{round(avg_r)}"
                return f"objects change ratio ~{avg_r:.2f}"

        # COLOR_MISMATCH: qual cor aparece/some?
        if self.failure_type == "COLOR_MISMATCH":
            missing = set()
            added = set()
            for s in sigs:
                missing |= s.input_colors - s.output_colors
                added |= s.output_colors - s.input_colors
            if missing:
                return f"remove color(s): {missing}"
            if added:
                return f"add color(s): {added}"

        return self.failure_type


# ─── OperatorDiscovery — infere operador candidato ────────

@dataclass
class CandidateOperator:
    """Um operador candidato descoberto de um padrao de falha."""
    name: str
    failure_type: str
    description: str
    confidence: float = 0.0
    validated: bool = False
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.attempts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "failure_type": self.failure_type,
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "validated": self.validated,
            "success_rate": round(self.success_rate, 3),
        }


class OperatorDiscovery:
    """Infere operadores candidatos a partir de clusters de falha.

    O mecanismo:
      1. Recebe um FailureCluster
      2. Analisa o padrao estrutural
      3. Gera um ou mais candidatos a operador
      4. Cada candidato vem com uma funcao de execucao GENERICA

    NADA e ARC-specific. As funcoes sao operacoes gerais de grid.
    """

    def __init__(self):
        self.candidates: List[CandidateOperator] = []

    def discover(self, cluster: FailureCluster) -> List[CandidateOperator]:
        """Descobre candidatos a partir de um cluster de falha."""
        new_candidates = []

        if cluster.failure_type == "WRONG_OBJECT_COUNT":
            pattern = cluster.infer_pattern()
            if "multiply" in pattern:
                factor = int(pattern.split("~")[1]) if "~" in pattern else 2
                new_candidates.append(self._make_candidate(
                    name="replicate",
                    failure_type=cluster.failure_type,
                    description=f"replicate objects ~{factor}x",
                    exec_fn=lambda g, f=factor: self._exec_replicate(g, f),
                ))
            else:
                new_candidates.append(self._make_candidate(
                    name="fill_gaps",
                    failure_type=cluster.failure_type,
                    description="fill empty cells around objects",
                    exec_fn=self._exec_fill_gaps,
                ))

        elif cluster.failure_type == "COLOR_MISMATCH":
            pattern = cluster.infer_pattern()
            if "remove color" in pattern:
                new_candidates.append(self._make_candidate(
                    name="remove_unmatched_colors",
                    failure_type=cluster.failure_type,
                    description="remove colors that don't match target",
                    exec_fn=self._exec_remove_unmatched,
                ))

        elif cluster.failure_type == "SHAPE_MISMATCH":
            new_candidates.append(self._make_candidate(
                name="resize_grid",
                failure_type=cluster.failure_type,
                description="resize grid to match target dimensions",
                exec_fn=self._exec_resize,
            ))

        elif cluster.failure_type == "NO_OPERATOR_MATCH":
            new_candidates.append(self._make_candidate(
                name="structural_remap",
                failure_type=cluster.failure_type,
                description="remap non-zero pixels to target structure",
                exec_fn=self._exec_structural_remap,
            ))

        self.candidates.extend(new_candidates)
        return new_candidates

    def get_candidates(self, min_confidence: float = 0.0) -> List[CandidateOperator]:
        return [c for c in self.candidates if c.confidence >= min_confidence]

    def get_stats(self) -> dict:
        return {
            "total_candidates": len(self.candidates),
            "validated": sum(1 for c in self.candidates if c.validated),
            "avg_confidence": sum(c.confidence for c in self.candidates) / max(1, len(self.candidates)),
        }

    def _make_candidate(self, name, failure_type, description, exec_fn) -> CandidateOperator:
        c = CandidateOperator(
            name=f"discovered_{name}_{int(time.time())}",
            failure_type=failure_type,
            description=description,
        )
        c._exec_fn = exec_fn
        return c

    # ─── Funcoes de execucao genericas ─────────────────────

    @staticmethod
    def _exec_replicate(grid: np.ndarray, factor: int = 2) -> np.ndarray:
        """Replica objetos no grid (operacao geral, nao ARC-specific)."""
        h, w = grid.shape
        new_h = min(h * factor, 30)
        new_w = min(w * factor, 30)
        result = np.zeros((new_h, new_w), dtype=grid.dtype)
        for i in range(min(factor, 3)):
            for j in range(min(factor, 3)):
                rh = i * h
                rw = j * w
                if rh + h <= new_h and rw + w <= new_w:
                    result[rh:rh + h, rw:rw + w] = grid
        return result

    @staticmethod
    def _exec_fill_gaps(grid: np.ndarray) -> np.ndarray:
        """Preenche celulas vazias entre objetos."""
        result = grid.copy()
        for r in range(1, grid.shape[0] - 1):
            for c in range(1, grid.shape[1] - 1):
                if grid[r, c] == 0:
                    neighbors = [grid[r - 1, c], grid[r + 1, c],
                                 grid[r, c - 1], grid[r, c + 1]]
                    non_zero = [n for n in neighbors if n > 0]
                    if len(non_zero) >= 3:
                        result[r, c] = max(set(non_zero), key=non_zero.count)
        return result

    @staticmethod
    def _exec_remove_unmatched(grid: np.ndarray) -> np.ndarray:
        """Placeholder: precisa do target para saber quais cores remover."""
        return grid

    @staticmethod
    def _exec_resize(grid: np.ndarray) -> np.ndarray:
        """Redimensiona grid para 30x30 (tamanho padrao ARC)."""
        result = np.zeros((30, 30), dtype=grid.dtype)
        h, w = grid.shape
        result[:min(h, 30), :min(w, 30)] = grid[:min(h, 30), :min(w, 30)]
        return result

    @staticmethod
    def _exec_structural_remap(grid: np.ndarray) -> np.ndarray:
        """Placeholder para remapeamento estrutural."""
        return grid


# ─── OperatorValidator — testa candidato em falhas similares ─

class OperatorValidator:
    """Valida um operador candidato testando em falhas similares.

    So promove o operador se ele:
      1. Reduz o gap em pelo menos 50% das falhas do cluster
      2. Nao piora o gap em nenhuma
      3. Generaliza (funciona em pelo menos 2 tasks diferentes)
    """

    def __init__(self, goal_comparator: GoalComparator = None):
        self.comparator = goal_comparator or GoalComparator()

    def validate(self, candidate: CandidateOperator,
                 cluster: FailureCluster,
                 input_grids: List[np.ndarray],
                 output_grids: List[np.ndarray]) -> bool:
        """Valida um candidato contra falhas do cluster.

        Args:
            candidate: operador candidato
            cluster: cluster de falhas
            input_grids: grids de entrada das falhas
            output_grids: grids de saida (target) das falhas

        Returns: True se candidato for validado.
        """
        if not hasattr(candidate, "_exec_fn") or not input_grids:
            return False

        successes = 0
        total = len(input_grids)

        for inp, out in zip(input_grids, output_grids):
            try:
                before_gap = self.comparator.compute(inp, GoalState(target=out))
                result = candidate._exec_fn(inp)
                if result.shape != out.shape:
                    continue
                after_gap = self.comparator.compute(result, GoalState(target=out))
                if after_gap.total < before_gap.total * 0.8:
                    successes += 1
                candidate.attempts += 1
                if after_gap.total < 0.1:
                    candidate.successes += 1
            except Exception:
                pass

        if total >= 2 and successes >= max(1, total // 2):
            candidate.validated = True
            candidate.confidence = successes / total
            return True

        return False


# ─── FailureDrivenLoop — integracao completa ──────────────

class FailureDrivenLoop:
    """Loop completo de descoberta orientada a falhas.

    Uso:
        loop = FailureDrivenLoop(goal_engine, registry)
        loop.on_failure(input_grid, output_grid, current_state, gap, task_id)

    O loop automaticamente:
      1. Analisa a falha
      2. Agrupa com falhas similares
      3. Se cluster tem tamanho suficiente, tenta descobrir operador
      4. Valida o operador
      5. Se validado, adiciona ao registry
    """

    def __init__(self, registry=None, goal_engine=None,
                 min_cluster_size: int = 3):
        self.analyzer = FailureAnalyzer()
        self.discovery = OperatorDiscovery()
        self.validator = OperatorValidator()
        self.clusters: Dict[str, FailureCluster] = {}
        self.failure_history: List[FailureSignature] = []
        self.registry = registry
        self.goal_engine = goal_engine
        self.min_cluster_size = min_cluster_size
        self._input_cache: Dict[str, np.ndarray] = {}
        self._output_cache: Dict[str, np.ndarray] = {}

    def on_failure(self, input_grid: np.ndarray, output_grid: np.ndarray,
                   current_state: np.ndarray, gap: Gap,
                   task_id: str = "",
                   primitives_tried: list = None) -> Optional[CandidateOperator]:
        """Processa uma falha. Se descobrir novo operador, retorna-o."""

        # 1. Analisa
        sig = self.analyzer.analyze(
            input_grid, output_grid, current_state, gap,
            task_id, primitives_tried,
        )
        self.failure_history.append(sig)
        self._input_cache[task_id] = input_grid
        self._output_cache[task_id] = output_grid

        ftype = sig.failure_type

        # 2. Agrupa em cluster
        if ftype not in self.clusters:
            self.clusters[ftype] = FailureCluster(failure_type=ftype)
        self.clusters[ftype].signatures.append(sig)

        # 3. Tenta descobrir operador se cluster grande o suficiente
        cluster = self.clusters[ftype]
        if cluster.count >= self.min_cluster_size:
            # So tenta uma vez por tipo de falha
            existing = [c for c in self.discovery.candidates
                       if c.failure_type == ftype]
            if existing:
                return None

            candidates = self.discovery.discover(cluster)
            if not candidates:
                return None

            candidate = candidates[0]

            # 4. Valida
            task_ids = [s.task_id for s in cluster.signatures]
            inputs = [self._input_cache.get(tid) for tid in task_ids if tid in self._input_cache]
            outputs = [self._output_cache.get(tid) for tid in task_ids if tid in self._output_cache]

            if len(inputs) < 2:
                return None

            validated = self.validator.validate(candidate, cluster, inputs, outputs)

            # 5. Se validado, adiciona ao registry
            if validated and self.registry is not None:
                self._add_to_registry(candidate, ftype)

            return candidate

        return None

    def get_stats(self) -> dict:
        return {
            "total_failures": len(self.failure_history),
            "clusters": {ft: c.count for ft, c in self.clusters.items()},
            "candidates": self.discovery.get_stats(),
        }

    def get_clusters(self) -> dict:
        return {
            ft: {
                "count": c.count,
                "avg_gap": c.avg_gap,
                "pattern": c.infer_pattern(),
            }
            for ft, c in self.clusters.items()
        }

    def _add_to_registry(self, candidate: CandidateOperator, failure_type: str):
        """Adiciona operador validado ao PrimitiveRegistry."""
        if self.registry is None:
            return

        from .program_induction.primitive import Primitive

        prim = Primitive(
            name=candidate.name,
            structure_type=failure_type.lower(),
            description=candidate.description,
        )

        if hasattr(candidate, "_exec_fn"):
            fn = candidate._exec_fn
            def _make_exec(f):
                def _exec(g, p):
                    try:
                        return f(g)
                    except:
                        return g.copy()
                return _exec
            prim._execute = _make_exec(fn)

        prim.confidence = candidate.confidence
        self.registry.register(prim)
        print(f"[FAILURE-DRIVEN] Novo operador descoberto: {candidate.name} "
              f"(type={failure_type}, conf={candidate.confidence:.2f})")

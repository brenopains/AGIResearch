"""
GoalDrivenEngine — Transforma o sistema de INDUCTOR PASSIVO em AGENTE ORIENTADO A OBJETIVOS.

O salto conceitual:
  Antes: input → delta → programa → output (observador)
  Depois: goal → gap → programa → acao → novo estado → feedback (agente)

A engine usa o ProgramInductionEngine existente para descobrir programas,
mas agora a SELECAO do programa e guiada por "qual reduz mais o gap ate o goal".
"""
from typing import List, Optional, Dict, Tuple, Callable, Any
from dataclasses import dataclass, field
import time
import numpy as np


# ─── GoalState — o que o sistema quer atingir ─────────────

@dataclass
class GoalState:
    """Estado desejado que o sistema quer atingir.

    Pode ser:
      - Um grid especifico (para ARC/visao)
      - Um estado CGE (para cognicao)
      - Uma representacao abstrata (para goals genericos)

    O gap comparator sabe como medir distancia ate este estado.
    """
    target: Any = None                  # estado desejado (grid, CGE state, etc.)
    constraints: List[str] = field(default_factory=list)   # "whale is mammal"
    priority: float = 0.5               # 0-1, importancia
    uncertainty: float = 0.0             # 0-1, quao bem definido
    description: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def is_well_defined(self) -> bool:
        return self.target is not None and self.uncertainty < 0.5


# ─── Gap — distancia entre estado atual e goal ────────────

@dataclass
class Gap:
    """Diferenca quantitativa entre estado atual e estado desejado.

    Tres componentes:
      - structural_gap: diferenca de estrutura (objetos, nos, arestas)
      - semantic_gap: diferenca de significado (confianca, tipo)
      - confidence_gap: quao certo estamos de que o gap existe
    """
    structural_gap: float = 1.0
    semantic_gap: float = 1.0
    confidence_gap: float = 0.0
    details: List[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        """Gap total normalizado (0 = goal atingido, 1 = max gap)."""
        w_s, w_m, w_c = 0.4, 0.4, 0.2
        return w_s * self.structural_gap + w_m * self.semantic_gap + w_c * self.confidence_gap

    @property
    def is_zero(self) -> bool:
        return self.total < 0.01

    def to_dict(self) -> dict:
        return {
            "structural": round(self.structural_gap, 3),
            "semantic": round(self.semantic_gap, 3),
            "confidence": round(self.confidence_gap, 3),
            "total": round(self.total, 3),
        }


# ─── GoalComparator — mede distancia ate o goal ──────────

class GoalComparator:
    """Compara estado atual com estado desejado e computa gap.

    Funciona para MULTIPLOS DOMINIOS:
      - grids (ARC/visao): comparacao pixel + objeto + cor
      - CGE (cognicao): comparacao de nos + arestas + confianca
      - abstrato: comparacao por funcao customizada
    """

    @staticmethod
    def compute(current: Any, goal: GoalState) -> Gap:
        """Computa gap entre estado atual e goal.

        O metodo exato depende do tipo do target.
        """
        target = goal.target
        if target is None:
            return Gap(structural_gap=1.0, semantic_gap=1.0,
                       details=["no target defined"])

        # Grid -> grid
        if isinstance(current, np.ndarray) and isinstance(target, np.ndarray):
            return GoalComparator._grid_gap(current, target)

        # Dict (CGE state) -> Dict
        if isinstance(current, dict) and isinstance(target, dict):
            return GoalComparator._cge_gap(current, target)

        # Fallback: funcao customizada
        if callable(target):
            return target(current)

        return Gap(structural_gap=1.0, semantic_gap=1.0,
                   details=["unknown type"])

    @staticmethod
    def _grid_gap(current: np.ndarray, target: np.ndarray) -> Gap:
        """Gap entre dois grids."""
        if current.shape != target.shape:
            return Gap(structural_gap=1.0, semantic_gap=1.0,
                       details=[f"shape mismatch: {current.shape} vs {target.shape}"])

        total = int(target.size)
        wrong = int((current != target).sum())
        pixel_diff = wrong / max(1, total)

        # Diferenca de objetos
        c_objs = (current > 0).sum()
        t_objs = (target > 0).sum()
        obj_diff = abs(c_objs - t_objs) / max(1, max(c_objs, t_objs, 1))

        # Diferenca de cores
        c_colors = set(int(c) for c in np.unique(current) if c != 0)
        t_colors = set(int(c) for c in np.unique(target) if c != 0)
        color_diff = 1.0 - (len(c_colors & t_colors) / max(1, len(t_colors))) if t_colors else 0.0

        details = []
        if pixel_diff > 0.1:
            details.append(f"pixel difference: {wrong}/{total}")
        if obj_diff > 0.1:
            details.append(f"object count: {c_objs} vs {t_objs}")
        if color_diff > 0.1:
            details.append(f"color set differs")

        return Gap(
            structural_gap=round(pixel_diff * 0.5 + obj_diff * 0.3 + color_diff * 0.2, 4),
            semantic_gap=round(color_diff, 4),
            details=details,
        )

    @staticmethod
    def _cge_gap(current: dict, target: dict) -> Gap:
        """Gap entre dois estados CGE."""
        current_nodes = set(current.get("nodes", {}).keys())
        target_nodes = set(target.get("nodes", {}).keys())

        missing_nodes = target_nodes - current_nodes
        extra_nodes = current_nodes - target_nodes
        node_diff = (len(missing_nodes) + len(extra_nodes)) / max(1, len(target_nodes))

        # Arestas
        current_edges = set((e.get("from"), e.get("to")) for e in current.get("edges", []))
        target_edges = set((e.get("from"), e.get("to")) for e in target.get("edges", []))
        missing_edges = target_edges - current_edges
        edge_diff = len(missing_edges) / max(1, len(target_edges))

        details = []
        if missing_nodes:
            details.append(f"missing nodes: {list(missing_nodes)[:3]}")
        if missing_edges:
            details.append(f"missing edges: {len(missing_edges)}")

        return Gap(
            structural_gap=round(node_diff * 0.4 + edge_diff * 0.6, 4),
            semantic_gap=round(edge_diff, 4),
            details=details,
        )


# ─── ActionSelector — escolhe programa que minimiza gap ───

@dataclass
class Action:
    """Uma acao candidata: programa + gap estimado apos execucao."""
    program: Any                     # Program do ProgramInductionEngine
    estimated_gap_after: Gap = field(default_factory=Gap)
    execution_cost: float = 0.0
    confidence: float = 0.0

    @property
    def gap_reduction(self) -> float:
        """Quanto este acao reduz o gap (0-1)."""
        return 1.0 - self.estimated_gap_after.total

    @property
    def utility(self) -> float:
        """Utilidade = reducao de gap * confianca / custo."""
        return self.gap_reduction * self.confidence / max(0.01, self.execution_cost)


class ActionSelector:
    """Seleciona a melhor acao baseado em reducao de gap.

    Nao seleciona por score maximo.
    Seleciona por MAXIMA REDUCAO DE GAP.
    Pode receber concept_hint do CFE para preferir primitives conceituais.
    """

    def __init__(self):
        self._history: List[Dict] = []

    def select(self, actions: List[Action], goal: GoalState,
               current_gap: Gap,
               concept_hint: str = None,
               concept_primitives: set = None) -> Optional[Action]:
        """Seleciona a acao que MAIS REDUZ o gap.

        Se concept_hint for fornecido (do CFE), da um boost de 15%
        no score de programas que usam primitives do conceito.
        """
        if not actions:
            return None

        scored = []
        for a in actions:
            reduction = a.gap_reduction
            structural_reduction = 1.0 - a.estimated_gap_after.structural_gap
            score = reduction * 0.6 + structural_reduction * 0.4

            # Boost do CFE: prefere programas que usam primitives do conceito
            if concept_hint and concept_primitives:
                program_types = self._extract_types(a.program)
                if program_types & concept_primitives:
                    score *= 1.15  # Boost de 15% para primitives conceituais

            scored.append((score, a))

        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]

        self._history.append({
            "time": time.time(),
            "goal": goal.description,
            "n_actions": len(actions),
            "selected": str(best.program)[:60] if hasattr(best.program, "name") else "?",
            "gap_before": current_gap.total,
            "gap_after": best.estimated_gap_after.total,
            "reduction": best.gap_reduction,
            "concept_hint": concept_hint,
        })

        return best

    def _extract_types(self, program) -> set:
        """Extrai tipos de primitives de um programa."""
        types = set()
        if hasattr(program, 'instances'):
            for inst in program.instances:
                if hasattr(inst, 'primitive') and hasattr(inst.primitive, 'structure_type'):
                    types.add(inst.primitive.structure_type)
                elif hasattr(inst, 'name'):
                    types.add(inst.name)
        if hasattr(program, 'name'):
            types.add(program.name)
        return types

    def get_history(self) -> List[Dict]:
        return self._history[-50:]


# ─── GoalDrivenEngine — loop completo goal -> acao -> feedback ─

class GoalDrivenEngine:
    """Engine completa de agencia orientada a objetivos.

    Integra:
      - ProgramInductionEngine (descobre programas)
      - GoalComparator (mede gap)
      - ActionSelector (escolhe acao)

    Loop:
      goal → perceber estado atual → computar gap →
      induzir programas → simular reducao de gap →
      executar melhor acao → atualizar goal → repeat
    """

    def __init__(self, program_induction_engine=None,
                 composition_graph=None,
                 use_planner: bool = True,
                 cfe_bridge=None,
                 cfe_bridge_search=None,
                 hypothesis_engine=None):
        self.pie = program_induction_engine
        self.composition_graph = composition_graph
        self.comparator = GoalComparator()
        self.selector = ActionSelector()
        self.current_goal: Optional[GoalState] = None
        self.current_state: Any = None
        self.history: List[Dict] = []
        self.use_planner = use_planner
        self.cfe_bridge = cfe_bridge
        self.cfe_bridge_search = cfe_bridge_search
        self.hypothesis_engine = hypothesis_engine
        self.last_concept: Optional[str] = None
        self.last_hypothesis: Optional[Dict] = None
        if use_planner:
            from .hierarchical_planner import HierarchicalPlanner
            self.planner = HierarchicalPlanner(synthesizer=self.pie)

    def set_goal(self, goal: GoalState):
        """Define o goal atual."""
        self.current_goal = goal
        self.history.append({
            "event": "goal_set",
            "goal": goal.description,
            "time": time.time(),
        })

    def set_goal_from_target(self, target: Any, description: str = "",
                             priority: float = 0.5):
        """Define goal a partir de um estado alvo."""
        self.set_goal(GoalState(
            target=target,
            description=description or "reach target state",
            priority=priority,
        ))

    def perceive(self, state: Any):
        """Atualiza o estado atual percebido."""
        self.current_state = state

    def step(self) -> Optional[Action]:
        """Um ciclo completo: goal → gap → programa → acao.

        Returns: Action executada, ou None se goal ja atingido.
        """
        if self.current_goal is None or self.current_state is None:
            return None

        # 1. Computa gap
        gap = self.comparator.compute(self.current_state, self.current_goal)

        # 2. Goal ja atingido?
        if gap.is_zero:
            self.history.append({
                "event": "goal_achieved",
                "goal": self.current_goal.description,
                "gap": gap.to_dict(),
                "time": time.time(),
            })
            return None

        # 3. Se tem CFE, obtem concept hint para guiar a busca
        cfe_filter = None
        if self.cfe_bridge is not None:
            if isinstance(self.current_state, np.ndarray) and isinstance(self.current_goal.target, np.ndarray):
                cfe_filter = self.cfe_bridge.get_primitive_filter(
                    self.current_state, self.current_goal.target
                )
                if cfe_filter:
                    pass  # filter stored for use below

        # 4. Tenta planejamento hierarquico primeiro
        if self.use_planner and hasattr(self, 'planner'):
            if isinstance(self.current_state, np.ndarray) and isinstance(self.current_goal.target, np.ndarray):
                programs = self.planner.plan(self.current_state, self.current_goal.target)
                if programs:
                    state = self.current_state.copy()
                    for prog in programs:
                        state = prog.execute(state)
                    new_gap = self.comparator.compute(state, self.current_goal)
                    if new_gap.total < gap.total:
                        self.current_state = state
                        action_name = " -> ".join([self._extract_program_name(p) for p in programs])
                        best = Action(
                            program=programs,
                            estimated_gap_after=new_gap,
                            execution_cost=len(programs),
                            confidence=1.0 - new_gap.total,
                        )
                        self.history.append({
                            "event": "action_executed",
                            "goal": self.current_goal.description,
                            "action": action_name,
                            "gap_before": gap.total,
                            "gap_after": new_gap.total,
                            "time": time.time(),
                        })
                        return best

        # 5. Fallback: induz programas individuais
        # Se tem CFE, nao usa filter restritivo — usa boost conceitual no selector
        candidates = self._induce_programs(self.current_state, self.current_goal,
                                           primitive_filter=None)  # sem filtro restritivo

        actions = []
        for prog in candidates[:10]:
            try:
                result = prog.execute(self.current_state)
                new_gap = self.comparator.compute(result, self.current_goal)
                actions.append(Action(
                    program=prog,
                    estimated_gap_after=new_gap,
                    execution_cost=prog.n_steps,
                    confidence=prog.score,
                ))
            except Exception:
                pass

        # Extrai concept info do CFE para boost no selector
        concept_hint = None
        concept_prims = None
        if self.cfe_bridge is not None and actions:
            hint = self.cfe_bridge.get_concept_hint(self.current_state, self.current_goal.target)
            if hint:
                concept_hint = hint
                from .conceptual_discovery.cfe_bridge import CONCEPT_TO_PRIMITIVES as CTP
                concept_prims = CTP.get(hint, set())

        # 6. Seleciona melhor acao (max gap reduction + boost conceitual)
        best = self.selector.select(actions, self.current_goal, gap,
                                    concept_hint=concept_hint,
                                    concept_primitives=concept_prims)
        if best is None:
            self.history.append({
                "event": "no_action",
                "goal": self.current_goal.description,
                "gap": gap.to_dict(),
                "time": time.time(),
            })
            return None

        # 6. Executa
        try:
            new_state = best.program.execute(self.current_state)
            self.current_state = new_state
            best.executed = True
        except Exception:
            pass

        # Refina composicao no grafo se disponivel
        if self.composition_graph:
            types = self._extract_types(best.program)
            if types:
                self.composition_graph.record_program(
                    types, succeeded=gap.total - best.estimated_gap_after.total > 0.1
                )

        self.history.append({
            "event": "action_executed",
            "goal": self.current_goal.description,
            "action": self._extract_program_name(best.program),
            "gap_before": gap.total,
            "gap_after": best.estimated_gap_after.total,
            "time": time.time(),
        })

        return best

    def run(self, max_steps: int = 10) -> bool:
        """Roda o loop ate goal atingido ou max_steps.

        Returns: True se goal foi atingido.
        """
        for i in range(max_steps):
            action = self.step()
            if action is None:
                # Goal atingido ou sem acao viavel
                return self._is_goal_achieved()

            gap = self.comparator.compute(self.current_state, self.current_goal)
            if gap.is_zero:
                return True

        return self._is_goal_achieved()

    def _extract_types(self, program) -> list:
        """Extrai tipos de primitives de um programa (ou lista de programas)."""
        if isinstance(program, list):
            types = []
            for p in program:
                types.extend(self._extract_types(p))
            return types
        if hasattr(program, "instances"):
            return [inst.primitive.structure_type for inst in program.instances
                   ] if program.instances else []
        if hasattr(program, "name"):
            return [program.name]
        return []

    def _extract_program_name(self, program) -> str:
        """Extrai nome legivel de um programa."""
        if hasattr(program, "name") and program.name:
            return program.name[:80]
        if hasattr(program, "instances") and program.instances:
            names = []
            for inst in program.instances[:3]:
                if hasattr(inst, "name"):
                    names.append(inst.name[:20])
                elif hasattr(inst, "primitive") and hasattr(inst.primitive, "name"):
                    names.append(inst.primitive.name[:20])
            if names:
                return " -> ".join(names)
        return str(program)[:60]

    def _induce_programs(self, state: Any, goal: GoalState,
                         primitive_filter: set = None) -> list:
        """Induz programas para transformar estado atual em goal.

        Prioridade:
          1. HypothesisEngine (conceito → hipotese → guidance)
          2. CFEBridgeSearch (soft guidance + fallback)
          3. CFEBridge (hard filter)
          4. Synthesizer puro
        """
        # 1. HypothesisEngine: conceito → hipotese → guidance
        if self.hypothesis_engine is not None:
            try:
                observation = {"grid": state, "target": goal.target}
                if isinstance(state, np.ndarray):
                    obs_shape = state.shape
                    if isinstance(goal.target, np.ndarray):
                        observation["shape_preserved"] = (state.shape == goal.target.shape)

                hypotheses = self.hypothesis_engine.generate_hypotheses(
                    observation, domain="arc", modality="grid"
                )
                best = self.hypothesis_engine.select_best(hypotheses)
                if best:
                    guidance = self.hypothesis_engine.hypothesis_to_guidance(best)
                    self.last_concept = best.concept
                    self.last_hypothesis = guidance
                    weights = guidance.get("primitive_weights", {})
                    if weights and self.pie is not None:
                        return self.pie.synthesize(
                            state, goal.target, primitive_weights=weights
                        )
            except Exception:
                pass

        # 2. CFEBridgeSearch
        if self.cfe_bridge_search is not None:
            try:
                result = self.cfe_bridge_search.search(state, goal.target)
                return result
            except Exception:
                return []

        # 3. Synthesizer com/sem CFE filter
        if self.pie is not None and hasattr(self.pie, "synthesize"):
            try:
                if primitive_filter is None and self.cfe_bridge is not None:
                    if isinstance(state, np.ndarray) and isinstance(goal.target, np.ndarray):
                        primitive_filter = self.cfe_bridge.get_primitive_filter(
                            state, goal.target
                        )

                kwargs = {}
                if primitive_filter is not None:
                    kwargs["primitive_filter"] = primitive_filter

                return self.pie.synthesize(state, goal.target, **kwargs)
            except Exception:
                pass
        return []

    def _is_goal_achieved(self) -> bool:
        if self.current_goal is None or self.current_state is None:
            return False
        gap = self.comparator.compute(self.current_state, self.current_goal)
        return gap.is_zero

    def get_stats(self) -> dict:
        n_actions = sum(1 for h in self.history if h.get("event") == "action_executed")
        return {
            "has_goal": self.current_goal is not None,
            "goal_achieved": self._is_goal_achieved() if self.current_goal else False,
            "n_actions": n_actions,
            "n_cycles": len(self.history),
        }

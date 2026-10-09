"""
SimulationEngine — Motor de Simulacao Unificado

Quatro modos, um mecanismo:
  - EXPECT: prever o futuro mais provavel
  - COUNTERFACTUAL: "e se..." cenarios alternativos
  - PLAN: sequencia de acoes para atingir um objetivo
  - IMAGINE: geracao criativa de possibilidades

Cada modo gera BRANCHES (futuros possiveis) que competem
via ModelCompetition. A realidade observada reforca ou
enfraquece cada branch.
"""
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class SimulationMode(Enum):
    EXPECT = "expect"
    COUNTERFACTUAL = "counterfactual"
    PLAN = "plan"
    IMAGINE = "imagine"


@dataclass
class Branch:
    """Um futuro possivel — gerado pela simulacao."""
    id: str
    description: str = ""
    probability: float = 0.0  # 0..1
    utility: float = 0.0  # quao desejavel
    cost: float = 0.0  # custo computacional
    steps: List[str] = field(default_factory=list)  # acoes previstas
    predicted_state: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_updated: float = 0.0

    def update(self, actual_state: Dict[str, float]) -> float:
        """Atualiza probablidade baseada na realidade observada."""
        error = 0.0
        n = 0
        for key, predicted in self.predicted_state.items():
            actual = actual_state.get(key)
            if actual is not None:
                error += abs(predicted - actual)
                n += 1

        avg_error = error / max(1, n)
        # Quanto menor o erro, mais reforcada a branch
        accuracy = 1.0 - min(1.0, avg_error)
        self.probability = self.probability * 0.7 + accuracy * 0.3
        self.confidence = self.confidence * 0.8 + accuracy * 0.2
        self.last_updated = time.time()
        return avg_error


class SimulationEngine:
    """Motor de simulacao unificado.

    Gera branches (futuros possiveis), atualiza com realidade,
    e integra com ModelCompetition + CGE.
    """

    def __init__(self, model_ecosystem=None, cge=None):
        self.branches: List[Branch] = []
        self.history: List[Dict] = []
        self.model_ecosystem = model_ecosystem
        self.cge = cge
        self.max_branches: int = 20

        # Modelos de transicao de estado (simples)
        self._transition_models: Dict[str, Callable] = {}

    def add_transition_model(self, name: str, fn: Callable):
        """Registra um modelo de transicao de estado."""
        self._transition_models[name] = fn

    def simulate(self, current_state: Dict[str, float],
                 mode: SimulationMode = SimulationMode.EXPECT,
                 action: Optional[str] = None,
                 goal: Optional[Dict[str, float]] = None,
                 n_branches: int = 5) -> List[Branch]:
        """Gera multiplos futuros possiveis e armazena em self.branches."""
        if mode == SimulationMode.EXPECT:
            branches = self._expect(current_state, n_branches)
        elif mode == SimulationMode.COUNTERFACTUAL:
            branches = self._counterfactual(current_state, action, n_branches)
        elif mode == SimulationMode.PLAN:
            branches = self._plan(current_state, goal, n_branches)
        elif mode == SimulationMode.IMAGINE:
            branches = self._imagine(current_state, n_branches)

        self.branches.extend(branches)
        # Limitar numero de branches
        if len(self.branches) > self.max_branches:
            self.branches.sort(key=lambda b: -b.probability)
            self.branches = self.branches[:self.max_branches]

        self.history.append({
            "time": time.time(),
            "mode": mode.value,
            "n_new": len(branches),
            "n_total": len(self.branches),
        })
        return branches

    def _expect(self, state: Dict[str, float], n: int = 5) -> List[Branch]:
        """PREVER: futuro mais provavel."""
        branches = []

        # Branch principal: usar modelos de transicao
        predicted = dict(state)
        for name, model in self._transition_models.items():
            try:
                pred = model(state)
                predicted.update(pred)
            except Exception:
                pass

        main_branch = Branch(
            id=f"expect_main_{int(time.time())}",
            description="Futuro esperado (modelos de transicao)",
            probability=0.5,
            utility=0.5,
            predicted_state=predicted,
            confidence=0.6,
        )
        branches.append(main_branch)

        # Branches alternativas com pequenas variacoes
        for i in range(n - 1):
            alt_state = dict(state)
            # Adicionar ruido gaussiano
            for key in alt_state:
                alt_state[key] += random.uniform(-0.2, 0.2)
                alt_state[key] = max(0.0, min(1.0, alt_state[key]))

            alt_branch = Branch(
                id=f"expect_alt_{i}_{int(time.time())}",
                description=f"Variacao {i+1}",
                probability=0.5 / max(1, n),
                predicted_state=alt_state,
                confidence=0.3,
            )
            branches.append(alt_branch)

        return branches

    def _counterfactual(self, state: Dict[str, float],
                        action: Optional[str] = None, n: int = 5) -> List[Branch]:
        """E SE: cenarios alternativos com acao vs sem acao."""
        branches = []

        # Branch: sem acao (linha base)
        base_branch = Branch(
            id=f"cf_base_{int(time.time())}",
            description=f"Sem acao ('{action or 'nenhuma'}')",
            probability=0.3,
            cost=0.1,
            predicted_state=dict(state),
        )
        branches.append(base_branch)

        # Branch: com acao
        if action:
            action_state = dict(state)
            action_state[f"action_{action}"] = 1.0
            action_state["uncertainty"] = min(1.0, state.get("uncertainty", 0.5) + 0.1)
            action_branch = Branch(
                id=f"cf_action_{int(time.time())}",
                description=f"Se fizer '{action}'",
                probability=0.3,
                cost=0.3,
                predicted_state=action_state,
            )
            branches.append(action_branch)

        # Branches: e se X mudar?
        for i, key in enumerate(list(state.keys())[:n - 2]):
            cf_state = dict(state)
            val = state.get(key, 0.5)
            cf_state[key] = 1.0 - val  # inverte
            cf_branch = Branch(
                id=f"cf_{key}_{int(time.time())}",
                description=f"E se '{key}' fosse {1.0 - val:.1f}?",
                probability=0.1,
                cost=0.2,
                predicted_state=cf_state,
            )
            branches.append(cf_branch)

        return branches

    def _plan(self, state: Dict[str, float],
              goal: Optional[Dict[str, float]] = None, n: int = 5) -> List[Branch]:
        """PLANEJAR: sequencia de acoes para atingir objetivo."""
        if goal is None:
            goal = {"goal_achieved": 1.0}

        branches = []
        # Gerar N planos candidatos
        for i in range(n):
            plan_steps = self._generate_plan(state, goal)
            # Simular estado apos o plano
            plan_state = dict(state)
            for step in plan_steps[:5]:  # limitar a 5 passos
                plan_state[f"step_{step}"] = plan_state.get(f"step_{step}", 0) + 1

            # Calcular utilidade do plano
            utility = 0.0
            for k, v in goal.items():
                actual = plan_state.get(k, 0)
                utility += 1.0 - min(1.0, abs(actual - v))

            plan_branch = Branch(
                id=f"plan_{i}_{int(time.time())}",
                description=" -> ".join(plan_steps[:4]) if plan_steps else "plano vazio",
                probability=0.5 / max(1, n),
                utility=utility / max(1, len(goal)),
                cost=len(plan_steps) * 0.1,
                steps=plan_steps,
                predicted_state=plan_state,
            )
            branches.append(plan_branch)

        return branches

    def _imagine(self, state: Dict[str, float], n: int = 5) -> List[Branch]:
        """IMAGINAR: geracao criativa de possibilidades."""
        branches = []

        # Combinacoes aleatorias de dimensoes do estado
        keys = list(state.keys())
        for i in range(n):
            imagined = dict(state)
            # Pegar 2-3 dimensoes e combinar aleatoriamente
            dims = random.sample(keys, min(3, len(keys)))
            for d in dims:
                imagined[d] = random.random()

            img_branch = Branch(
                id=f"imagine_{i}_{int(time.time())}",
                description=f"Cenario imaginado {i+1}",
                probability=0.1,
                utility=random.random(),
                cost=0.3,
                predicted_state=imagined,
                confidence=0.2,  # baixa confianca (imaginacao)
            )
            branches.append(img_branch)

        return branches

    def _generate_plan(self, state: Dict[str, float],
                       goal: Dict[str, float]) -> List[str]:
        """Gera um plano (sequencia de acoes) para atingir objetivo."""
        # Usar operadores disponiveis se CGE presente
        available_ops = ["observe", "predict", "test", "explore",
                         "focus", "compare", "generalize"]

        # Planos sao sequencias aleatorias
        plan_length = random.randint(3, 7)
        plan = random.choices(available_ops, k=plan_length)
        return plan

    def observe_reality(self, actual_state: Dict[str, float]) -> Dict[str, float]:
        """Observa a realidade e atualiza todas as branches.

        Retorna: {branch_id: error} para cada branch atualizada.
        """
        results = {}
        for branch in self.branches:
            error = branch.update(actual_state)
            results[branch.id] = error

        # Remover branches com probabilidade muito baixa
        self.branches = [b for b in self.branches if b.probability > 0.01]

        # Registrar no historico
        self.history.append({
            "time": time.time(),
            "n_branches": len(self.branches),
            "best_branch": max(self.branches, key=lambda b: b.probability).id
                           if self.branches else None,
            "avg_error": sum(results.values()) / max(1, len(results)),
        })

        return results

    def get_best_branch(self, mode: str = "probability") -> Optional[Branch]:
        """Melhor branch por probabilidade ou utilidade."""
        if not self.branches:
            return None
        if mode == "probability":
            return max(self.branches, key=lambda b: b.probability)
        else:
            return max(self.branches, key=lambda b: b.utility)

    def clear(self):
        """Limpa branches (novo ciclo)."""
        self.branches = []

    def get_state(self) -> Dict:
        return {
            "n_branches": len(self.branches),
            "best": self.get_best_branch().id if self.branches else None,
            "best_prob": round(self.get_best_branch().probability, 2) if self.branches else 0,
            "total_simulations": len(self.history),
            "transition_models": list(self._transition_models.keys()),
        }

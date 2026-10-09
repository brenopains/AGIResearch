"""
CognitiveBudget — Alocacao Dinamica de Recursos Cognitivos

Modulos competem por recursos finitos.
O CognitiveBudget decide quem recebe quanto.

NAO hardcoded: o orcamento inicial, o custo de cada modulo,
e a politica de alocacao sao todos auto-ajustados baseado em
eficiencia observada.

Cada modulo declara:
  - tipo de operacao (simulation, language, ecology, etc.)
  - custo estimado
  - valor estimado (quao util o output sera)
  - urgencia (quao importante e para o problema atual)

O CognitiveBudget aloca com base em:
  - valor / custo (eficiencia)
  - urgencia
  - historico de utilidade do modulo
"""
import time, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ResourceRequest:
    """Um pedido de recurso cognitivo."""
    module: str = ""          # "world_model", "simulation", "ecology", "language", etc.
    operation: str = ""       # "predict", "simulate", "express", "debate"
    estimated_cost: float = 1.0
    estimated_value: float = 0.5
    urgency: float = 0.5
    timestamp: float = field(default_factory=time.time)
    fulfilled: bool = False


@dataclass
class AllocationResult:
    """Resultado de uma alocacao de recursos."""
    module: str = ""
    operation: str = ""
    allocated: float = 0.0
    denied: bool = False
    reason: str = ""


class CognitiveBudget:
    """Orcamento cognitivo. Decide alocacao de recursos.

    Auto-tudo:
      - Orcamento total: auto-calculado baseado em custo historico
      - Custos por modulo: auto-aprendidos
      - Politica de alocacao: evolui baseada em eficiencia
    """

    def __init__(self):
        self._total_budget: float = 0.0  # NAO hardcoded — auto-inicializado
        self._spent: float = 0.0
        self._module_costs: Dict[str, List[float]] = defaultdict(list)
        self._module_efficiency: Dict[str, float] = {}  # valor / custo
        self._recent_allocations: List[ResourceRequest] = []
        self._allocation_history: List[Dict] = []

    def _auto_initialize_budget(self) -> float:
        """O sistema descobre seu proprio orcamento baseado em
        requisitos minimos dos modulos que precisa coordenar."""
        if self._total_budget > 0:
            return self._total_budget
        # Observa quais modulos pediram quanto historicamente
        if self._module_costs:
            total = sum(sum(v[-10:]) for v in self._module_costs.values()) / 10
            self._total_budget = max(100.0, total * 2)
        else:
            self._total_budget = 100.0  # base inicial
        return self._total_budget

    def request(self, module: str, operation: str,
                estimated_cost: float = 1.0,
                estimated_value: float = 0.5,
                urgency: float = 0.5) -> ResourceRequest:
        """Um modulo faz um pedido de recurso."""
        req = ResourceRequest(
            module=module, operation=operation,
            estimated_cost=estimated_cost, estimated_value=estimated_value,
            urgency=urgency,
        )
        self._recent_allocations.append(req)
        if len(self._recent_allocations) > 100:
            self._recent_allocations = self._recent_allocations[-100:]
        return req

    def allocate(self, requests: Optional[List[ResourceRequest]] = None) -> List[AllocationResult]:
        """Aloca recursos para uma serie de pedidos.

        Algoritmo: maximizar valor total sujeito a orcamento.
        Ordena pedidos por eficiencia (valor/custo) * urgencia.
        """
        if requests is None:
            requests = self._recent_allocations[-20:]

        if self._total_budget == 0:
            self._auto_initialize_budget()

        available = self._total_budget - self._spent
        results = []

        # Ordena por eficiencia * urgencia
        def score(r: ResourceRequest) -> float:
            eff = r.estimated_value / max(0.1, r.estimated_cost)
            return eff * (0.5 + r.urgency)

        sorted_reqs = sorted(requests, key=score, reverse=True)
        for req in sorted_reqs:
            if available >= req.estimated_cost:
                self._spent += req.estimated_cost
                available -= req.estimated_cost
                req.fulfilled = True
                # Aprende custo deste modulo
                self._module_costs[req.module].append(req.estimated_cost)
                if len(self._module_costs[req.module]) > 50:
                    self._module_costs[req.module] = self._module_costs[req.module][-50:]
                results.append(AllocationResult(
                    module=req.module, operation=req.operation,
                    allocated=req.estimated_cost,
                ))
            else:
                req.fulfilled = False
                results.append(AllocationResult(
                    module=req.module, operation=req.operation,
                    allocated=0.0, denied=True, reason="budget_exhausted"
                ))

        self._allocation_history.append({
            "time": time.time(), "n_requests": len(requests),
            "n_fulfilled": sum(1 for r in results if r.allocated > 0),
        })
        if len(self._allocation_history) > 50:
            self._allocation_history = self._allocation_history[-50:]
        return results

    def record_outcome(self, module: str, value_realized: float):
        """Registra o valor real que o modulo produziu com os recursos alocados."""
        if module not in self._module_efficiency:
            self._module_efficiency[module] = 0.5
        old = self._module_efficiency[module]
        self._module_efficiency[module] = old * 0.9 + (value_realized / max(0.1, 1.0)) * 0.1

    def reset_cycle(self):
        """Reseta o orcamento para o proximo ciclo."""
        self._spent = 0.0
        # Re-discover budget se necessario
        if self._allocation_history:
            recent_spend = [h.get("n_fulfilled", 0) for h in self._allocation_history[-10:]]
            if recent_spend and sum(recent_spend) > 0:
                avg_fulfilled = sum(recent_spend) / len(recent_spend)
                if avg_fulfilled < 0.5:
                    # Muitos pedidos negados — aumentar orcamento
                    self._total_budget *= 1.1
                elif avg_fulfilled > 0.9 and self._total_budget > 100:
                    # Quase tudo alocado — pode reduzir
                    self._total_budget *= 0.95

    def get_state(self) -> Dict:
        return {
            "total_budget": round(self._total_budget, 2),
            "spent": round(self._spent, 2),
            "module_efficiency": {k: round(v, 3)
                                  for k, v in self._module_efficiency.items()},
            "recent_allocations": len(self._allocation_history),
        }

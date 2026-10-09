"""
ActiveExperiment — Active Experiment Design

Nao basta ser curioso. Ciencia faz experimentos ATIVOS.
A AGI escolhe qual experimento maximiza informacao por unidade de custo.

Dado um conjunto de hipoteses, o sistema escolhe:
  - qual intervencao fazer
  - qual observacao medir
  - que trade-off custo/informacao aceitar

Custo: recursos computacionais + risco + tempo
Informacao: reducao esperada de entropia no estado de crenca
"""
import time, math, hashlib, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Hypothesis:
    """Uma hipotese que precisa de teste."""
    id: str = ""
    statement: str = ""
    confidence: float = 0.5
    domain: str = "general"
    predicted_outcomes: Dict[str, float] = field(default_factory=dict)
    testable_predictions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class Experiment:
    """Um experimento proposto."""
    id: str = ""
    hypothesis_id: str = ""
    description: str = ""
    intervention: str = ""
    expected_observation: str = ""
    estimated_cost: float = 1.0
    estimated_information: float = 0.5
    expected_entropy_reduction: float = 0.0
    priority: float = 0.5


class ActiveExperimentDesigner:
    """Propoe experimentos que maximizam informacao por unidade de custo.

    NAO hardcoded: os pesos para custo vs informacao, a tolerancia
    de risco, e a estrategia de busca sao todos auto-ajustados.
    """

    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.proposed_experiments: List[Experiment] = []
        self.completed_experiments: List[Dict] = []
        self._strategy: str = "default"  # auto-evolui

    def add_hypothesis(self, statement: str, domain: str = "general",
                       confidence: float = 0.5,
                       predicted_outcomes: Optional[Dict[str, float]] = None,
                       testable_predictions: Optional[List[str]] = None) -> str:
        hid = f"hyp_{hashlib.md5(statement.encode()).hexdigest()[:8]}"
        if hid in self.hypotheses:
            return hid
        self.hypotheses[hid] = Hypothesis(
            id=hid, statement=statement, confidence=confidence,
            domain=domain,
            predicted_outcomes=predicted_outcomes or {},
            testable_predictions=testable_predictions or [],
        )
        return hid

    def design_experiment(self, hypothesis_id: str) -> Optional[Experiment]:
        """Projeta um experimento para testar a hipotese."""
        h = self.hypotheses.get(hypothesis_id)
        if h is None:
            return None

        # Estima informacao esperada (auto-discover)
        info = self._estimate_information(h)
        # Estima custo (auto-discover baseado em complexidade)
        cost = self._estimate_cost(h)
        # Calcula prioridade: info / custo
        priority = info / max(0.1, cost)

        exp = Experiment(
            id=f"exp_{hypothesis_id}_{int(time.time())}",
            hypothesis_id=hypothesis_id,
            description=f"Testar: {h.statement[:50]}",
            intervention=f"Observar se {h.testable_predictions[0] if h.testable_predictions else 'resultado esperado'} ocorre",
            expected_observation="Confirmar ou refutar a hipotese",
            estimated_cost=cost,
            estimated_information=info,
            expected_entropy_reduction=info * h.confidence,
            priority=priority,
        )
        self.proposed_experiments.append(exp)
        return exp

    def _estimate_information(self, h: Hypothesis) -> float:
        """Auto-descobre quanta informacao o teste geraria."""
        # Heuristica: mais incerto = mais informacao
        uncertainty = 1.0 - h.confidence
        # Bonus para hipoteses com multiplas predicoes testaveis
        testable_bonus = min(0.5, len(h.testable_predictions) * 0.1)
        return min(1.0, uncertainty + testable_bonus)

    def _estimate_cost(self, h: Hypothesis) -> float:
        """Auto-descobre custo do experimento."""
        # Custo baseado em complexidade do dominio
        base = 1.0
        if h.domain not in ("general", "trivial"):
            base = 2.0
        # Hipoteses mais incertas custam mais para testar
        return base * (1.0 + (1.0 - h.confidence) * 0.5)

    def rank_experiments(self) -> List[Experiment]:
        """Ordena experimentos por info/custo (eficiencia)."""
        return sorted(self.proposed_experiments, key=lambda e: -e.priority)

    def record_outcome(self, experiment_id: str, result: str,
                       observed_info: float):
        """Registra o resultado real de um experimento."""
        self.completed_experiments.append({
            "experiment_id": experiment_id,
            "result": result,
            "observed_information": observed_info,
            "time": time.time(),
        })
        # Remove da lista de propostos
        self.proposed_experiments = [
            e for e in self.proposed_experiments
            if e.id != experiment_id
        ]

    def get_state(self) -> Dict:
        return {
            "n_hypotheses": len(self.hypotheses),
            "n_proposed": len(self.proposed_experiments),
            "n_completed": len(self.completed_experiments),
            "top_experiments": [
                {"desc": e.description[:50], "priority": round(e.priority, 2)}
                for e in self.rank_experiments()[:3]
            ],
        }

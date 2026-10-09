"""
ErrorSignal — Sistema de Erro Multi-Dimensional

O erro nao e mais um numero. E uma estrutura contextual que carrega:
  - prediction_error: "o que esperava vs o que aconteceu" (dopamina)
  - novelty: "quao novo e este padrao?" (noradrenalina)
  - salience: "quao importante?" (atencao)
  - uncertainty: "quanta certeza tenho?" (acetilcolina)
  - violation: "isto quebra meu modelo atual?"
  - causal_attribution: "o que CAUSOU este erro?"

Cada cenario/hipotese pode ter MULTIPLOS erros ativos simultaneamente.
O Maestro usa estes sinais para decidir COMO pensar.
"""
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ErrorSignal:
    """Um sinal de erro multi-dimensional EXTENSIVEL.

    As dimensoes base sao:
      prediction_error, novelty, salience, uncertainty, violation

    Novas dimensoes podem ser adicionadas via `extra_dimensions`.
    Nao dependa de atributos fixos — use get() para acesso generico.
    """
    scenario_id: str  # qual cenario/hipotese gerou este erro
    prediction_error: float = 0.0  # 0..1, quao errada foi a predicao
    novelty: float = 0.0  # 0..1, quao novo e este padrao
    salience: float = 0.0  # 0..1, quao importante/urgente
    uncertainty: float = 0.5  # 0..1, quanta incerteza (0=certeza total)
    violation: float = 0.0  # 0..1, esta quebra regras conhecidas?
    causal_attribution: List[str] = field(default_factory=list)
    confidence: float = 1.0  # confianca neste sinal de erro
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    # DIMENSOES EXTENSIVEIS: qualquer codigo pode adicionar
    # Exemplo: {"coherence": 0.8, "surprise_decay": 0.3, "goal_distance": 0.6}
    extra_dimensions: Dict[str, float] = field(default_factory=dict)

    def get(self, dim: str, default: float = 0.0) -> float:
        """Acesso generico a qualquer dimensao (base ou extra)."""
        if hasattr(self, dim):
            return getattr(self, dim, default)
        return self.extra_dimensions.get(dim, default)

    def set_dim(self, dim: str, value: float):
        """Define qualquer dimensao (base ou extra)."""
        if hasattr(self, dim):
            setattr(self, dim, max(0.0, min(1.0, value)))
        else:
            self.extra_dimensions[dim] = max(0.0, min(1.0, value))

    @property
    def magnitude(self) -> float:
        """Magnitude agregada do erro (usa dimensoes disponiveis)."""
        dims = ["prediction_error", "novelty", "salience", "violation"]
        weights = [0.4, 0.2, 0.2, 0.2]
        total = sum(self.get(d) * w for d, w in zip(dims, weights))
        # Incluir extras com peso reduzido
        for k, v in self.extra_dimensions.items():
            total += v * 0.1
        return min(1.0, total)

    @property
    def drives_exploration(self) -> bool:
        return self.novelty > 0.3 and self.uncertainty > 0.4

    @property
    def drives_exploitation(self) -> bool:
        return self.prediction_error > 0.3 and self.uncertainty < 0.3

    @property
    def drives_reflection(self) -> bool:
        return self.violation > 0.5 or (self.prediction_error > 0.5 and self.novelty < 0.3)

    def dampen(self, factor: float = 0.9):
        for dim in ["prediction_error", "novelty", "salience", "violation"]:
            setattr(self, dim, getattr(self, dim, 0.0) * factor)
        for k in self.extra_dimensions:
            self.extra_dimensions[k] *= factor
        self.confidence = min(1.0, self.confidence + (1 - factor) * 0.5)


class MultiErrorTracker:
    """Mantem N erros ativos simultaneamente, um por cenario/hipotese.

    O Maestro pode:
      - criar erros para novos cenarios
      - descartar erros de cenarios resolvidos
      - consultar o erro mais saliente para decidir acao
      - priorizar erros por contexto
    """

    def __init__(self, max_errors: int = 16):
        self.errors: Dict[str, ErrorSignal] = {}
        self.history: List[ErrorSignal] = []
        self.max_errors = max_errors

    def create_error(self, scenario_id: str, prediction_error: float = 0.0,
                     novelty: float = 0.0, salience: float = 0.0,
                     uncertainty: float = 0.5, violation: float = 0.0,
                     causal_attribution: Optional[List[str]] = None,
                     context: Optional[Dict] = None,
                     extra: Optional[Dict[str, float]] = None) -> ErrorSignal:
        """Cria um novo sinal de erro para um cenario.

        Args:
            extra: dimensoes extras. Ex: {"coherence": 0.8, "goal_distance": 0.3}
        """
        if len(self.errors) >= self.max_errors:
            oldest_id = min(self.errors, key=lambda k: (
                self.errors[k].salience + self.errors[k].prediction_error))
            self.discard_error(oldest_id)

        signal = ErrorSignal(
            scenario_id=scenario_id,
            prediction_error=prediction_error,
            novelty=novelty,
            salience=salience,
            uncertainty=uncertainty,
            violation=violation,
            causal_attribution=causal_attribution or [],
            context=context or {},
            extra_dimensions=extra or {},
        )
        self.errors[scenario_id] = signal
        return signal

    def update_error(self, scenario_id: str, **kwargs):
        """Atualiza dimensoes de um erro existente.

        Suporta dimensoes base e extras (prefixo 'extra__')."""
        if scenario_id in self.errors:
            signal = self.errors[scenario_id]
            for key, val in kwargs.items():
                if key.startswith("extra__"):
                    signal.set_dim(key[7:], val)
                elif hasattr(signal, key):
                    setattr(signal, key, val)
            signal.timestamp = time.time()

    def discard_error(self, scenario_id: str):
        """Remove um erro (cenario resolvido ou hipotese refutada)."""
        if scenario_id in self.errors:
            self.history.append(self.errors[scenario_id])
            del self.errors[scenario_id]

    def get_most_pressing(self) -> Optional[ErrorSignal]:
        """Retorna o erro mais urgente (salience * prediction_error)."""
        if not self.errors:
            return None
        return max(self.errors.values(),
                   key=lambda e: e.salience * (0.5 + e.prediction_error) *
                   (2.0 - e.confidence))

    def get_by_scenario(self, scenario_id: str) -> Optional[ErrorSignal]:
        return self.errors.get(scenario_id)

    def get_state(self) -> Dict:
        """Estado serializavel para o Maestro."""
        return {
            "active_errors": len(self.errors),
            "most_pressing": self.get_most_pressing().scenario_id
            if self.get_most_pressing() else None,
            "avg_prediction_error": sum(
                e.prediction_error for e in self.errors.values()
            ) / max(1, len(self.errors)),
            "avg_novelty": sum(
                e.novelty for e in self.errors.values()
            ) / max(1, len(self.errors)),
            "avg_uncertainty": sum(
                e.uncertainty for e in self.errors.values()
            ) / max(1, len(self.errors)),
            "drives_exploration": any(e.drives_exploration for e in self.errors.values()),
            "drives_exploitation": any(e.drives_exploitation for e in self.errors.values()),
            "drives_reflection": any(e.drives_reflection for e in self.errors.values()),
        }

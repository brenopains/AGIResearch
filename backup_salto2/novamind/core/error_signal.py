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
    """Um sinal de erro multi-dimensional."""
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

    @property
    def magnitude(self) -> float:
        """Magnitude agregada do erro (para compatibilidade)."""
        return (self.prediction_error * 0.4 +
                self.novelty * 0.2 +
                self.salience * 0.2 +
                self.violation * 0.2)

    @property
    def drives_exploration(self) -> bool:
        """Este erro incentiva exploracao?"""
        return self.novelty > 0.3 and self.uncertainty > 0.4

    @property
    def drives_exploitation(self) -> bool:
        """Este erro incentiva uso de conhecimento consolidado?"""
        return self.prediction_error > 0.3 and self.uncertainty < 0.3

    @property
    def drives_reflection(self) -> bool:
        """Este erro sugere que o modelo atual esta errado?"""
        return self.violation > 0.5 or (self.prediction_error > 0.5 and self.novelty < 0.3)

    def dampen(self, factor: float = 0.9):
        """Atenua todas as dimensoes (usado quando erro persiste sem mudanca)."""
        self.prediction_error *= factor
        self.novelty *= factor
        self.salience *= factor
        self.violation *= factor
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
                     context: Optional[Dict] = None) -> ErrorSignal:
        """Cria um novo sinal de erro para um cenario."""
        if len(self.errors) >= self.max_errors:
            # Descartar o erro menos saliente
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
            context=context or {}
        )
        self.errors[scenario_id] = signal
        return signal

    def update_error(self, scenario_id: str, **kwargs):
        """Atualiza dimensoes de um erro existente."""
        if scenario_id in self.errors:
            for key, val in kwargs.items():
                if hasattr(self.errors[scenario_id], key):
                    setattr(self.errors[scenario_id], key, val)
            self.errors[scenario_id].timestamp = time.time()

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

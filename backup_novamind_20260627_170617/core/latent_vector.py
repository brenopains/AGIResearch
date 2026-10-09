"""
LatentVector — Variaveis Latentes Descobertas pelo Sistema

Cada objeto no mundo tem:
  - vetor observavel (o que e visto)
  - vetor latente (o que e inferido)
  - historico de estados
  - predicoes futuras

O vetor latente NAO significa temperatura, pressao, desgaste.
Ele significa QUALQUER COISA — o sistema descobre.

A descoberta funciona assim:
  1. Sistema faz predicao baseada no vetor observavel
  2. Erro de predicao indica que falta informacao
  3. Sistema CRIUMA uma dimensao latente para explicar o erro
  4. Ajusta a dimensao latente para minimizar erro futuro
  5. Se a dimensao explica consistentemente o erro, vira conceito
"""
import torch
import math
import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LatentState:
    """Estado latente inferido para um objeto."""
    object_id: str
    dims: int = 8  # numero de dimensoes latentes
    vector: Optional[torch.Tensor] = None
    observable_signature: Optional[torch.Tensor] = None
    prediction_errors: List[float] = field(default_factory=list)
    discovered_dims: List[str] = field(default_factory=list)  # nomes opcionais
    created_at: float = field(default_factory=time.time)
    last_updated: float = 0.0

    def __post_init__(self):
        if self.vector is None:
            self.vector = torch.zeros(self.dims)

    def update(self, prediction_error: float,
               observed_features: Optional[torch.Tensor] = None,
               lr: float = 0.05):
        """Ajusta vetor latente para minimizar erro de predicao."""
        self.prediction_errors.append(prediction_error)
        self.last_updated = time.time()

        # Se o erro e alto, ajustar o vetor latente
        if prediction_error > 0.3 and observed_features is not None:
            # Projetar erro no espaco latente
            error_signal = torch.ones(self.dims) * prediction_error
            # Ruido gaussiano para exploracao
            noise = torch.randn(self.dims) * 0.1
            self.vector = self.vector + lr * (error_signal + noise)
            # Normalizar
            self.vector = torch.tanh(self.vector)

    def get_dim_significance(self, dim_idx: int) -> float:
        """Quao significativa e uma dimensao (0..1)."""
        if dim_idx >= len(self.vector) or len(self.prediction_errors) < 5:
            return 0.0
        # Uma dimensao e significativa se seu valor muda sistematicamente
        # com o erro de predicao
        return min(1.0, abs(self.vector[dim_idx].item()))

    def discover_dim_name(self, cge=None) -> Optional[str]:
        """Tenta dar nome a uma dimensao via CGE."""
        if cge is None:
            return None
        unnamed = [(i, self.get_dim_significance(i))
                  for i in range(len(self.vector))
                  if i >= len(self.discovered_dims)]
        if not unnamed:
            return None
        dim_idx, significance = max(unnamed, key=lambda x: x[1])
        if significance < 0.3:
            return None
        # Procurar no CGE um conceito similar ao padrao desta dimensao
        name = f"latent_dim_{dim_idx}"
        self.discovered_dims.append(name)
        return name


class LatentSpace:
    """Espaco de variaveis latentes descobertas pelo sistema.

    Mantem um conjunto de estados latentes para objetos no mundo,
    e descobre novas dimensoes conforme necessario.
    """

    def __init__(self, n_dims: int = 8, cge=None):
        self.states: Dict[str, LatentState] = {}
        self.n_dims = n_dims
        self.cge = cge
        self.global_errors: List[float] = []
        self.discovery_history: List[Dict] = []

    def get_or_create(self, object_id: str,
                      observable_vec: Optional[torch.Tensor] = None) -> LatentState:
        """Obtem ou cria estado latente para um objeto."""
        if object_id not in self.states:
            self.states[object_id] = LatentState(
                object_id=object_id, dims=self.n_dims,
                observable_signature=observable_vec
            )
        return self.states[object_id]

    def predict(self, object_id: str,
                context: Dict[str, float]) -> Dict[str, float]:
        """Faz predicao usando vetor latente + observavel."""
        state = self.states.get(object_id)
        if state is None:
            return {}

        # Predicao simples: media movel + influencia latente
        latent_influence = state.vector.mean().item() * 0.3
        base_pred = sum(context.values()) / max(1, len(context))
        return {"predicted_value": base_pred + latent_influence}

    def observe_and_update(self, object_id: str, observation: Dict,
                           observable_vec: Optional[torch.Tensor] = None):
        """Observa um objeto, prediz, e ajusta latente com erro."""
        state = self.get_or_create(object_id, observable_vec)
        prediction = self.predict(object_id, observation)

        # Calcular erro
        total_error = 0.0
        n = 0
        for key, actual in observation.items():
            predicted = prediction.get(key, 0.5)
            total_error += abs(actual - predicted)
            n += 1

        avg_error = total_error / max(1, n)
        self.global_errors.append(avg_error)

        # Atualizar estado latente
        state.update(avg_error, observable_vec)

        # Se erro persistentemente alto, tentar descobrir nova dimensao
        if len(self.global_errors) >= 10 and self._can_discover(object_id):
            recent = self.global_errors[-10:]
            if sum(recent) / len(recent) > 0.5:
                self._discover_new_dimension(object_id)

    def _can_discover(self, object_id: str) -> bool:
        """Verifica se pode descobrir nova dimensao (cooldown)."""
        if not self.discovery_history:
            return True
        last = self.discovery_history[-1]
        if last.get("object") == object_id:
            elapsed = time.time() - last.get("time", 0)
            if elapsed < 3.0:  # cooldown de 3s entre descobertas
                return False
        return True

    def _discover_new_dimension(self, object_id: str):
        """Descobre uma nova dimensao latente para explicar erros."""
        state = self.states.get(object_id)
        if state is None:
            return

        current_dims = len(state.vector)
        if current_dims >= 16:
            return  # limite maximo

        novo_idx = current_dims  # indice da nova dimensao

        # Estender vetor latente com novo valor
        new_val = torch.randn(1) * 0.5
        state.vector = torch.cat([state.vector, new_val])
        state.dims = len(state.vector)

        # Tentar nomear via CGE
        name = state.discover_dim_name(self.cge)

        self.discovery_history.append({
            "time": time.time(),
            "object": object_id,
            "dimension": novo_idx,
            "name": name,
            "error_before": sum(self.global_errors[-10:]) / 10,
        })

        # Resetar erros apos descoberta
        self.global_errors = []

    def get_state(self) -> Dict:
        return {
            "n_objects": len(self.states),
            "n_dims": self.n_dims,
            "discoveries": len(self.discovery_history),
            "avg_global_error": sum(self.global_errors[-20:]) / max(1, len(self.global_errors[-20:])),
            "recent_discovery": self.discovery_history[-1] if self.discovery_history else None,
        }

"""
WorldModel — Modelo interno do mundo como cidadão de primeira classe.

Funcoes:
  1. Predizer: dado um estado, o que esperamos observar?
  2. Atualizar: quando a realidade chega, comparar com a predicao
  3. Surpresa: prediction error que dispara mudanca de modo cognitivo

O WorldModel e o centro do ciclo:
  observacao → predicao → execucao → erro → atualizacao do modelo

Em vez de ser apenas mais um modulo, ele e o estado compartilhado
que todos os modulos consultam e atualizam.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time
import math

try:
    from .causal_core import StructuralCausalModel
    CAUSAL_CORE_OK = True
except ImportError:
    CAUSAL_CORE_OK = False


@dataclass
class Prediction:
    """Uma predicao feita pelo WorldModel."""
    input_state: Any
    predicted_output: Any
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 3),
            "timestamp": round(self.timestamp, 3),
        }


@dataclass
class PredictionError:
    """Erro de predicao: o gap entre o previsto e o observado."""
    prediction: Prediction
    actual_output: Any
    error_magnitude: float      # 0.0 (perfeito) a 1.0 (completamente errado)
    surprise: float             # 0.0 a 1.0, quao inesperado foi o erro
    timestamp: float = field(default_factory=time.time)

    @property
    def is_surprising(self) -> bool:
        return self.surprise > 0.5

    def to_dict(self) -> dict:
        return {
            "error": round(self.error_magnitude, 3),
            "surprise": round(self.surprise, 3),
            "is_surprising": self.is_surprising,
        }


class WorldModel:
    """Modelo interno do mundo.

    Mantem:
      - Ultimas N predicoes e erros
      - Surpresa media recente
      - Confianca media nas predicoes
      - Estado atual do CGE como snapshot do modelo

    Uso:
        wm = WorldModel()
        pred = wm.predict(input_state)
        ... executa ...
        error = wm.update(input_state, actual_output, pred)
        if error.is_surprising:
            change_to_exploration_mode()
    """

    def __init__(self, max_history: int = 100,
                 law_registry=None,
                 hdv_memory=None,
                 knowledge_graph=None):
        self.predictions: deque = deque(maxlen=max_history)
        self.errors: deque = deque(maxlen=max_history)
        self.law_registry = law_registry
        self.hdv_memory = hdv_memory
        self.knowledge_graph = knowledge_graph

        # Metrics
        self.total_predictions: int = 0
        self.total_errors: int = 0
        self.recent_surprise: deque = deque(maxlen=20)
        self.recent_accuracy: deque = deque(maxlen=20)

        # Modelo interno
        self._concept_states: Dict[str, float] = {}
        self._transition_counts: Dict[Tuple[str, str], int] = {}

        # Causal Core (do-calculus, Pearl Niveis 2-3)
        self.causal_core = StructuralCausalModel() if CAUSAL_CORE_OK else None
        self._n_interventions = 0
        self._n_counterfactuals = 0

        # Consistency tracking
        self.consistency_checks: int = 0
        self.consistency_violations: int = 0

        self.last_update: float = time.time()

    def check_consistency(self, domain: str, observation: dict) -> List[dict]:
        """Verifica consistencia entre leis do mundo e observacao.

        Returns: lista de conflitos encontrados.
        """
        self.consistency_checks += 1
        if self.law_registry is None:
            return []

        conflicts = self.law_registry.check_consistency(domain, observation)
        if conflicts:
            self.consistency_violations += len(conflicts)
        return conflicts

    def get_consistency(self) -> float:
        """Proporcao de verificacoes consistentes."""
        if self.consistency_checks == 0:
            return 1.0
        return 1.0 - (self.consistency_violations / self.consistency_checks)

    def predict(self, input_state: Any, context: Optional[dict] = None) -> Prediction:
        """Faz uma predicao sobre o que esperamos observar.

        Quanto mais confiante o modelo, mais precisa a predicao.
        Quanto menos confiante, mais ampla a margem de erro esperada.
        """
        self.total_predictions += 1
        confidence = self._estimate_confidence(input_state)

        # Predicao baseada no modelo interno
        predicted = self._predict_from_model(input_state, context)

        return Prediction(
            input_state=input_state,
            predicted_output=predicted,
            confidence=confidence,
        )

    def _estimate_confidence(self, input_state: Any) -> float:
        """Estima a confianca do modelo para este estado."""
        if self.total_errors == 0:
            return 0.5  # confianca inicial moderada
        recent_acc = list(self.recent_accuracy)
        if recent_acc:
            return float(np.mean(recent_acc))
        return 0.5

    def _predict_from_model(self, input_state: Any,
                             context: Optional[dict] = None) -> Any:
        """Gera predicao real: consulta HDV + KG para estimar o esperado.

        1. Se input_state e texto, consulta o HDV brain por similaridade
        2. Se encontrou similar, retorna o rotulo como predicao
        3. Se tem KG, busca relacoes causais relevantes
        4. Se nada encontrou, retorna input_state (confianca baixa)
        """
        if isinstance(input_state, str):
            # Consulta o cerebro HDV
            if self.hdv_memory is not None:
                results = self.hdv_memory.query(input_state, top_k=3, min_sim=0.2)
                if results:
                    # Retorna o contexto do HDV como predicao
                    context_str = "; ".join(
                        meta.get("text", meta.get("id", ""))[:60]
                        for sim, meta, src in results
                    )
                    return context_str

            # Consulta o KG por relacoes causais
            if self.knowledge_graph is not None and hasattr(self.knowledge_graph, 'get_causal_relations'):
                rels = self.knowledge_graph.get_causal_relations(domain=input_state)
                if rels:
                    return f"{rels[0].source} -> {rels[0].target}"

        # Se input_state e dict com "task", tenta prever resultado
        if isinstance(input_state, dict) and context:
            task = input_state.get("task", "")
            if task and self.hdv_memory is not None:
                results = self.hdv_memory.query(task, top_k=2, min_sim=0.2)
                if results:
                    return results[0][1].get("text", "")[:100]

        return input_state

    def update(self, input_state: Any, actual_output: Any,
               prediction: Prediction) -> PredictionError:
        """Atualiza o modelo com o resultado real.

        Args:
            input_state: estado de entrada
            actual_output: saida real observada
            prediction: predicao que havia sido feita

        Returns: PredictionError com magnitude e surpresa.
        """
        self.total_errors += 1
        self.last_update = time.time()

        # Calcula erro
        error_mag = self._compute_error(actual_output, prediction.predicted_output)

        # Surpresa: erro * (1 - confianca)
        # Se confianca era alta e erro foi alto → surpresa maxima
        # Se confianca era baixa e erro foi alto → surpresa media
        surprise = error_mag * (1.0 - prediction.confidence)
        surprise = min(1.0, surprise * 2.0)  # escala

        error_record = PredictionError(
            prediction=prediction,
            actual_output=actual_output,
            error_magnitude=error_mag,
            surprise=surprise,
        )

        self.errors.append(error_record)
        self.recent_surprise.append(surprise)
        self.recent_accuracy.append(1.0 - error_mag)

        return error_record

    def _compute_error(self, actual: Any, predicted: Any) -> float:
        """Computa erro entre saida real e prevista.

        0.0 = identico, 1.0 = completamente diferente.
        """
        if predicted is None:
            return 0.5  # erro medio se nao havia predicao

        if isinstance(actual, np.ndarray) and isinstance(predicted, np.ndarray):
            if actual.shape != predicted.shape:
                return 1.0
            total = int(actual.size)
            if total == 0:
                return 0.0
            wrong = int((actual != predicted).sum())
            return wrong / total

        if actual == predicted:
            return 0.0
        return 1.0

    def get_surprise(self) -> float:
        """Surpresa media recente (0-1)."""
        if not self.recent_surprise:
            return 0.0
        return float(np.mean(self.recent_surprise))

    def get_accuracy(self) -> float:
        """Acuracia media recente (0-1)."""
        if not self.recent_accuracy:
            return 0.0
        return float(np.mean(self.recent_accuracy))

    def is_surprised(self, threshold: float = 0.5) -> bool:
        """O modelo esta surpreso com os resultados recentes?"""
        return self.get_surprise() > threshold

    def record_transition(self, from_concept: str, to_concept: str):
        """Registra uma transicao de estado conceitual."""
        key = (from_concept, to_concept)
        self._transition_counts[key] = self._transition_counts.get(key, 0) + 1

    def get_common_transitions(self, top_n: int = 10) -> List[Tuple[str, str, int]]:
        """Transicoes mais comuns."""
        sorted_trans = sorted(
            self._transition_counts.items(),
            key=lambda x: -x[1]
        )
        return [(k[0], k[1], v) for k, v in sorted_trans[:top_n]]

    def query_intervention(self, do: Dict[str, float], target: str = "",
                            n_samples: int = 200) -> Dict[str, float]:
        """Nivel 2: P(Y | do(X=x)). Efeito causal verdadeiro.

        Usa CausalCore se disponivel, senao cai para heuristica.

        Args:
            do: {var: valor} — intervencao
            target: variavel alvo
        Returns:
            {target: {"mean": ..., "std": ...}}
        """
        self._n_interventions += 1
        if self.causal_core is not None and hasattr(self.causal_core, 'query_intervention'):
            try:
                data = {}
                if self._concept_states:
                    for k, v in self._concept_states.items():
                        data[k] = np.array([v])
                if data:
                    self.causal_core.learn_structure(data, method="prior")
                    self.causal_core.learn_functions(data)
                return self.causal_core.query_intervention(do, target, n_samples)
            except Exception:
                pass

        for var, val in do.items():
            if var in self._concept_states:
                self._concept_states[var] = val
        return {target or list(do.keys())[0]: {"mean": 0.0, "std": 0.5}}

    def query_counterfactual(self, evidence: Dict[str, float],
                              do: Dict[str, float], target: str = "",
                              n_samples: int = 200) -> Dict[str, float]:
        """Nivel 3: P(Y_{X=x} | X=x', Y=y').

        Usa CausalCore se disponivel.

        Args:
            evidence: valores observados
            do: valores contrafactuais
            target: variavel alvo
        """
        self._n_counterfactuals += 1
        if self.causal_core is not None and hasattr(self.causal_core, 'query_counterfactual'):
            try:
                data = {}
                if self._concept_states:
                    for k, v in self._concept_states.items():
                        data[k] = np.array([v])
                if data:
                    self.causal_core.learn_structure(data, method="prior")
                    self.causal_core.learn_functions(data)
                return self.causal_core.query_counterfactual(evidence, do, target, n_samples)
            except Exception:
                pass
        return {target or list(do.keys())[0]: {"mean": 0.0, "std": 0.5}}

    def get_stats(self) -> dict:
        return {
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "recent_surprise": round(self.get_surprise(), 3),
            "recent_accuracy": round(self.get_accuracy(), 3),
            "is_surprised": self.is_surprised(),
            "n_interventions": self._n_interventions,
            "n_counterfactuals": self._n_counterfactuals,
            "has_causal_core": self.causal_core is not None,
            "model_size": {
                "concept_states": len(self._concept_states),
                "transitions": len(self._transition_counts),
            },
        }

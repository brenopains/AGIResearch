"""
Predictive Coding Engine — Free Energy via distância HDC
===========================================================
O prediction error (energia livre) agora é calculado como
distância de cosseno no espaço hiperdimensional (1 - sim),
não mais como divergência de strings (SequenceMatcher).

Isso permite que a AGI sinta surpresa/confusão REAL:
  - Input conhecido → alta similaridade HDC → baixa energia
  - Input desconhecido → baixa similaridade HDC → alta energia
"""

import numpy as np
import time
from typing import Optional, TYPE_CHECKING
from collections import deque

from .coglang import Manifold

if TYPE_CHECKING:
    from .grounding_engine import GroundingEngine
    from .hdc_memory import HDCMemory


class PredictiveCodingEngine:
    """
    Hierarchical Predictive Coding based on the Free Energy Principle.

    Each layer generates top-down predictions about what the layer below
    will produce. Prediction errors drive learning and precision weighting.

    Architecture:
        Top-down: expectations based on world model + memory
        Bottom-up: actual sensory input
        Precision: attention weighting based on prediction confidence
        Error: drives neuroplasticity and learning rate
    """

    def __init__(self, prediction_window: int = 5, engine: Optional['GroundingEngine'] = None):
        self.prediction_window = prediction_window
        self.last_prediction = ""
        self.last_actual = ""
        self.prediction_error = 0.0
        self.precision = 1.0
        self.error_history: deque = deque(maxlen=20)
        self.surprise_history: deque = deque(maxlen=20)
        self._last_context = ""
        self._engine = engine
        self._hdc_memory = None

    def set_engine(self, engine: 'GroundingEngine'):
        """Inject GroundingEngine for HDC-based prediction error."""
        self._engine = engine

    def set_hdc_memory(self, hdc_memory: 'HDCMemory'):
        """Inject HDCMemory for top-down prediction from .hdv."""
        self._hdc_memory = hdc_memory

    def _compute_hdc_error(self, actual_text: str, predicted_text: str) -> float:
        """
        Computa prediction error como distância de cosseno HDC.
        1 - cosine_similarity(encode(actual), encode(predicted))
        Retorna 0 se forem idênticos, ~1 se forem ortogonais.
        """
        if not actual_text or not predicted_text:
            return 0.5
        if self._engine is None:
            return 0.5
        try:
            actual_mf = self._engine.encode_text(actual_text[:2000])
            predicted_mf = self._engine.encode_text(predicted_text[:2000])
            sim = actual_mf.similarity(predicted_mf)
            # Normaliza de [-1, 1] para [0, 1]
            error = max(0.0, min(1.0, (1.0 - sim) / 2.0))
            return error
        except Exception:
            return 0.5

    def predict_next(self, current_input: str, world_model: object = None,
                     memory: object = None) -> str:
        predictions = []

        if self.error_history:
            min_err = min(list(self.error_history)[-3:]) if len(self.error_history) >= 3 else 0.5
            if min_err < 0.3:
                predictions.append(self.last_actual)

        if memory is not None:
            try:
                rules = memory.recall_semantic({"input": current_input}, top_k=3)
                if rules:
                    for rule in rules[:2]:
                        content = rule.content
                        if isinstance(content, dict):
                            pred = content.get("consequent") or content.get("prediction") or content.get("text", "")
                            if isinstance(pred, str) and len(pred) > 5:
                                predictions.append(pred)
            except Exception:
                pass

        if world_model is not None:
            try:
                best_node = None
                best_sim = 0
                if self._engine is not None:
                    input_mf = self._engine.encode_text(current_input[:500])
                    for nid, node in world_model.nodes.items():
                        label = node.label
                        if not label:
                            continue
                        try:
                            node_mf = self._engine.encode_text(label[:500])
                            sim = input_mf.similarity(node_mf)
                            if sim > best_sim:
                                best_sim = sim
                                best_node = nid
                        except Exception:
                            pass
                else:
                    from difflib import SequenceMatcher
                    for nid, node in world_model.nodes.items():
                        label = node.label
                        ratio = SequenceMatcher(None, current_input.lower(), label.lower()).ratio()
                        if ratio > best_sim:
                            best_sim = ratio
                            best_node = nid
                if best_node:
                    children = world_model.adjacency.get(best_node, set())
                    if children:
                        child = next(iter(children))
                        child_node = world_model.nodes.get(child)
                        if child_node:
                            predictions.append(child_node.label)
            except Exception:
                pass

        # HDV Memory: busca o texto mais similar como predição top-down
        if not predictions and self._hdc_memory is not None and self._engine is not None:
            try:
                hdc_results = self._hdc_memory.query(current_input, top_k=1, min_sim=0.0)
                if hdc_results:
                    sim, meta, src = hdc_results[0]
                    meta_text = meta.get('text', meta.get('id', ''))
                    if meta_text and len(meta_text) > 10:
                        predictions.append(meta_text[:200])
            except Exception:
                pass

        if not predictions:
            return ""

        self.last_prediction = max(set(predictions), key=predictions.count)
        return self.last_prediction

    def compute_prediction_error(self, actual: str, expected: str) -> float:
        """
        Compute prediction error using HDC cosine distance.
        Substitui o antigo SequenceMatcher.
        """
        if not expected or not actual:
            return 0.5
        error = self._compute_hdc_error(actual, expected)
        # Mínimo de 0.05 para evitar divisão por zero em learning rates
        error = max(0.05, min(1.0, error))
        self.last_actual = actual
        self.prediction_error = error
        self.error_history.append(error)
        return error

    def update_precision(self, error: float):
        if error < 0.2:
            self.precision = min(2.0, self.precision + 0.1)
        elif error > 0.6:
            self.precision = max(0.5, self.precision - 0.15)
        else:
            self.precision = 1.0
        self.surprise_history.append(error)

    def get_attention_weight(self) -> float:
        if not self.error_history:
            return 1.0
        recent = list(self.error_history)[-3:]
        avg_surprise = sum(recent) / len(recent)
        return min(3.0, max(0.5, avg_surprise * 3.0))

    def get_learning_rate_modulator(self) -> float:
        if len(self.error_history) < 3:
            return 1.0
        recent = list(self.error_history)[-3:]
        gradient = recent[-1] - recent[-2] if len(recent) >= 2 else 0
        if gradient > 0.1:
            return 2.0
        elif gradient < -0.1:
            return 0.5
        return 1.0

    def reset(self):
        self.last_prediction = ""
        self.last_actual = ""
        self.prediction_error = 0.0
        self.precision = 1.0
        self.error_history.clear()
        self.surprise_history.clear()

    def get_state(self) -> dict:
        return {
            "prediction_error": self.prediction_error,
            "precision": self.precision,
            "attention_weight": self.get_attention_weight(),
            "learning_rate_modulator": self.get_learning_rate_modulator(),
            "avg_surprise": float(np.mean(self.surprise_history)) if self.surprise_history else 0,
            "recent_errors": list(self.error_history)[-5:],
        }

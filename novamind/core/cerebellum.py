"""
Cerebellum — Timing e Coordenação de Gestos Articulatórios
============================================================
O cerebelo humano (lóbulos V, VI, VIII) recebe cópia do comando motor
via ponte e retorna correções de timing via tálamo.

Danos cerebelares → fala "scanned" (silabada, sem fluência).
O cerebelo NÃO inicia movimento, mas coordena a PRECISÃO TEMPORAL
entre múltiplos articuladores.

Circuito: córtex → ponte → cerebelo → núcleo denteado → tálamo → córtex
Referência: "The cerebellum's role in timing" (2024), OAEPublish
"""

import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque


GESTURE_TIMING_PROTOTYPES = {
    "jaw_open": {"min_ms": 30, "max_ms": 200, "default_ms": 80},
    "tongue": {"min_ms": 40, "max_ms": 250, "default_ms": 100},
    "lip": {"min_ms": 20, "max_ms": 150, "default_ms": 60},
    "velum": {"min_ms": 30, "max_ms": 120, "default_ms": 50},
    "vocal_fold": {"min_ms": 10, "max_ms": 80, "default_ms": 30},
}


class Cerebellum:
    def __init__(self, learning_rate: float = 0.05):
        self.lr = learning_rate
        self.timing_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.interval_predictions: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))

    def predict_timing(self, gesture: Dict) -> Dict[str, float]:
        """
        Prediz a duração ideal para cada articulador no gesto atual.
        Retorna dict com tempos em ms para cada articulador.
        """
        result = {}
        for key in ["jaw_open", "lip_rounding", "velum", "vocal_fold_tension"]:
            if key in gesture:
                proto = GESTURE_TIMING_PROTOTYPES.get(key, GESTURE_TIMING_PROTOTYPES["jaw_open"])
                delta = abs(gesture.get(key, 0.5) - 0.5) * 2  # 0-1 scaled to 0-1
                predicted = proto["min_ms"] + delta * (proto["max_ms"] - proto["min_ms"])

                history = self.timing_predictions.get(key)
                if history:
                    predicted = predicted * 0.7 + np.mean(history) * 0.3
                result[key + "_ms"] = float(predicted)
        if "tongue_dorsum_x" in gesture or "tongue_dorsum_y" in gesture:
            tx = gesture.get("tongue_dorsum_x", 0.5)
            ty = gesture.get("tongue_dorsum_y", 0.5)
            delta = (abs(tx - 0.5) + abs(ty - 0.5)) * 0.5
            proto = GESTURE_TIMING_PROTOTYPES["tongue"]
            predicted = proto["min_ms"] + delta * (proto["max_ms"] - proto["min_ms"])
            result["tongue_ms"] = float(predicted)

        result["burst_ms"] = 20.0
        result["closure_ms"] = 80.0
        return result

    def predict_interval(self, prev_gesture: Dict, next_gesture: Dict) -> float:
        """Prediz intervalo entre dois gestos consecutivos em ms."""
        key = (prev_gesture.get("type", ""), next_gesture.get("type", ""))
        defaults = {
            ("plosive", "vowel"): 20.0,
            ("fricative", "vowel"): 30.0,
            ("nasal", "vowel"): 40.0,
            ("vowel", "vowel"): 50.0,
            ("vowel", "plosive"): 30.0,
            ("closure", "burst"): 5.0,
            ("burst", "vowel"): 10.0,
        }
        predicted = defaults.get(key, 40.0)
        history = self.interval_predictions.get(key)
        if history:
            predicted = predicted * 0.6 + np.mean(history) * 0.4
        return float(predicted)

    def correct_timing(self, actual_ms: float, predicted_ms: float) -> float:
        """
        Corrige timing via erro de predição (sinal de Purkinje).
        Retorna correção em ms (positivo = mais lento, negativo = mais rápido).
        """
        error = actual_ms - predicted_ms
        correction = -self.lr * error
        return float(np.clip(correction, -50.0, 50.0))

    def learn(self, key: str, actual_duration_ms: float):
        """Atualiza predição de timing baseado no resultado real."""
        self.timing_predictions[key].append(actual_duration_ms)

    def learn_interval(self, prev_type: str, next_type: str, actual_interval_ms: float):
        """Atualiza predição de intervalo entre gestos."""
        self.interval_predictions[(prev_type, next_type)].append(actual_interval_ms)

    def get_state(self) -> Dict:
        return {
            "timing_prototypes": len(self.timing_predictions),
            "interval_prototypes": len(self.interval_predictions),
        }

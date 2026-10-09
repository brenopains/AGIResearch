"""
calibration_loop.py — Loop Continuo de Calibracao Metacognitiva
================================================================
Conecta o ConfidenceCalibrator ao ciclo cognitivo real.

Toda vez que o sistema faz uma predicao (WorldModel.predict,
Competition, simulacao), o loop registra (confianca, acertou?).
Periodicamente, treina o calibrador com os pares acumulados.

Isto resolve o gap fundamental: o ConfidenceCalibrator.fit()
NUNCA era chamado. Agora e chamado continuamente durante a
operacao normal do sistema.

Uso:
    loop = CalibrationLoop(confidence_calibrator)
    loop.record_prediction(confidence=0.7, correct=True)
    loop.record_prediction(confidence=0.3, correct=False)
    ...
    loop.maybe_train()  # a cada N registros, treina calibrador
"""
import time
import numpy as np
from typing import Optional, List, Dict, Any
from collections import deque

from .calibration_metrics import CalibrationMetrics


class CalibrationLoop:
    """
    Loop de calibracao que treina o ConfidenceCalibrator com
    predicoes reais do sistema.

    Deve ser chamado pelo Maestro ou CognitiveExecutive a cada ciclo.
    """

    def __init__(self, calibrator=None, max_history: int = 5000,
                 train_interval: int = 100, min_train_samples: int = 50):
        self.calibrator = calibrator
        self.metrics = CalibrationMetrics(max_history=max_history)
        self.train_interval = train_interval
        self.min_train_samples = min_train_samples
        self._n_since_train = 0
        self._last_train_time = time.time()
        self._train_count = 0
        self._calibration_history: List[dict] = []

    def record_prediction(self, confidence: float, correct: bool,
                          is_unknowable: bool = False, domain: str = "") -> None:
        """Registra uma predicao e sua acuracia real."""
        self.metrics.add_prediction(confidence, correct, is_unknowable, domain)
        self._n_since_train += 1

    def record_batch(self, confidences: List[float], corrects: List[bool],
                     unknowables: Optional[List[bool]] = None) -> None:
        """Registra varias predicoes de uma vez."""
        self.metrics.add_batch(confidences, corrects, unknowables)
        self._n_since_train += len(confidences)

    def maybe_train(self, force: bool = False) -> Optional[dict]:
        """
        Treina o calibrador se acumulou predicoes suficientes.
        
        Returns: dict com metricas pre-treino, ou None se nao treinou.
        """
        if self.calibrator is None:
            return None
        if not force and self._n_since_train < self.train_interval:
            return None
        if self.metrics.n < self.min_train_samples:
            return None

        pre_metrics = self.metrics.summary()

        confs = self.metrics.confidence_array()
        corrects = self.metrics.correct_array()

        try:
            if hasattr(self.calibrator, 'fit'):
                self.calibrator.fit(confs.tolist(), corrects.tolist())
                self._train_count += 1
                self._n_since_train = 0
                self._last_train_time = time.time()
            elif hasattr(self.calibrator, 'calibrate'):
                self.calibrator.calibrate(confs.tolist())
                self._train_count += 1
                self._n_since_train = 0
        except Exception:
            return None

        post_metrics = self.metrics.summary()

        result = {
            "train_count": self._train_count,
            "pre": pre_metrics,
            "post": post_metrics,
            "improvement": {
                "ECE_delta": pre_metrics.get("ECE", 1.0) - post_metrics.get("ECE", 1.0),
                "Brier_delta": pre_metrics.get("Brier", 1.0) - post_metrics.get("Brier", 1.0),
            }
        }
        self._calibration_history.append(result)
        if len(self._calibration_history) > 100:
            self._calibration_history = self._calibration_history[-100:]

        return result

    def apply_calibration(self, raw_confidence: float) -> float:
        """Aplica calibracao a uma confianca bruta."""
        if self.calibrator is None:
            return raw_confidence
        try:
            if hasattr(self.calibrator, 'calibrate'):
                return self.calibrator.calibrate(raw_confidence)
            elif hasattr(self.calibrator, 'calibrate_batch'):
                return self.calibrator.calibrate_batch([raw_confidence])[0]
        except Exception:
            pass
        return raw_confidence

    def get_metrics(self) -> dict:
        """Retorna metricas atuais de calibracao."""
        return self.metrics.summary()

    def get_history(self) -> List[dict]:
        """Retorna historico de treinos."""
        return self._calibration_history

    def get_stats(self) -> dict:
        return {
            "n_predictions": self.metrics.n,
            "n_trains": self._train_count,
            "last_train": self._last_train_time,
            "metrics": self.metrics.summary(),
        }

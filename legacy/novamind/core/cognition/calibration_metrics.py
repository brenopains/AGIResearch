"""
calibration_metrics.py — Metricas de Calibracao Probabilistica
================================================================
ECE, Brier Score, Reliability Diagram, Abstention AUROC.

Essas metricas NAO existiam no repositorio — sao o que falta para
o sistema saber se suas confiancas sao calibradas.

Uso:
    metrics = CalibrationMetrics()
    metrics.add_prediction(confidence=0.7, correct=True)
    ...
    ece = metrics.ece()  # Expected Calibration Error
    brier = metrics.brier()
    auroc = metrics.abstention_auroc()
    metrics.reliability_diagram()  # plot (se matplotlib disponivel)
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PredictionRecord:
    """Registro de uma predicao com confianca e resultado."""
    confidence: float = 0.5
    correct: bool = False
    is_unknowable: bool = False
    timestamp: float = 0.0
    domain: str = ""


class CalibrationMetrics:
    """
    Acumula predicoes e computa metricas de calibracao.
    
    Nao depende de nada do NovaMind — pode ser usado por qualquer
    sistema que produza pares (confianca, acertou).
    """

    def __init__(self, max_history: int = 10000):
        self.predictions: List[PredictionRecord] = []
        self.max_history = max_history

    def add_prediction(self, confidence: float, correct: bool,
                       is_unknowable: bool = False, domain: str = "",
                       timestamp: Optional[float] = None) -> None:
        """Registra uma predicao."""
        if confidence < 0 or confidence > 1:
            confidence = np.clip(confidence, 0, 1)
        import time
        rec = PredictionRecord(
            confidence=confidence,
            correct=correct,
            is_unknowable=is_unknowable,
            timestamp=timestamp or time.time(),
            domain=domain,
        )
        self.predictions.append(rec)
        if len(self.predictions) > self.max_history:
            self.predictions = self.predictions[-self.max_history:]

    def add_batch(self, confidences: List[float], corrects: List[bool],
                  unknowables: Optional[List[bool]] = None) -> None:
        """Registra varias predicoes de uma vez."""
        if unknowables is None:
            unknowables = [False] * len(confidences)
        for c, ok, uk in zip(confidences, corrects, unknowables):
            self.add_prediction(c, ok, uk)

    def clear(self):
        self.predictions = []

    def confidence_array(self) -> np.ndarray:
        return np.array([p.confidence for p in self.predictions])

    def correct_array(self) -> np.ndarray:
        return np.array([1.0 if p.correct else 0.0 for p in self.predictions])

    @property
    def n(self) -> int:
        return len(self.predictions)

    # ─── ECE (Expected Calibration Error) ───

    def ece(self, n_bins: int = 10) -> float:
        """
        Expected Calibration Error.
        
        Mede: |confianca_media_no_bin - acuracia_no_bin|,
        ponderado pela fração de amostras em cada bin.
        
        0 = perfeito. 1 = pessimo.
        """
        if self.n < 5:
            return 1.0
        
        confs = self.confidence_array()
        corrects = self.correct_array()
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece_val = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            bin_conf = np.mean(confs[mask])
            bin_acc = np.mean(corrects[mask])
            ece_val += n_in_bin * abs(bin_conf - bin_acc)
        
        return ece_val / self.n

    def max_calibration_error(self, n_bins: int = 10) -> float:
        """MCE — pior desvio em qualquer bin."""
        if self.n < 5:
            return 1.0
        
        confs = self.confidence_array()
        corrects = self.correct_array()
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        max_err = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            bin_conf = np.mean(confs[mask])
            bin_acc = np.mean(corrects[mask])
            max_err = max(max_err, abs(bin_conf - bin_acc))
        
        return max_err

    # ─── Brier Score ───

    def brier(self) -> float:
        """
        Brier Score: MSE entre confianca e resultado.
        
        Menor = melhor. 0 = perfeito. 0.25 = sempre 50%.
        """
        if self.n == 0:
            return 1.0
        confs = self.confidence_array()
        corrects = self.correct_array()
        return float(np.mean((confs - corrects) ** 2))

    def brier_skill_score(self, reference: Optional[float] = None) -> float:
        """
        Brier Skill Score: 1 - Brier/Brier_ref.
        
        1 = perfeito. 0 = igual a referencia. <0 = pior que referencia.
        """
        b = self.brier()
        if reference is None:
            ref = self.uniform_brier()
        else:
            ref = reference
        return 1.0 - b / ref if ref > 0 else 0.0

    @staticmethod
    def uniform_brier() -> float:
        """Brier de um preditor que sempre chuta 0.5."""
        return 0.25

    # ─── Abstention AUROC ───

    def abstention_auroc(self) -> float:
        """
        AUROC para deteccao de ignorancia.
        
        Alta confianca em perguntas respondiveis, baixa em impossiveis.
        1.0 = perfeitamente calibrado. 0.5 = aleatorio.
        """
        if self.n < 3:
            return 0.5
        
        uk_mask = np.array([p.is_unknowable for p in self.predictions])
        if uk_mask.sum() == 0 or uk_mask.sum() == len(uk_mask):
            return 0.5
        
        confs = self.confidence_array()
        
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(uk_mask.astype(int), -confs))
        except Exception:
            return 0.5

    # ─── Reliability Diagram ───

    def reliability_data(self, n_bins: int = 10) -> dict:
        """Retorna dados para reliability diagram."""
        if self.n < 5:
            return {"bins": [], "accuracies": [], "confidences": [], "counts": []}
        
        confs = self.confidence_array()
        corrects = self.correct_array()
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        result = {"bins": [], "accuracies": [], "confidences": [], "counts": []}
        for i in range(n_bins):
            mask = bin_indices == i
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            result["bins"].append((bins[i] + bins[i+1]) / 2)
            result["confidences"].append(float(np.mean(confs[mask])))
            result["accuracies"].append(float(np.mean(corrects[mask])))
            result["counts"].append(int(n_in_bin))
        
        return result

    # ─── Resumo ───

    def summary(self) -> dict:
        return {
            "n": self.n,
            "ECE": round(self.ece(), 4),
            "MCE": round(self.max_calibration_error(), 4),
            "Brier": round(self.brier(), 4),
            "BSS": round(self.brier_skill_score(), 4),
            "AUROC_abstention": round(self.abstention_auroc(), 4),
            "mean_confidence": round(float(np.mean(self.confidence_array())), 4),
            "accuracy": round(float(np.mean(self.correct_array())), 4),
        }

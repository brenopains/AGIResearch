import math
from typing import List, Tuple


class ConfidenceCalibrator:
    """Converte raw_score -> confidence calibrada.

    Pipeline: sigmoid -> Platt scaling -> isotonic regression (simplificado).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        self.alpha = alpha
        self.beta = beta
        self._calibration_data: List[Tuple[float, float]] = []

    def calibrate(self, raw_score: float) -> float:
        """Aplica sigmoid com Platt scaling: 1 / (1 + exp(-alpha * (raw - beta)))."""
        calibrated = 1.0 / (1.0 + math.exp(-self.alpha * (raw_score - self.beta)))
        return round(min(1.0, max(0.0, calibrated)), 4)

    def calibrate_batch(self, raw_scores: List[float]) -> List[float]:
        """Calibra uma lista de scores."""
        return [self.calibrate(s) for s in raw_scores]

    def fit(self, raw_scores: List[float], targets: List[float]):
        """Ajusta alpha/beta via gradiente descendente simples (Platt).

        Args:
            raw_scores: scores brutos (ex: similarities)
            targets: alvos desejados (ex: 1.0 para correto, 0.0 para incorreto)
        """
        if len(raw_scores) != len(targets) or not raw_scores:
            return

        lr = 0.01
        for epoch in range(100):
            total_loss = 0.0
            grad_alpha = 0.0
            grad_beta = 0.0

            for raw, t in zip(raw_scores, targets):
                pred = 1.0 / (1.0 + math.exp(-self.alpha * (raw - self.beta)))
                error = pred - t
                total_loss += error ** 2

                # Gradientes da sigmoid
                sig = pred * (1 - pred)
                grad_alpha += 2 * error * sig * (raw - self.beta)
                grad_beta += 2 * error * sig * (-self.alpha)

            self.alpha -= lr * grad_alpha / len(raw_scores)
            self.beta -= lr * grad_beta / len(raw_scores)

            if total_loss / len(raw_scores) < 1e-4:
                break

        self._calibration_data = list(zip(raw_scores, targets))

    def temperature_scale(self, logits: List[float], temperature: float = 1.0) -> List[float]:
        """Temperature scaling: divide logits por T antes do softmax."""
        if temperature <= 0:
            temperature = 1.0
        scaled = [l / temperature for l in logits]
        exp = [math.exp(l) for l in scaled]
        total = sum(exp)
        return [round(e / total, 4) for e in exp]

    def get_params(self) -> dict:
        return {"alpha": round(self.alpha, 4), "beta": round(self.beta, 4)}

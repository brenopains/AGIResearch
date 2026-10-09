import numpy as np
import math

class gen_PlattScalingDomainAdapter_1784605506:
    """
    Calibrador Platt por dominio, gerado para preencher gap de calibracao.
    Criado automaticamente pelo ModuleGenerator.
    """
    def __init__(self):
        self.name = "gen_PlattScalingDomainAdapter_1784605506"
        self.gap_type = "calibration_missing"
        self.trial = "T3"
        self.alpha = 1.0
        self.beta = 0.0
        self._fitted = False
        self._domains = {}

    def fit(self, confidences, corrects, domain="unknown"):
        import numpy as np
        import math
        targets = np.clip(np.array(corrects), 1e-10, 1 - 1e-10)
        logits = np.clip(np.array(confidences), 1e-10, 1 - 1e-10)
        log_odds = np.log(targets / (1 - targets))
        A = np.column_stack([logits, np.ones_like(logits)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, log_odds, rcond=None)
            self._domains[domain] = {"alpha": float(coeffs[0]), "beta": float(coeffs[1])}
            self._fitted = True
        except Exception:
            self._domains[domain] = {"alpha": 1.0, "beta": 0.0}

    def calibrate(self, confidence, domain="unknown"):
        import math
        if domain in self._domains:
            a = self._domains[domain]["alpha"]
            b = self._domains[domain]["beta"]
        elif self._fitted:
            a = list(self._domains.values())[0]["alpha"]
            b = list(self._domains.values())[0]["beta"]
        else:
            return confidence
        calibrated = 1.0 / (1.0 + math.exp(-(a * confidence + b)))
        return max(0.01, min(0.99, calibrated))

    def __call__(self, confidence, domain="unknown"):
        return self.calibrate(confidence, domain)


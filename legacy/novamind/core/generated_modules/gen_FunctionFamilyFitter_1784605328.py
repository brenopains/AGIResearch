import numpy as np

import numpy as np

class gen_FunctionFamilyFitter_1784605328:
    """
    Ajustador de funcao com selecao de familia, gerado para gap de fitting.
    Criado automaticamente pelo ModuleGenerator.
    """
    def __init__(self):
        self.name = "gen_FunctionFamilyFitter_1784605328"
        self.gap_type = "poor_function_fitting"
        self.trial = "T4"
        self._families = ["linear", "log_linear", "polynomial"]
        self._best = None
        self._params = {}

    def fit(self, X, y):
        n = len(y)
        candidates = []
        import numpy as np

        # Linear
        try:
            X_aug = np.column_stack([np.ones(n), X])
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            preds = X_aug @ coeffs
            resid = float(np.std(y - preds))
            if resid < np.std(y) * 0.95:
                candidates.append(("linear", resid, coeffs))
        except Exception:
            pass

        # Log-linear
        try:
            shift = 1.0
            y_log = np.log(np.maximum(y + shift, 1e-6))
            X_log = np.column_stack([np.log(np.maximum(X[:, i] + shift, 1e-6)) for i in range(X.shape[1])])
            X_log_aug = np.column_stack([np.ones(n), X_log])
            coeffs, _, _, _ = np.linalg.lstsq(X_log_aug, y_log, rcond=None)
            preds = np.exp(X_log_aug @ coeffs) - shift
            resid = float(np.std(y - preds))
            if resid < np.std(y) * 0.95:
                candidates.append(("log_linear", resid, coeffs))
        except Exception:
            pass

        if candidates:
            candidates.sort(key=lambda c: c[1])
            self._best = candidates[0][0]
            self._params = {"type": self._best, "coeffs": candidates[0][2].tolist()}
        return self._best is not None

    def predict(self, X):
        if self._best is None:
            return np.zeros(X.shape[0])
        if self._best == "linear":
            return X @ self._params["coeffs"][1:] + self._params["coeffs"][0]
        elif self._best == "log_linear":
            shift = 1.0
            X_log = np.column_stack([np.log(np.maximum(X[:, i] + shift, 1e-6)) for i in range(X.shape[1])])
            return np.exp(np.column_stack([np.ones(X.shape[0]), X_log]) @ self._params["coeffs"]) - shift
        return np.zeros(X.shape[0])


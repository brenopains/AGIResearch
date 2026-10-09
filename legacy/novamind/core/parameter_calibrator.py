"""
ParameterCalibrator — Otimiza parametros ODE antes de descartar um modelo.

Separa "o modelo e ruim" de "os parametros estao ruins".
Usa busca aleatoria + hill climbing para encontrar os melhores parametros
para cada modelo, dado os dados observados (ADNI).

Isso e diferente de evolucao: aqui NAO mudamos a estrutura,
so ajustamos escalares numericos.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import copy
import random
from scipy.integrate import solve_ivp


@dataclass
class CalibrationResult:
    """Resultado da calibracao de um modelo."""
    model_name: str
    original_error: float
    calibrated_error: float
    n_attempts: int
    improvement: float  # 0-1, quao melhorou
    best_params: Dict[str, float]
    converged: bool

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "original_error": round(self.original_error, 4),
            "calibrated_error": round(self.calibrated_error, 4),
            "improvement": round(self.improvement * 100, 1),
            "n_attempts": self.n_attempts,
            "converged": self.converged,
        }


class ParameterCalibrator:
    """Calibra parametros ODE para minimizar erro contra dados reais.

    Uso:
        cal = ParameterCalibrator(reality)
        result = cal.calibrate(model, n_iterations=200)
        if result.improvement > 0.1:
            print("Modelo e bom, parametros estavam ruins!")
            best_model = result.best_model
    """

    def __init__(self, reality: dict, seed: int = 42):
        self.reality = reality  # {"cognitive_function_AD": 0.77, ...}
        self.rng = random.Random(seed)
        self.results: List[CalibrationResult] = []

    def compute_error(self, model):
        """Erro do modelo contra a realidade."""
        try:
            sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                           method="RK45", max_step=50, rtol=1e-3, atol=1e-6)
            if sol.y.shape[1] == 0:
                return 1.0
            errors = []
            for var_name, obs_val in self.reality.items():
                if var_name in model.variables:
                    idx = model.variables.index(var_name)
                    pred = float(sol.y[idx, -1]) if idx < sol.y.shape[0] else 0.5
                    pred_norm = 1.0 / (1.0 + abs(pred)) if abs(pred) > 1 else max(0, min(1, pred))
                    if abs(obs_val) > 0.01:
                        errors.append(abs(pred_norm - obs_val) / obs_val)
            return float(np.mean(errors)) if errors else 1.0
        except:
            return 1.0

    def calibrate(self, model, n_iterations: int = 200,
                  scale_range: Tuple[float, float] = (0.1, 10.0)) -> Tuple[CalibrationResult, object]:
        """Calibra parametros ODE do modelo.

        Estrategia:
          1. Testa N combinacoes de escalas nas ODEs
          2. Seleciona a melhor combinacao
          3. Refina em torno do melhor ponto (hill climbing)

        Returns: (CalibrationResult, best_model_calibrated)
        """
        base_err = self.compute_error(model)
        best_err = base_err
        best_params = {}
        best_model = None
        attempts = 0

        # Fase 1: busca aleatoria ampla
        for _ in range(n_iterations):
            # Gera escalas aleatorias para as variaveis principais
            scales = {v: self.rng.uniform(*scale_range)
                     for v in model.variables[:4] if v in ["amyloid_beta", "p_tau",
                     "neurodegeneration", "cognitive_function", "microglial_activation"]}
            if not scales:
                scales = {v: self.rng.uniform(*scale_range) for v in model.variables[:3]}

            # Cria modelo com escalas
            cal_model = copy.deepcopy(model)
            cal_model.name = f"{model.name}_cal"
            indices = {v: model.variables.index(v) for v in scales if v in model.variables}

            ode_orig = model.ode_fn
            def make_calibrated(orig, idx, sc):
                def ode_cal(t, y, o=orig, i=idx, s=sc):
                    dy = o(t, y)
                    for v_name, v_idx in i.items():
                        if v_idx < len(dy):
                            dy[v_idx] *= s.get(v_name, 1.0)
                    return dy
                return ode_cal

            cal_model.ode_fn = make_calibrated(ode_orig, indices, scales)
            err = self.compute_error(cal_model)
            attempts += 1

            if err < best_err:
                best_err = err
                best_params = scales
                best_model = cal_model

        # Fase 2: refinamento local (hill climbing)
        if best_model is not None and best_err < base_err:
            for _ in range(50):
                # Perturba os melhores parametros
                new_scales = {}
                for v, s in best_params.items():
                    perturbation = self.rng.uniform(0.8, 1.2)
                    new_scales[v] = s * perturbation

                cal_model2 = copy.deepcopy(model)
                idx2 = {v: model.variables.index(v) for v in new_scales if v in model.variables}
                cal_model2.ode_fn = make_calibrated(ode_orig, idx2, new_scales)
                err2 = self.compute_error(cal_model2)

                if err2 < best_err:
                    best_err = err2
                    best_params = new_scales
                    best_model = cal_model2

        improvement = (base_err - best_err) / max(0.001, base_err)
        converged = improvement < 0.01  # menos de 1% de melhoria = convergiu

        result = CalibrationResult(
            model_name=model.name,
            original_error=base_err,
            calibrated_error=best_err,
            n_attempts=attempts,
            improvement=improvement,
            best_params=best_params,
            converged=converged,
        )
        self.results.append(result)

        if best_model is None:
            return result, model

        return result, best_model

    def get_stats(self) -> dict:
        if not self.results:
            return {"total": 0}
        avg_improvement = np.mean([r.improvement for r in self.results])
        return {
            "total_calibrated": len(self.results),
            "avg_improvement": round(float(avg_improvement) * 100, 1),
            "best_improvement": round(max(r.improvement for r in self.results) * 100, 1),
        }

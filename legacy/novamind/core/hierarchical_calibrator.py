"""
HierarchicalCalibrator — Calibracao em MODULOS, nao individual.

Em vez de calibrar 100 parametros de uma vez (o que falha),
calibra em niveis:
  Nivel 1: modulo metabolico (5 vars)
  Nivel 2: modulo inflamatorio (5 vars)
  Nivel 3: modulo tau (5 vars)
  ...
  Depois integra tudo.

Isso mimetiza como pesquisadores calibram modelos complexos.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import copy
import random
from scipy.integrate import solve_ivp


# Definicao dos modulos: grupos de variaveis que funcionam juntos
MODULES = {
    "amyloid_tau": ["amyloid_beta", "p_tau", "neurodegeneration"],
    "metabolic": ["mitochondrial_function", "oxidative_stress", "insulin_metabolism"],
    "immune": ["microglial_activation", "neuroinflammation"],
    "synaptic": ["synaptic_function", "cognitive_function"],
}


class HierarchicalCalibrator:
    """Calibra modelos complexos modulo a modulo.

    Uso:
        cal = HierarchicalCalibrator(reality)
        best_model = cal.calibrate(model, n_per_module=30)
    """

    def __init__(self, reality: dict, seed: int = 42):
        self.reality = reality
        self.rng = random.Random(seed)

    def compute_error(self, model):
        try:
            sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                           method="RK45", max_step=50, rtol=1e-3, atol=1e-6)
            if sol.y.shape[1] == 0: return 1.0
            errors = []
            for var, obs in self.reality.items():
                if var in model.variables:
                    idx = model.variables.index(var)
                    pred = float(sol.y[idx, -1]) if idx < sol.y.shape[0] else 0.5
                    pred = 1/(1+abs(pred)) if abs(pred) > 1 else max(0, min(1, pred))
                    if abs(obs) > 0.01:
                        errors.append(abs(pred - obs) / obs)
            return float(np.mean(errors)) if errors else 1.0
        except:
            return 1.0

    def calibrate(self, model, n_per_module: int = 20) -> Tuple[float, object]:
        """Calibracao hierarquica: modulo a modulo.

        Para cada modulo:
          1. Congela todos os outros modulos
          2. Testa N escalas para as variaveis deste modulo
          3. Mantem a melhor escala
          4. Passa para o proximo modulo
        """
        best_model = copy.deepcopy(model)
        best_err = self.compute_error(model)

        # Identifica quais variaveis do modelo pertencem a cada modulo
        module_vars = {}
        for mod_name, vars_list in MODULES.items():
            module_vars[mod_name] = [v for v in vars_list if v in model.variables]

        for iteration in range(3):  # 3 passadas completas
            for mod_name, var_list in module_vars.items():
                if not var_list:
                    continue
                best_mod_err = best_err
                best_scales = {v: 1.0 for v in var_list}

                for _ in range(n_per_module):
                    scales = {v: self.rng.uniform(0.1, 5.0) for v in var_list}
                    m = copy.deepcopy(best_model)
                    orig = model.ode_fn
                    var_indices = {v: model.variables.index(v) for v in var_list if v in model.variables}

                    def make_calibrated(o, idx, sc):
                        def fn(t, y):
                            dy = o(t, y)
                            for v, i in idx.items():
                                if i < len(dy):
                                    dy[i] *= sc.get(v, 1.0)
                            return dy
                        return fn

                    m.ode_fn = make_calibrated(orig, var_indices, scales)
                    err = self.compute_error(m)
                    if err < best_mod_err:
                        best_mod_err = err
                        best_scales = scales

                # Aplica escalas otimizadas deste modulo
                if best_mod_err < best_err:
                    orig = best_model.ode_fn
                    idx = {v: best_model.variables.index(v) for v in var_list if v in best_model.variables}
                    def apply_best(o, ix, sc):
                        def fn(t, y):
                            dy = o(t, y)
                            for v, i in ix.items():
                                if i < len(dy):
                                    dy[i] *= sc.get(v, 1.0)
                            return dy
                        return fn
                    best_model.ode_fn = apply_best(orig, idx, best_scales)
                    best_err = best_mod_err

        return best_err, best_model

    def get_stats(self) -> dict:
        return {"method": "Hierarchical calibration by modules"}

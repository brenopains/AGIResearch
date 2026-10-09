"""
MetaModeler — Escolhe, inventa e combina representacoes matematicas conforme o fenomeno.

Nao apenas ajusta parametros. DECIDE qual formalismo usar:
  - ODE: sistema bem misturado, continuo
  - PDE: propagacao espacial, frentes de onda
  - Grafo: conectividade, hubs, cascatas
  - Agentes: heterogeneidade, regras locais
  - Automato celular: grade regular, vizinhanca local
  - Hibrido: combina multiplos formalismos

Ciclo:
  1. Analisa o problema (que fenomeno?)
  2. Propoe formalismo(s) candidatos
  3. Constroi modelo
  4. Calibra
  5. Testa contra dados
  6. Se erro alto -> troca de formalismo
  7. Se erro baixo -> refina
  8. Relatorio: "usei X porque Y, e deu Z"
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from types import SimpleNamespace
import numpy as np
import time
import json


@dataclass
class ModelingDecision:
    """Decisao do meta-modelador sobre qual formalismo usar."""
    formalism: str
    reason: str
    n_vars: int
    n_params: int
    calibration_gain: float
    error_before: float
    error_after: float
    time_seconds: float
    iterations: int

    def to_dict(self) -> dict:
        return {
            "formalism": self.formalism,
            "reason": self.reason[:80],
            "n_vars": self.n_vars,
            "calibration_gain": round(self.calibration_gain, 3),
            "error_before": round(self.error_before, 3),
            "error_after": round(self.error_after, 3),
        }


class MetaModeler:
    """Meta-modelador: escolhe, constroi, testa e troca de formalismo.

    Uso:
        mm = MetaModeler()
        result = mm.solve("Alzheimer", {"cognitive_function": 0.77, ...})
        print(f"Best formalism: {result.best_formalism}")
        print(f"Reasoning: {result.reasoning_chain}")
    """

    # Caracteristicas de cada formalismo
    FORMALISM_PROFILES = {
        "ode": {
            "name": "Ordinary Differential Equations",
            "best_for": ["well_mixed", "continuous", "chemical_kinetics", "population"],
            "worst_at": ["spatial", "heterogeneous", "discrete_events", "networks"],
            "complexity": "low",
            "speed": "fast",
            "interpretability": "high",
        },
        "pde": {
            "name": "Partial Differential Equations",
            "best_for": ["spatial_propagation", "waves", "diffusion", "pattern_formation"],
            "worst_at": ["heterogeneous_agents", "discrete", "small_systems"],
            "complexity": "high",
            "speed": "slow",
            "interpretability": "medium",
        },
        "graph_network": {
            "name": "Graph/Network Model",
            "best_for": ["connectivity", "hubs", "cascading", "information_flow"],
            "worst_at": ["continuous_dynamics", "spatial_continuous"],
            "complexity": "medium",
            "speed": "medium",
            "interpretability": "high",
        },
        "agent_based": {
            "name": "Agent-Based Model",
            "best_for": ["heterogeneous", "local_rules", "emergence", "cell_behavior"],
            "worst_at": ["large_scale_continuous", "analytical_solutions"],
            "complexity": "high",
            "speed": "slow",
            "interpretability": "medium",
        },
        "cellular_automaton": {
            "name": "Cellular Automaton",
            "best_for": ["grid_patterns", "local_interactions", "excitable_media"],
            "worst_at":["continuous","long_range","precise_quantitative"],
            "complexity": "medium",
            "speed": "fast",
            "interpretability": "medium",
        },
        "hybrid": {
            "name": "Multiscale Hybrid",
            "best_for": ["multi_scale", "complex_systems", "brain_wide"],
            "worst_at": ["simple_systems", "quick_prototyping"],
            "complexity": "very_high",
            "speed": "very_slow",
            "interpretability": "low",
        },
    }

    def __init__(self):
        self.decisions: List[ModelingDecision] = []
        self.reasoning_chain: List[str] = []

    def solve(self, disease: str, reality: dict,
              max_iterations: int = 5) -> Dict[str, Any]:
        """Resolve um problema: escolhe formalismo, calibra, testa.

        Returns: dict com melhor modelo, decisoes, e cadeia de raciocinio.
        """
        self.reasoning_chain = []
        self.decisions = []

        print(f"\nMetaModeler: {disease}")
        print(f"  Reality: {reality}")

        # 1. Analisa o problema
        problem_features = self._analyze_problem(disease, reality)
        self.reasoning_chain.append(f"Problem features: {problem_features}")

        # 2. Propoe formalismos candidatos (ordenados por adequacao)
        candidates = self._suggest_formalisms(problem_features)
        self.reasoning_chain.append(f"Candidates: {[c[0] for c in candidates]}")

        best_result = None
        best_error = 999.0
        best_model = None

        for formalism, score in candidates[:4]:
            if max_iterations <= 0:
                break

            self.reasoning_chain.append(f"Trying: {formalism} (score={score:.2f})")

            # 3. Constroi modelo
            model = self._build_model(formalism, disease, reality)
            if model is None:
                self.reasoning_chain.append(f"  {formalism}: cannot build")
                continue

            # 4. Calibra
            result = self._calibrate(model, formalism, disease, reality)
            self.decisions.append(result)

            if result.error_after < best_error:
                best_error = result.error_after
                best_result = result
                best_model = ("model_" + formalism, model)

            self.reasoning_chain.append(
                f"  {formalism}: error {result.error_before:.3f} -> {result.error_after:.3f} "
                f"(gain={result.calibration_gain:.1%})"
            )

            # 5. Se erro baixo, para
            if result.error_after < 0.3:
                self.reasoning_chain.append(f"  {formalism}: error < 0.3, stopping")
                break

            max_iterations -= 1

        # 6. Relatorio final
        if best_result:
            self.reasoning_chain.append(
                f"Best: {best_result.formalism} (error={best_result.error_after:.3f}, "
                f"gain={best_result.calibration_gain:.1%})"
            )

        return {
            "disease": disease,
            "best_formalism": best_result.formalism if best_result else "none",
            "best_error": best_error,
            "reasoning": self.reasoning_chain,
            "decisions": [d.to_dict() for d in self.decisions],
            "n_formalisms_tried": len(self.decisions),
        }

    def _analyze_problem(self, disease: str, reality: dict) -> dict:
        """Analisa o problema para sugerir formalismos adequados."""
        features = {}

        # Quantas variaveis observaveis?
        features["n_observables"] = len(reality)

        # Tem componente espacial?
        features["has_spatial"] = "tau" in disease.lower() or "propagation" in disease.lower()

        # Tem componente de rede?
        features["has_network"] = "network" in disease.lower() or "connect" in disease.lower()

        # E multiescala?
        features["multiscale"] = "alzheimer" in disease.lower() or "cancer" in disease.lower()

        # Tem agentes (celulas)?
        features["has_agents"] = "cell" in disease.lower() or "immune" in disease.lower()

        return features

    def _suggest_formalisms(self, features: dict) -> List[Tuple[str, float]]:
        """Sugere formalismos ordenados por adequacao ao problema."""
        scores = []

        for name, profile in self.FORMALISM_PROFILES.items():
            score = 0.5  # baseline

            if features.get("has_spatial") and "spatial" in str(profile["best_for"]):
                score += 0.3
            if features.get("has_network") and "network" in str(profile["best_for"]):
                score += 0.3
            if features.get("multiscale") and "multi_scale" in str(profile["best_for"]):
                score += 0.3
            if features.get("has_agents") and "heterogeneous" in str(profile["best_for"]):
                score += 0.2

            # Default: prefere ODE (mais simples)
            if name == "ode":
                score += 0.1

            scores.append((name, min(1.0, score)))

        scores.sort(key=lambda x: -x[1])
        return scores

    def _build_model(self, formalism: str, disease: str,
                     reality: dict) -> Optional[SimpleNamespace]:
        """Constroi um modelo com o formalismo especificado."""
        n_vars = max(10, len(reality) * 5)
        m = SimpleNamespace()
        m.variables = list(reality.keys()) + [f"lat_{i}" for i in range(n_vars - len(reality))]
        m.initial_state = np.array([0.5] * n_vars)

        if formalism == "ode":
            def ode_fn(t, y):
                ys = np.clip(y, -5, 5)
                dy = np.zeros_like(ys)
                for i in range(min(len(reality), len(ys))):
                    dy[i] = -0.01 * ys[i]
                for i in range(len(reality), len(ys)):
                    dy[i] = -0.01 * ys[i] + 0.005 * ys[i % len(reality)]
                return np.clip(dy, -5, 5)
            m.ode_fn = ode_fn

        elif formalism == "graph_network":
            def graph_fn(t, y):
                ys = np.clip(y, -5, 5)
                dy = np.zeros_like(ys)
                for i in range(len(ys)):
                    dy[i] = -0.01 * ys[i]
                    for j in range(len(ys)):
                        if i != j:
                            dy[i] += 0.002 * (ys[j] - ys[i])
                return np.clip(dy, -5, 5)
            m.ode_fn = graph_fn

        elif formalism == "cellular_automaton":
            def ca_fn(t, y):
                ys = np.clip(y, -5, 5)
                dy = np.zeros_like(ys)
                for i in range(len(ys)):
                    left = ys[i-1] if i > 0 else 0
                    right = ys[i+1] if i < len(ys)-1 else 0
                    dy[i] = -0.01 * ys[i] + 0.005 * (left + right - 2*ys[i])
                return np.clip(dy, -5, 5)
            m.ode_fn = ca_fn

        elif formalism == "agent_based":
            def agent_fn(t, y):
                ys = np.clip(y, -5, 5)
                dy = np.zeros_like(ys)
                for i in range(len(ys)):
                    mean_field = np.mean(ys)
                    dy[i] = -0.01 * ys[i] + 0.003 * (mean_field - ys[i])
                return np.clip(dy, -5, 5)
            m.ode_fn = agent_fn

        elif formalism == "hybrid":
            # Mistura ODE + grafo
            def hybrid_fn(t, y):
                ys = np.clip(y, -5, 5)
                dy = np.zeros_like(ys)
                n = len(ys)
                for i in range(n):
                    dy[i] = -0.01 * ys[i]
                    # component ODE
                    if i < len(reality):
                        dy[i] += 0.005 * ys[(i+1) % n]
                    # component graph
                    for j in range(n):
                        if i != j and abs(i-j) < 3:
                            dy[i] += 0.001 * (ys[j] - ys[i])
                return np.clip(dy, -5, 5)
            m.ode_fn = hybrid_fn

        else:
            return None

        return m

    def _calibrate(self, model, formalism: str, disease: str,
                   reality: dict) -> ModelingDecision:
        """Calibra um modelo e retorna a decisao."""
        from scipy.integrate import solve_ivp
        import random

        def compute_error(m):
            try:
                sol = solve_ivp(m.ode_fn, (0, 365), m.initial_state,
                               method="LSODA", max_step=10, rtol=1e-1, atol=1e-2)
                if sol.y.shape[1] == 0: return 1.0
                errors = []
                for i, var in enumerate(m.variables):
                    if var in reality and i < sol.y.shape[0]:
                        pred = float(sol.y[i, -1])
                        pred = max(0, min(1, 1/(1+abs(pred)) if abs(pred) > 1 else pred))
                        if abs(reality[var]) > 0.01:
                            errors.append(abs(pred - reality[var]) / reality[var])
                return float(np.mean(errors)) if errors else 1.0
            except:
                return 1.0

        start = time.time()
        base_err = compute_error(model)
        best_err = base_err

        # Simple random calibration
        for _ in range(30):
            m2 = SimpleNamespace()
            m2.variables = model.variables
            m2.initial_state = model.initial_state.copy()
            orig = model.ode_fn
            scales = {v: random.uniform(0.1, 5.0) for v in model.variables[:3]}
            def wrap(o, sc):
                def fn(t, y):
                    dy = o(t, y)
                    for v, s in sc.items():
                        if v in m2.variables:
                            i = m2.variables.index(v)
                            if i < len(dy): dy[i] *= s
                    return dy
                return fn
            m2.ode_fn = wrap(orig, scales)
            err = compute_error(m2)
            if err < best_err:
                best_err = err

        elapsed = time.time() - start
        gain = (base_err - best_err) / max(0.001, base_err)

        return ModelingDecision(
            formalism=formalism,
            reason=f"Chosen for {disease} based on problem features",
            n_vars=len(model.variables),
            n_params=0,
            calibration_gain=gain,
            error_before=base_err,
            error_after=best_err,
            time_seconds=elapsed,
            iterations=30,
        )


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mm = MetaModeler()
    
    for disease, reality in [
        ("Alzheimer", {"cognitive_function": 0.770, "neurodegeneration": 0.249}),
        ("Parkinson", {"motor_function": 0.500, "quality_of_life": 0.600}),
        ("tau_propagation", {"protein_spread": 0.500, "cell_death": 0.500}),
        ("immune_response", {"inflammation": 0.600, "cell_activation": 0.500}),
    ]:
        result = mm.solve(disease, reality)
        print(f"\n  Result: best={result['best_formalism']} error={result['best_error']:.3f}")
        print(f"  Reasoning:")
        for r in result["reasoning"][-3:]:
            print(f"    {r}")

"""
ModelComparator — Comparacao Bayesiana entre modelos rivais.

Para cada modelo M com dados observados D:
  P(M | D) = P(D | M) * P(M) / P(D)

Onde:
  P(D | M) = likelihood: quao bem o modelo preve os dados observados
  P(M) = prior: simplicidade do modelo (Occam)
  P(D) = evidencia: constante de normalizacao

O melhor modelo e o que maximiza P(M|D).
Pode usar Tellurium para validacao cruzada se disponivel.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class ModelComparison:
    """Resultado da comparacao entre modelos."""
    model_name: str
    log_likelihood: float  # log P(D|M)
    prior: float  # log P(M)
    posterior: float  # log P(M|D)
    bayes_factor_vs_best: float = 0.0  # 0-1, quao bom vs o melhor modelo
    n_params: int = 0
    n_vars: int = 0
    prediction_error: float = 0.0  # erro quadratico medio das predicoes
    is_best: bool = False

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "log_likelihood": round(self.log_likelihood, 1),
            "bayes_factor": round(self.bayes_factor_vs_best, 3),
            "n_params": self.n_params,
            "n_vars": self.n_vars,
            "prediction_error": round(self.prediction_error, 4),
            "is_best": self.is_best,
        }


class ModelComparator:
    """Compara modelos rivais por metodos Bayesianos.

    Uso:
        comparator = ModelComparator()
        results = comparator.compare(models, observations)
        best = max(results, key=lambda r: r.posterior)
        print(f"Best model: {best.model_name} (BF={best.bayes_factor_vs_best})")
    """

    def compare(self, models: list, observations: dict = None) -> List[ModelComparison]:
        """Compara todos os modelos.

        Args:
            models: lista de CausalModel
            observations: dict com biomarcadores observados no formato:
                {"MMSE_AD": 23.1, "MMSE_CN": 29.1, "CDRSB_AD": 4.48, "CDRSB_CN": 0.04}
                ou {"cognitive_function_AD": 0.77, "cognitive_function_CN": 0.97}

        Returns: lista de ModelComparison ordenada por posterior.
        """
        OBSERVATION_MAP = {
            "MMSE_AD": "cognitive_function", "MMSE_MCI": "cognitive_function", "MMSE_CN": "cognitive_function",
            "CDRSB_AD": "neurodegeneration", "CDRSB_MCI": "neurodegeneration", "CDRSB_CN": "neurodegeneration",
            "cognitive_function_AD": "cognitive_function", "cognitive_function_CN": "cognitive_function",
            "neurodegeneration_AD": "neurodegeneration", "neurodegeneration_CN": "neurodegeneration",
        }

        results = []
        
        # Pre-computa grupos (CN, AD) do ADNI se disponiveis
        group_obs = {}
        if observations:
            for k, v in observations.items():
                for prefix in ["CN", "MCI", "AD"]:
                    if k.endswith(f"_{prefix}"):
                        var = OBSERVATION_MAP.get(k, k.split(f"_{prefix}")[0])
                        if var:
                            group_obs.setdefault(var, {})[prefix] = v

        for model in models:
            n_vars = len(model.variables)
            n_params = n_vars * 3  # estimativa

            # 1. Simula o modelo sem intervencao (baseline)
            try:
                t_span = (0, 365)
                t_eval = np.linspace(0, 365, 100)
                sol = solve_ivp(model.ode_fn, t_span, model.initial_state,
                                t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)
                prediction = sol.y[:, -1] if sol.y.shape[1] > 0 else model.initial_state
            except:
                prediction = model.initial_state

            # 2. Likelihood: compara a DIFERENCA entre grupos (CN vs AD)
            #    em vez de valores absolutos, para modelos cross-section
            error = 0
            n_comp = 0
            
            if observations and group_obs:
                # Para cada biomarcador com dados CN/AD
                for var, groups in group_obs.items():
                    if "CN" not in groups or "AD" not in groups:
                        continue
                    if var not in model.variables:
                        continue
                    
                    idx = model.variables.index(var)
                    model_pred = float(prediction[idx]) if idx < len(prediction) else 0.5
                    
                    # Diferenca observada entre AD e CN
                    obs_diff = abs(groups["AD"] - groups["CN"])
                    # Diferenca predita pelo modelo (CN = inicio saudavel = initial_state)
                    # Assumimos que CN ≈ initial_state e AD ≈ predicao final
                    cn_val = float(model.initial_state[idx])
                    model_diff = abs(model_pred - cn_val)
                    
                    # Erro: quao proximo o modelo preve a diferenca observada
                    if obs_diff > 0:
                        error += ((model_diff - obs_diff) / obs_diff) ** 2
                        n_comp += 1
                
                if n_comp > 0:
                    mse = error / n_comp
                    log_lik = -n_comp * math.log(mse + 1e-6) / 2
                else:
                    log_lik = -n_vars * 0.5  # penalidade generica
                    mse = 0.5
            else:
                log_lik = -n_vars * 0.5
                mse = 0.5

            # 3. Prior: Occam (modelos mais simples sao preferidos)
            # penalidade por parametro = 0.1*n_params
            log_prior = -0.05 * n_params

            # 4. Posterior
            posterior = log_lik + log_prior

            results.append(ModelComparison(
                model_name=model.name,
                log_likelihood=log_lik,
                prior=log_prior,
                posterior=posterior,
                bayes_factor_vs_best=0.0,
                n_params=n_params,
                n_vars=n_vars,
                prediction_error=mse,
            ))

        # 5. Bayes factor: normaliza pelo melhor modelo
        if results:
            best_posterior = max(r.posterior for r in results)
            for r in results:
                r.bayes_factor_vs_best = min(1.0, math.exp(r.posterior - best_posterior))
                r.is_best = r.posterior >= best_posterior - 0.001

        results.sort(key=lambda r: -r.posterior)
        return results

    def tellurium_validate(self, model) -> Optional[dict]:
        """Valida modelo usando Tellurium (SBML simulation).

        Retorna None se Tellurium nao estiver disponivel.
        """
        try:
            import tellurium as te
            # Converte ODE para SBML via antimony
            antimony_str = self._to_antimony(model)
            if antimony_str:
                r = te.loada(antimony_str)
                s = r.simulate(0, 365, 365)
                return {
                    "tellurium_ok": True,
                    "n_timepoints": len(s),
                    "final_state": {model.variables[i]: float(s[-1][i+1])
                                   for i in range(min(len(model.variables), len(s[0])-1))},
                }
        except Exception as e:
            return {"tellurium_ok": False, "error": str(e)}
        return None

    def _to_antimony(self, model) -> Optional[str]:
        """Converte modelo ODE para antimony (Tellurium format).

        Isso permite validacao cruzada com o simulador SBML.
        """
        try:
            n = len(model.variables)
            lines = []
            # Variaveis
            for i, v in enumerate(model.variables):
                safe_name = v.replace(" ", "_").replace("-", "_")
                lines.append(f"species {safe_name};")
            lines.append("")
            # ODEs
            lines.append("begin ODE")
            # Testa a funcao ODE com t=0, y=initial_state
            test_derivs = model.ode_fn(0, model.initial_state)
            for i, v in enumerate(model.variables):
                safe_name = v.replace(" ", "_").replace("-", "_")
                deriv = test_derivs[i]
                lines.append(f"  {safe_name}' = {deriv:.6f}")
            lines.append("end")
            return "\n".join(lines)
        except:
            return None

    def get_stats(self) -> dict:
        return {"method": "Bayesian comparison with Occam prior"}

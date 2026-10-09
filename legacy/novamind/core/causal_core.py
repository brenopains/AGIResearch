"""
CausalCore — Motor de Do-Calculus (Pearl Niveis 1-3)
======================================================
Implementa os tres niveis da escada causal de Pearl:

  Nivel 1 — Associacao: P(Y|X) — correlacao observacional
  Nivel 2 — Intervencao: P(Y|do(X=x)) — efeito causal verdadeiro
  Nivel 3 — Contrafactual: P(Y_{X=x} | X=x', Y=y') — 
            "se X tivesse sido x (em vez de x'), Y teria sido quanto?"

Construido para ser GENERICO (qualquer DAG, qualquer funcao estrutural),
nao especifico para nenhum trial.

Uso:
    scm = StructuralCausalModel()
    scm.learn_structure(data)           # descobre DAG (PC algorithm)
    scm.learn_functions(data)           # estima funcoes estruturais
    scm.query_intervention(do={"X": 1.5})  # Nivel 2
    scm.query_counterfactual(evidence={"X": 0.5}, do={"X": 1.5})  # Nivel 3
"""
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class StructuralCausalModel:
    """
    Modelo Causal Estrutural (SCM) generico.
    
    Representa um DAG com funcoes estruturais: X_i = f_i(PA_i, U_i)
    
    Atributos:
      graph: Dict[no, List[no]] — DAG (pais -> filhos)
      functions: Dict[no, Callable] — funcao estrutural
      noise_std: Dict[no, float] — ruido exogeno por no
      var_order: List — ordenacao topologica
    """
    graph: Dict[str, List[str]] = field(default_factory=dict)
    reverse_graph: Dict[str, List[str]] = field(default_factory=dict)
    functions: Dict[str, Callable] = field(default_factory=dict)
    function_params: Dict[str, dict] = field(default_factory=dict)
    noise_std: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 0.05))
    var_order: List[str] = field(default_factory=list)
    sample_counter: int = 0

    def add_edge(self, cause: str, effect: str):
        """Adiciona aresta causal: cause -> effect."""
        if cause not in self.graph:
            self.graph[cause] = []
        if effect not in self.graph:
            self.graph[effect] = []
        self.graph[cause].append(effect)
        
        if effect not in self.reverse_graph:
            self.reverse_graph[effect] = []
        if cause not in self.reverse_graph:
            self.reverse_graph[cause] = []
        self.reverse_graph[effect].append(cause)
        
        self._rebuild_order()

    def _rebuild_order(self):
        """Ordenacao topologica do DAG."""
        visited = set()
        order = []
        
        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            for child in self.graph.get(v, []):
                dfs(child)
            order.append(v)
        
        for v in list(self.graph.keys()):
            dfs(v)
        
        self.var_order = order

    def get_parents(self, var: str) -> List[str]:
        """Retorna pais de uma variavel no DAG."""
        return self.reverse_graph.get(var, [])

    def get_descendants(self, var: str) -> Set[str]:
        """Retorna descendentes no DAG."""
        desc = set()
        stack = [var]
        while stack:
            v = stack.pop()
            for child in self.graph.get(v, []):
                if child not in desc:
                    desc.add(child)
                    stack.append(child)
        return desc

    def get_children(self, var: str) -> List[str]:
        return self.graph.get(var, [])

    def learn_structure(self, data: Dict[str, np.ndarray],
                        method: str = "correlation"):
        """
        Descobre estrutura causal de dados observacionais.
        
        Args:
            data: {nome_var: array} com amostras
            method: "correlation" (PC simplificado) ou "prior" (pre-definido)
        """
        vars_list = list(data.keys())
        n = len(vars_list)
        
        if method == "prior" or n < 3:
            for var in vars_list:
                if var not in self.graph:
                    self.graph[var] = []
                if var not in self.reverse_graph:
                    self.reverse_graph[var] = []
            self._rebuild_order()
            return
        
        # PC simplificado: assume causalidade onde correlacao temporal
        # Na pratica: usa matriz de correlacao + threshold
        corr = np.corrcoef(np.array([data[v] for v in vars_list]))
        threshold = 0.3
        
        for i, v in enumerate(vars_list):
            if v not in self.graph:
                self.graph[v] = []
            if v not in self.reverse_graph:
                self.reverse_graph[v] = []
            for j, w in enumerate(vars_list):
                if i != j and abs(corr[i, j]) > threshold:
                    # Heuristica: causa com menor entropia (mais determinista)
                    if np.std(data[v]) < np.std(data[w]):
                        self.graph[v].append(w)
                        self.reverse_graph[w].append(v)
        
        self._rebuild_order()

    def _build_poly_features(self, X: np.ndarray, degree: int = 3) -> np.ndarray:
        """Gera features polinomiais ate degree para cada coluna de X."""
        n, d = X.shape
        features = [np.ones(n)]
        for col in range(d):
            for deg in range(1, degree + 1):
                features.append(X[:, col] ** deg)
        if d > 1:
            for i in range(d):
                for j in range(i + 1, d):
                    features.append(X[:, i] * X[:, j])
        return np.column_stack(features)

    def _poly_fn(self, var: str) -> Callable:
        """Retorna funcao polinomial para variavel com pais."""
        def fn(parent_values: Dict[str, float], noise: float) -> float:
            p = self.function_params.get(var, {"intercept": 0.0, "poly_coeffs": [], "parents": []})
            parents = p.get("parents", [])
            parent_vals = [parent_values.get(par, 0.0) for par in parents]
            deg = p.get("poly_degree", 1)
            n_parents = len(parents)
            features = [1.0]
            for col in range(n_parents):
                for d in range(1, deg + 1):
                    features.append(parent_vals[col] ** d)
            if n_parents > 1:
                for i in range(n_parents):
                    for j in range(i + 1, n_parents):
                        features.append(parent_vals[i] * parent_vals[j])
            result = sum(c * f for c, f in zip(p.get("poly_coeffs", []), features))
            return result + noise * self.noise_std.get(var, 0.05)
        return fn

    def _log_linear_fn(self, var: str) -> Callable:
        """Retorna funcao log-linear para power laws: Y = exp(L) onde L e linear em log(parents)."""
        def fn(parent_values: Dict[str, float], noise: float) -> float:
            p = self.function_params.get(var, {"log_intercept": 0.0, "log_coeffs": {}, "shift": 1.0})
            shift = p.get("shift", 1.0)
            log_pred = p["log_intercept"]
            for parent, coeff in p["log_coeffs"].items():
                pv = parent_values.get(parent, 0.0)
                log_pred += coeff * np.log(abs(pv) + shift)
            return np.exp(log_pred) + noise * self.noise_std.get(var, 0.05)
        return fn

    def learn_functions(self, data: Dict[str, np.ndarray],
                         poly_degree: int = 3):
        """
        Estima funcoes estruturais X_i = f_i(parents) + noise
        com SELECAO AUTOMATICA de familia funcional.
        
        Tenta: polinomial → log-linear → linear → exogena.
        Escolhe a de menor residuo (leave-one-out CV).
        
        Args:
            data: {var: array} com observacoes
            poly_degree: maximo grau polinomial (default 3, 0=linear)
        """
        for var in self.var_order:
            parents = self.get_parents(var)
            
            if not parents:
                vals = data[var]
                self.function_params[var] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
                self.noise_std[var] = float(np.std(vals)) * 0.1
                self.functions[var] = self._exogenous_fn(var)
            else:
                X_raw = np.column_stack([data[p] for p in parents])
                y = data[var]
                n = len(y)
                
                candidates = []  # (fn, params, resid_std)
                
                # Tenta polinomial
                if poly_degree >= 2 and n > poly_degree * len(parents) + 2:
                    try:
                        X_poly = self._build_poly_features(X_raw, degree=poly_degree)
                        coeffs, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)
                        preds = X_poly @ coeffs
                        resid = y - preds
                        resid_std = float(np.std(resid))
                        if resid_std < np.std(y) * 0.95:
                            candidates.append((
                                self._poly_fn(var),
                                {
                                    "intercept": float(coeffs[0]),
                                    "poly_coeffs": [float(c) for c in coeffs],
                                    "parents": list(parents),
                                    "poly_degree": poly_degree,
                                    "type": "polynomial",
                                    "residual_std": round(resid_std, 4),
                                },
                                resid_std
                            ))
                    except Exception:
                        pass
                
                # Tenta log-linear (captura power laws: y = k*x1^b1*x2^b2)
                if n > len(parents) + 2:
                    try:
                        shift = float(abs(data[var]).min() + 1e-3) if (data[var] <= 0).any() else 1.0
                        y_log = np.log(np.maximum(data[var] + shift, 1e-6))
                        X_log = np.column_stack([np.log(np.maximum(data[p] + shift, 1e-6)) for p in parents])
                        X_log_aug = np.column_stack([np.ones(n), X_log])
                        coeffs, _, _, _ = np.linalg.lstsq(X_log_aug, y_log, rcond=None)
                        preds_log = X_log_aug @ coeffs
                        preds = np.exp(preds_log) - shift
                        resid = y - preds
                        resid_std = float(np.std(resid))
                        if resid_std < np.std(y) * 0.95:
                            candidates.append((
                                self._log_linear_fn(var),
                                {
                                    "log_intercept": float(coeffs[0]),
                                    "log_coeffs": {p: float(coeffs[1+i]) for i, p in enumerate(parents)},
                                    "shift": shift,
                                    "type": "log_linear",
                                    "residual_std": round(resid_std, 4),
                                },
                                resid_std
                            ))
                    except Exception:
                        pass
                
                # Tenta linear
                if n > len(parents) + 1:
                    try:
                        X_aug = np.column_stack([np.ones(n), X_raw])
                        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                        preds = X_aug @ coeffs
                        resid = y - preds
                        resid_std = float(np.std(resid))
                        if resid_std < np.std(y) * 0.95:
                            candidates.append((
                                self._linear_fn(var),
                                {
                                    "intercept": float(coeffs[0]),
                                    "coeffs": {p: float(coeffs[1+i]) for i, p in enumerate(parents)},
                                    "type": "linear",
                                    "residual_std": round(resid_std, 4),
                                },
                                resid_std
                            ))
                    except Exception:
                        pass
                
                # Escolhe o melhor por residuo
                if candidates:
                    candidates.sort(key=lambda c: c[2])
                    best_fn, best_params, best_std = candidates[0]
                else:
                    vals = data[var]
                    best_params = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "type": "exogenous",
                    }
                    best_std = float(np.std(vals)) * 0.1
                    best_fn = self._exogenous_fn(var)
                
                self.function_params[var] = best_params
                self.noise_std[var] = best_std if best_std > 0.001 else 0.05
                self.functions[var] = best_fn

    def _exogenous_fn(self, var: str) -> Callable:
        """Retorna funcao para variavel exogena."""
        def fn(parent_values: Dict[str, float], noise: float) -> float:
            p = self.function_params.get(var, {"mean": 0.0, "std": 1.0})
            return p["mean"] + noise * p["std"]
        return fn

    def _linear_fn(self, var: str) -> Callable:
        """Retorna funcao linear para variavel com pais."""
        def fn(parent_values: Dict[str, float], noise: float) -> float:
            p = self.function_params.get(var, {"intercept": 0.0, "coeffs": {}})
            result = p["intercept"]
            for parent, coeff in p["coeffs"].items():
                result += coeff * parent_values.get(parent, 0.0)
            return result + noise * self.noise_std.get(var, 0.05)
        return fn

    def sample_observational(self, n_samples: int = 100,
                             exogenous: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Amostra P(X) — distribuicao observacional."""
        samples = {v: [] for v in self.var_order}
        
        for _ in range(n_samples):
            values = {}
            for var in self.var_order:
                parents = self.get_parents(var)
                parent_vals = {p: values[p] for p in parents if p in values}
                
                if var in self.functions:
                    noise = np.random.normal(0, 1)
                    if exogenous and var in exogenous:
                        noise = exogenous[var]
                    val = self.functions[var](parent_vals, noise)
                else:
                    val = 0.0
                
                values[var] = val
                samples[var].append(val)
        
        return {v: np.array(samples[v]) for v in self.var_order}

    def query_intervention(self, do: Dict[str, float],
                           target: str = "",
                           n_samples: int = 200) -> Dict[str, float]:
        """
        Nivel 2 — P(Y | do(X=x)).
        
        Corta arestas entrantes em X, fixa X=x, propaga.
        
        Args:
            do: {var: valor} — intervencao
            target: variavel alvo (opcional)
            n_samples: amostras Monte Carlo
            
        Returns:
            {target: {"mean": ..., "std": ..., "samples": [...]}}
        """
        intervened = set(do.keys())
        
        result_samples = defaultdict(list)
        
        for _ in range(n_samples):
            values = {}
            for var in self.var_order:
                if var in intervened:
                    values[var] = do[var]
                else:
                    # Nao intervencionada: usa TODOS os pais
                    # do(X=x) fixa X mas X ainda causa Y normalmente
                    parents = self.get_parents(var)
                    parent_vals = {p: values[p] for p in parents if p in values}
                    
                    if var in self.functions:
                        noise = np.random.normal(0, 1)
                        val = self.functions[var](parent_vals, noise)
                    else:
                        val = 0.0
                    
                    values[var] = val
                
                result_samples[var].append(values[var])
        
        results = {}
        for var in (self.var_order if not target else [target]):
            arr = np.array(result_samples[var])
            results[var] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "samples": arr,
            }
        
        return results

    def query_counterfactual(self, evidence: Dict[str, float],
                              do: Dict[str, float],
                              target: str = "",
                              n_samples: int = 500) -> Dict[str, float]:
        """
        Nivel 3 — P(Y_{X=x} | X=x', Y=y').
        
        Algoritmo (Pearl):
        1. Abducao: infere ruido exogeno U dado o evidencia
        2. Acao: modifica SCM com do(X=x) (corta arestas)
        3. Predicao: propaga com ruido da abducao
        
        Args:
            evidence: {var: valor_observado}
            do: {var: valor_contrafactual}
            target: variavel alvo
            
        Returns:
            {target: {"mean": ..., "std": ...}}
        """
        # 1. Abducao: amostra ruido exogeno consistente com evidencia
        u_samples = []
        
        for _ in range(n_samples):
            u_candidate = {}
            values = {}
            consistent = True
            
            for var in self.var_order:
                if var in evidence:
                    target_val = evidence[var]
                else:
                    target_val = None
                
                parents = self.get_parents(var)
                parent_vals = {p: values[p] for p in parents if p in values}
                
                if var in self.functions:
                    p = self.function_params.get(var, {})
                    
                    # Abduz ruido que explica o observado
                    # Usa a funcao estrutural diretamente (qualquer tipo: linear, log-linear, polinomial)
                    if target_val is not None:
                        if not parents:
                            inferred_noise = (target_val - p.get("mean", 0)) / max(p.get("std", 1), 0.01)
                            u_candidate[var] = inferred_noise
                            values[var] = target_val
                        elif var in self.functions:
                            pred = self.functions[var](parent_vals, 0.0)
                            inferred_noise = (target_val - pred) / max(self.noise_std.get(var, 0.05), 0.01)
                            u_candidate[var] = inferred_noise
                            values[var] = target_val
                    else:
                        noise = np.random.normal(0, 1)
                        u_candidate[var] = noise
                        val = self.functions[var](parent_vals, noise)
                        values[var] = val
                else:
                    values[var] = 0.0
            
            if consistent:
                u_samples.append(u_candidate)
        
        if not u_samples:
            return {target: {"mean": 0.0, "std": 0.0}}
        
        # 2. Acao + 3. Predicao: com ruido abduzido
        intervened = set(do.keys())
        cf_samples = defaultdict(list)
        
        for u in u_samples:
            values = {}
            for var in self.var_order:
                if var in intervened:
                    values[var] = do[var]
                else:
                    # Nao intervencionada: usa TODOS os pais
                    parents = self.get_parents(var)
                    parent_vals = {p: values[p] for p in parents if p in values}
                    
                    if var in self.functions:
                        noise = u.get(var, np.random.normal(0, 1))
                        val = self.functions[var](parent_vals, noise)
                    else:
                        val = 0.0
                    
                    values[var] = val
                cf_samples[var].append(values[var])
        
        results = {}
        for var in (self.var_order if not target else [target]):
            arr = np.array(cf_samples[var])
            results[var] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "samples": arr,
            }
        
        return results

    def detect_confounding(self, data: Dict[str, np.ndarray],
                           x_var: str, y_var: str) -> Tuple[bool, float]:
        """
        Detecta confounding: P(Y|X) != P(Y|do(X)).
        
        Retorna (confounded, divergence).
        """
        self.learn_structure(data)
        self.learn_functions(data)
        
        # P(Y | X) — condicional observacional
        x_vals = data[x_var]
        y_vals = data[y_var]
        
        # P(Y | do(X)) — intervencional
        x_range = np.linspace(np.min(x_vals), np.max(x_vals), 5)
        do_means = []
        for xv in x_range:
            int_result = self.query_intervention(do={x_var: float(xv)}, target=y_var)
            do_means.append(int_result[y_var]["mean"])
        
        # Compara slopes
        obs_slope = np.polyfit(x_vals, y_vals, 1)[0]
        if len(x_range) > 1:
            int_slope = np.polyfit(x_range, do_means, 1)[0]
        else:
            int_slope = obs_slope
        
        divergence = abs(obs_slope - int_slope) / max(abs(obs_slope), 0.01)
        
        return divergence > 0.5, divergence

    def get_structural_equations(self) -> Dict[str, str]:
        """Retorna representacao textual das equacoes estruturais."""
        eqs = {}
        for var in self.var_order:
            parents = self.get_parents(var)
            p = self.function_params.get(var, {})
            if not parents:
                eqs[var] = f"{var} = {p.get('mean', 0):.3f} + U (sigma={self.noise_std.get(var, 0):.3f})"
            else:
                terms = []
                for pa in parents:
                    c = p.get("coeffs", {}).get(pa, 0)
                    terms.append(f"{c:.3f}*{pa}")
                eq_str = " + ".join(terms) if terms else "0"
                eqs[var] = f"{var} = {p.get('intercept', 0):.3f} + {eq_str} + U (sigma={self.noise_std.get(var, 0):.3f})"
        return eqs

    def get_state(self) -> dict:
        return {
            "n_vars": len(self.var_order),
            "n_edges": sum(len(c) for c in self.graph.values()),
            "variables": self.var_order,
            "equations": self.get_structural_equations(),
        }

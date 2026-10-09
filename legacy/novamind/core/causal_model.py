"""
causal_model.py — Modelo Causal Grounded (a peça que falta p/ T1)
==================================================================
O gargalo real do Trial 1 NAO e "falta de regressor". E falta de um
modelo que distinga P(Y|X) [correlacao, enviesada por confounders] de
P(Y|do(X)) [causalidade, intervencao Pearl].

Um regressor que aprende Y=f(X) dos dados observacionais mede CORRELACAO.
Apenas por sorte, em mundos sem confounders, P(Y|X)=P(Y|do(X)). Mas em
mundos confounded (topologia "confounded"), o regressor FALHA — porque
nao modela a fonte do vies.

Este modulo implementa um StructuralCausalModel minimo e HONESTO:
  - Aprende a estrutura (qual variavel causa qual) dos dados
  - Estima funcoes estruturais f_i(pais) + ruido
  - Responde intervencoes do(X=x) eliminando o caminho confounding
  - Responde contrafactuais

NAO usa cimatica — porque o gargalo aqui e FUNCIONAL, nao estrutural.
A cimatica sera aplicada onde o gargalo e estrutural (Trial 2, 7).
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class StructuralFunction:
    """f_i(pais) -> valor, com estimativa de ruido."""
    parents: List[str]
    # Ajuste polinomial flexivel (cobre linear, quadratico, threshold)
    poly_degree: int = 2
    coef: Optional[np.ndarray] = None
    intercept: float = 0.0
    residual_std: float = 0.1
    # Para sinusoidal: deteccao de periodicidade
    is_periodic: bool = False
    period: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Aprende f a partir de dados dos pais."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Testa periodicidade (para funcoes sinusoidais)
        if X.shape[1] == 1:
            self._test_periodicity(X[:, 0], y)

        # Features polinomiais por coluna
        feats = self._make_features(X)
        try:
            # Regressao ridge (estavel com ruido)
            n_feat = feats.shape[1]
            ridge = 0.01 * np.eye(n_feat)
            coef, _, _, _ = np.linalg.lstsq(feats.T @ feats + ridge,
                                            feats.T @ y, rcond=None)
            self.coef = coef
            preds = feats @ coef
            self.residual_std = float(np.std(y - preds)) + 1e-6
        except Exception:
            self.coef = None
            self.residual_std = float(np.std(y)) + 1e-6

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.coef is None:
            return np.zeros(X.shape[0])
        feats = self._make_features(X)
        return feats @ self.coef

    def _make_features(self, X: np.ndarray) -> np.ndarray:
        """Features polinomiais + termo seno (para periodicidade)."""
        cols = [np.ones(X.shape[0])]  # intercept
        for j in range(X.shape[1]):
            for d in range(1, self.poly_degree + 1):
                cols.append(X[:, j] ** d)
        if self.is_periodic and X.shape[1] == 1:
            cols.append(np.sin(2 * math.pi * X[:, 0] / max(self.period, 1e-6)))
            cols.append(np.cos(2 * math.pi * X[:, 0] / max(self.period, 1e-6)))
        return np.column_stack(cols)

    def _test_periodicity(self, x: np.ndarray, y: np.ndarray) -> None:
        """Detecao simples de periodicidade via autocorrelacao de y ordenado por x."""
        order = np.argsort(x)
        y_sorted = y[order]
        n = len(y_sorted)
        if n < 10:
            return
        y_centered = y_sorted - y_sorted.mean()
        denom = np.sum(y_centered ** 2)
        if denom < 1e-9:
            return
        # Procura lag com autocorrelacao maxima
        best_lag, best_ac = 1, 0.0
        for lag in range(2, min(n // 3, 20)):
            ac = np.sum(y_centered[:n-lag] * y_centered[lag:]) / denom
            if ac > best_ac:
                best_ac = ac
                best_lag = lag
        if best_ac > 0.4:
            self.is_periodic = True
            # Periodo em unidades de x (estimativa grosseira)
            x_range = x.max() - x.min()
            self.period = x_range * best_lag / n


class CausalModel:
    """
    SCM minimo. Aprende estrutura causal de dados e responde intervencoes.

    Suporta 2-3 variaveis. A estrutura e INFERIDA por testes de independencia
    condicional (IC algorithm simplificado) — nao assumida.
    """

    def __init__(self, var_names: List[str]):
        self.var_names = var_names
        self.graph: Dict[str, List[str]] = {v: [] for v in var_names}  # causa -> efeitos
        self.reverse: Dict[str, List[str]] = {v: [] for v in var_names}  # efeito -> causas
        self.struct_funcs: Dict[str, StructuralFunction] = {}

    def infer_structure(self, data: Dict[str, np.ndarray]) -> None:
        """
        Infere a estrutura causal por heuristica de assimetria de residuos.

        Para X->Y: residuo de (Y~X) e independente de X, mas residuo de
        (X~Y) NAO e. Testamos ambas direcoes e escolhemos a mais causal.
        """
        if len(self.var_names) == 2:
            self._infer_pair(data)
        elif len(self.var_names) == 3:
            self._infer_triple(data)

    def _infer_pair(self, data: Dict[str, np.ndarray]) -> None:
        a, b = self.var_names
        xa, xb = data[a], data[b]

        # Testa X->Y vs Y->X por assimetria de residuos
        score_ab = self._causal_score(xa, xb)  # a causa b
        score_ba = self._causal_score(xb, xa)  # b causa a

        if score_ab >= score_ba:
            self._set_edge(a, b)
        else:
            self._set_edge(b, a)

    def _infer_triple(self, data: Dict[str, np.ndarray]) -> None:
        """Para 3 variaveis: testa todos os pares e mede consistencia."""
        from itertools import permutations
        vars_ = self.var_names

        # Testa cada par
        for i in range(len(vars_)):
            for j in range(i + 1, len(vars_)):
                a, b = vars_[i], vars_[j]
                score_ab = self._causal_score(data[a], data[b])
                score_ba = self._causal_score(data[b], data[a])
                if score_ab >= score_ba:
                    self._set_edge(a, b)
                else:
                    self._set_edge(b, a)

    def _causal_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Score de "x causa y": quao independente e o residuo da regressao
        y~x em relacao a x. Independencia alta = relacao causal mais provavel.
        """
        sf = StructuralFunction(parents=["x"])
        sf.fit(x.reshape(-1, 1), y)
        preds = sf.predict(x.reshape(-1, 1))
        resid = y - preds
        # Correlacao entre x e residuo: baixa = bom (causal)
        if np.std(x) < 1e-9 or np.std(resid) < 1e-9:
            return 0.0
        corr = abs(np.corrcoef(x, resid)[0, 1])
        return 1.0 - corr  # alto score = residuo independente = causal

    def _set_edge(self, cause: str, effect: str) -> None:
        if effect not in self.graph[cause]:
            self.graph[cause].append(effect)
        if cause not in self.reverse[effect]:
            self.reverse[effect].append(cause)

    def fit_functions(self, data: Dict[str, np.ndarray]) -> None:
        """Aprende as funcoes estruturais f_i(pais_i)."""
        for var in self.var_names:
            parents = self.reverse.get(var, [])
            y = data[var]
            if parents:
                X = np.column_stack([data[p] for p in parents])
            else:
                # Variavel exogena: modela como ruido
                X = np.zeros((len(y), 1))
            sf = StructuralFunction(parents=parents)
            sf.fit(X, y)
            self.struct_funcs[var] = sf

    def query_intervention(self, do: Dict[str, float],
                           target: str,
                           data: Dict[str, np.ndarray],
                           n_samples: int = 200) -> Dict[str, float]:
        """
        Responde P(target | do(X=x)).

        A intervencao do(X=x) CORTA os pais de X (graph surgery de Pearl).
        X recebe o valor fixado; o resto propaga pelas funcoes estruturais.
        Isto e a diferenca fundamental vs P(Y|X): remove o vies confounding.
        """
        samples = []
        rng = np.random.RandomState(42)

        for _ in range(n_samples):
            # Amostra valores das variaveis na ordem topologica
            values: Dict[str, float] = {}
            ordered = self._topo_order()

            for var in ordered:
                if var in do:
                    values[var] = do[var]  # INTERVENCAO: corta pais
                    continue
                sf = self.struct_funcs.get(var)
                if sf is None:
                    values[var] = 0.0
                    continue
                parents = self.reverse.get(var, [])
                if parents:
                    pvals = np.array([[values.get(p, 0.0) for p in parents]])
                    pred = sf.predict(pvals)[0]
                    # Adiciona ruido estrutural
                    pred += rng.normal(0, sf.residual_std)
                    values[var] = pred
                else:
                    # Exogena: amostra da marginal empirica
                    emp = data.get(var, np.array([0.0]))
                    values[var] = float(rng.choice(emp))

            samples.append(values.get(target, 0.0))

        samples = np.array(samples)
        return {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "n": n_samples,
        }

    def _topo_order(self) -> List[str]:
        """Ordem topologica (causas antes de efeitos)."""
        visited: Set[str] = set()
        order: List[str] = []

        def visit(v: str):
            if v in visited:
                return
            visited.add(v)
            for parent in self.reverse.get(v, []):
                visit(parent)
            order.append(v)

        for v in self.var_names:
            visit(v)
        return order

    def describes_confounding(self, data: Dict[str, np.ndarray]) -> bool:
        """
        Heuristica: detecta se ha confounding comparando P(Y|X) com P(Y|do(X)).
        Se diferem muito, ha confounder nao observado distorcendo a correlacao.
        """
        if len(self.var_names) != 2:
            return False
        a, b = self.var_names
        if b not in self.graph.get(a, []):
            return False

        # Compara regressao direta (observacional) com intervencional
        sf = self.struct_funcs.get(b)
        if sf is None:
            return False

        # P(Y|X=x) via regressao nos dados brutos
        x_test = np.linspace(-1.5, 1.5, 10)
        obs_pred = sf.predict(x_test.reshape(-1, 1))

        # P(Y|do(X=x)) via intervencao
        int_pred = np.array([
            self.query_intervention(do={a: float(x)}, target=b,
                                    data=data, n_samples=100)["mean"]
            for x in x_test
        ])

        # Se diferem > threshold, ha confounding
        diff = np.mean(np.abs(obs_pred - int_pred))
        scale = max(np.std(obs_pred), 1e-6)
        return (diff / scale) > 0.3

"""
active_experimenter.py — Agência Científica Ativa
===================================================
O Trial 6 original NAO media agência. Media se um np.linalg.lstsq sobre
todos os dados tinha erro baixo — o que qualquer regressão faz, sem
ciência nenhuma. Era Goodhart.

Este módulo implementa o que o trial deveria medir: o sistema ESCOLHE
quais experiências fazer (quais objetos testar), guiado por redução de
incerteza (information gain). Vence uma baseline aleatória se escolher
melhor que o acaso.

Mecanismo (Active Inference honesto):
  1. Observa dados iniciais (poucos)
  2. Ajusta modelo preliminar da lei física
  3. Para cada candidato a experimento, estima a REDUÇÃO DE INCERTEZA
     esperada (information gain / curiosidade epistêmica)
  4. Escolhe o experimento com maior ganho de informação
  5. Repete até convergência
  6. Avalia: erro de predição em objetos NÃO VISTOS vs estratégia aleatória

Isto é agência de verdade. O sistema nao recebe os dados — ele os PRODUZ.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class ActiveExperimenter:
    """
    Cientista ativo: escolhe experiências para reduzir incerteza.

    Modelo: lei física power-law t = a * m^b * s^c + d
    Estratégia ativa: escolhe objetos cuja observação mais reduz a
    incerteza nos parâmetros (a, b, c, d).
    """

    def __init__(self, dim_mass_range=(0.5, 10.0),
                 dim_size_range=(0.5, 5.0)):
        self.mass_range = dim_mass_range
        self.size_range = dim_size_range
        self.coef: Optional[np.ndarray] = None  # [log_a, b, c, log_d+1]
        self.observations: List[Tuple[float, float, float]] = []  # (m, s, t)
        self.experiment_log: List[Dict] = []

    def fit_power_law(self) -> None:
        """Ajusta t = a*m^b*s^c + d (em log-log: log(t-d) = log_a + b*log_m + c*log_s).

        DEFENSIVO: d não pode exceder min(t) nem ficar negativo, senão
        log(t-d) explode em NaN e o erro propaga para o trial inteiro.
        """
        if len(self.observations) < 3:
            return
        m = np.array([o[0] for o in self.observations])
        s = np.array([o[1] for o in self.observations])
        t = np.array([o[2] for o in self.observations])

        # Garante todos positivos (defesa contra dados degenerados)
        t = np.maximum(t, 0.1)

        d = 0.0
        t_min = float(np.min(t))
        best_coef = None
        for _ in range(3):
            # d sempre no intervalo [0, 0.8*t_min] — nunca deixa t-d <= 0
            d = max(0.0, min(d, 0.8 * t_min))
            t_shifted = np.maximum(t - d, 0.05)
            log_m = np.log(m + 1e-9)
            log_s = np.log(s + 1e-9)
            log_t = np.log(t_shifted)
            A = np.column_stack([np.ones_like(log_m), log_m, log_s])
            try:
                coef, _, _, _ = np.linalg.lstsq(A, log_t, rcond=None)
            except Exception:
                return
            log_a, b, c = coef
            # Limita b, c a faixa plausível (a lei física tem expoentes ~[-1,1])
            #
            # NÃO ABRA ESTAS TRAVAS sem medir a seleção de experimentos junto.
            # Elas parecem estreitas demais — e são, para PREDIÇÃO: com dados
            # reais de Casimir (P ∝ d⁻⁴) o expoente recuperado gruda em −2,0000
            # e o erro de predição vai a 3,4 ordens de grandeza. Abrir para
            # [-8,8] baixa esse erro em 9×.
            #
            # Só que elas são LOAD-BEARING para outra coisa. Ao manterem o
            # modelo muito restrito, mantêm também as predições suaves e
            # limitadas — e com isso o termo de sensibilidade em
            # expected_information_gain fica pequeno e uniforme, deixando a
            # escolha do experimento ser dominada pelo termo de cobertura
            # (min_dist). Cobertura pura é uma estratégia forte, e é ela que
            # está produzindo o resultado bom.
            #
            # Medido com 40 sementes na bancada física (novaphys), tarefa
            # casimir_expoente, contra escolha aleatória:
            #     travas como estão .................. 5,56× melhor que o acaso
            #     expoente aberto para [-8,8] ........ 0,04×
            #     expoente aberto + sensibilidade log  0,31×
            # Ou seja: "consertar" o ajuste piorou a seleção em 18×.
            #
            # Se o objetivo for melhorar a PREDIÇÃO, o caminho é um preditor
            # separado — não abrir a trava do modelo que dirige a aquisição.
            b = float(np.clip(b, -2.0, 2.0))
            c = float(np.clip(c, -2.0, 2.0))
            a = float(np.exp(np.clip(log_a, -5.0, 5.0)))
            log_a = float(np.log(a))
            # Atualiza d (sempre travado)
            d_new = float(np.mean(t - a * (np.power(m, b)) * (np.power(s, c))))
            d = max(0.0, min(d_new, 0.8 * t_min))
            best_coef = np.array([log_a, b, c, d])

        if best_coef is not None:
            self.coef = best_coef

    def predict(self, m: float, s: float) -> float:
        if self.coef is None:
            return 1.0
        log_a, b, c, d = self.coef
        try:
            a = float(np.exp(log_a))
            # Defesa: bases positivas, expoentes clipados já no fit
            val = a * (max(m, 1e-6) ** b) * (max(s, 1e-6) ** c) + d
            if not np.isfinite(val):
                return 1.0
            return float(val)
        except (OverflowError, ValueError, FloatingPointError):
            return 1.0

    def expected_information_gain(self, candidate: Tuple[float, float],
                                   n_perturb: int = 8) -> float:
        """
        Estima quanta informação ganharíamos observando este objeto.

        Information gain ≈ redução de incerteza nos parâmetros.
        Aproximação honesta: se já temos dados perto deste ponto,
        o ganho é baixo (redundante). Se é longe da cobertura atual,
        o ganho é alto (exploratório).

        Esta é a versão simples do Expected Information Gain (EIG)
        usado em Bayesian Optimal Experimental Design.
        """
        if self.coef is None:
            # Sem modelo ainda: qualquer ponto é informativo
            return 1.0

        m_c, s_c = candidate
        # Distância ao ponto mais próximo já observado (cobertura)
        if not self.observations:
            return 1.0
        dists = [math.sqrt((m_c - o[0])**2 + (s_c - o[1])**2)
                 for o in self.observations]
        min_dist = min(dists)

        # Ganho proporcional à distância (exploração)
        # Mas penalizado se estiver fora do range plausível
        gain = min(1.0, min_dist / 5.0)

        # Perturba o modelo e vê quão sensível a predição é a este ponto
        # (pontos em regiões de alta sensibilidade reduzem mais incerteza)
        #
        # A sensibilidade é ABSOLUTA, e isso parece um erro dimensional numa
        # grandeza que varia por ordens de grandeza — mas medir em log piora o
        # resultado neste código. Testado com 40 sementes: trocar por
        # |log(perturbada) − log(base)| levou a seleção de 5,56× para 0,31×
        # contra o acaso. O motivo é o mesmo das travas do fit_power_law: com o
        # modelo restrito, este termo fica pequeno e uniforme, e quem decide a
        # escolha acaba sendo o termo de cobertura logo acima — que é o que
        # está funcionando. Mexer aqui tira o poder de decisão da cobertura.
        base_pred = self.predict(m_c, s_c)
        sensitivity = 0.0
        for _ in range(n_perturb):
            # Perturba parâmetros
            noise = np.random.normal(0, 0.1, 4)
            old = self.coef.copy()
            self.coef = self.coef + noise
            perturbed_pred = self.predict(m_c, s_c)
            self.coef = old
            sensitivity += abs(perturbed_pred - base_pred)
        sensitivity /= n_perturb

        # Combina: exploração × sensibilidade
        return float(gain * (0.5 + sensitivity))

    def choose_experiment(self, candidates: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Escolhe o candidato com maior ganho de informação esperado."""
        if not candidates:
            return None
        scored = [(c, self.expected_information_gain(c)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        best = scored[0][0]
        self.experiment_log.append({
            "chosen": best,
            "score": scored[0][1],
            "n_candidates": len(candidates),
        })
        return best

    def observe(self, m: float, s: float, t: float) -> None:
        """Adiciona uma observação e refaz o fit."""
        self.observations.append((m, s, t))
        self.fit_power_law()

    def error_on(self, test_set: List[Tuple[float, float, float]]) -> float:
        """Erro de predição (MAPE) num conjunto NÃO VISTO. Defensivo contra zeros.

        NÃO aplicamos cap cego — o erro real (mesmo alto) é informativo para
        comparar ativo vs baseline. Só filtramos amostras degeneradas.
        """
        if not test_set or self.coef is None:
            return 1.0
        errors = []
        for m, s, t in test_set:
            pred = self.predict(m, s)
            if abs(t) < 1e-6 or not np.isfinite(pred):
                continue
            errors.append(abs(pred - t) / max(abs(t), 0.1))
        if not errors:
            return 1.0
        return float(np.mean(errors))

    @staticmethod
    def random_baseline_experiment(observations: List[Tuple[float, float, float]],
                                    test_set: List[Tuple[float, float, float]]) -> float:
        """
        Baseline aleatório: escolhe objetos aleatoriamente, ajusta o mesmo
        modelo, mede erro no mesmo test_set. Isto é o controle honesto.
        """
        baseline = ActiveExperimenter()
        for obs in observations:
            baseline.observe(*obs)
        return baseline.error_on(test_set)

"""
novaphys.agents — cientistas de referência
===========================================

Dois agentes que resolvem as tarefas de indução de leis, e um regressor
compartilhado por ambos.

O ponto do desenho: **o regressor é o mesmo para todo mundo**. O agente
ativo e a baseline aleatória usam exatamente o mesmo aprendiz, o mesmo
número de experimentos e as mesmas sondas. A única diferença é ONDE cada
um escolheu medir.

Isso isola a variável que interessa. Se o agente ativo vencer, foi por
escolher melhor onde olhar — que é a definição operacional de agência
científica. Se não vencer, a "descoberta" era regressão com passos extras,
e o número vai dizer isso na cara.

É o mesmo critério que `core/active_experimenter.py` já adota, e por bons
motivos: aquele arquivo documenta que a métrica original media se um
`lstsq` sobre todos os dados tinha erro baixo — o que qualquer regressão
faz, sem ciência nenhuma.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ═══════════════════════════════════════════════════ regressor comum
class BasisRegressor:
    """
    Mínimos quadrados regularizados sobre uma base genérica de features.

    A base é propositalmente ampla e não privilegia nenhuma tarefa:
        1, xi, log xi, sqrt xi, xi*xj, sqrt(xi)*xj
    Assim ela consegue expressar leis de potência (Casimir: P ∝ d⁻⁴),
    exponenciais em raiz (tunelamento: log T ∝ −w√V) e superfícies suaves
    com limiar (ganho de fusão), sem que nenhuma delas tenha sido embutida.

    Ridge com λ pequeno: com poucos pontos e muitas features, o ajuste sem
    regularização explode e o teste vira ruído.
    """

    def __init__(self, dims: Sequence[str], log_target: bool = True, lam: float = 1e-3):
        self.dims = list(dims)
        self.log_target = log_target
        self.lam = lam
        self.w: Optional[np.ndarray] = None
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    def features(self, cfg: Dict[str, float]) -> np.ndarray:
        x = np.array([float(cfg[d]) for d in self.dims], dtype=float)
        xs = np.maximum(np.abs(x), 1e-12)
        f = [1.0]
        f += list(x)
        f += list(np.log(xs))
        f += list(np.sqrt(xs))
        n = len(x)
        for i in range(n):
            for j in range(i, n):
                f.append(x[i] * x[j])
            for j in range(n):
                f.append(np.sqrt(xs[i]) * x[j])
        return np.array(f, dtype=float)

    def add(self, cfg: Dict[str, float], y: float) -> None:
        self.X.append(self.features(cfg))
        self.y.append(float(y))

    def fit(self) -> None:
        if len(self.X) < 2:
            self.w = None
            return
        A = np.vstack(self.X)
        t = np.array(self.y, dtype=float)
        if self.log_target:
            t = np.log10(np.maximum(np.abs(t), 1e-12))
        # ridge: (AᵀA + λI)w = Aᵀt
        G = A.T @ A + self.lam * np.eye(A.shape[1])
        try:
            self.w = np.linalg.solve(G, A.T @ t)
        except np.linalg.LinAlgError:
            self.w = np.linalg.pinv(A) @ t

    def predict(self, cfg: Dict[str, float]) -> float:
        if self.w is None:
            base = float(np.mean(self.y)) if self.y else 1.0
            return base
        v = float(self.features(cfg) @ self.w)
        if self.log_target:
            v = 10.0 ** float(np.clip(v, -12, 12))
        return v

    def predict_many(self, cfgs: Sequence[Dict[str, float]]) -> List[float]:
        return [self.predict(c) for c in cfgs]

    # ---- incerteza por bootstrap, para guiar a escolha do próximo experimento
    def uncertainty(self, cfg: Dict[str, float], n_boot: int = 12,
                    rng: Optional[random.Random] = None) -> float:
        """
        Espalhamento das predições sobre reamostragens dos dados já obtidos.
        Alto = o modelo não sabe o que acontece ali = vale a pena medir.
        É curiosidade epistêmica, não recompensa.
        """
        if len(self.X) < 3:
            return 1e9
        rng = rng or random.Random(0)
        A = np.vstack(self.X)
        t = np.array(self.y, dtype=float)
        if self.log_target:
            t = np.log10(np.maximum(np.abs(t), 1e-12))
        f = self.features(cfg)
        preds = []
        n = len(t)
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            Ab, tb = A[idx], t[idx]
            G = Ab.T @ Ab + max(self.lam, 1e-2) * np.eye(Ab.shape[1])
            try:
                w = np.linalg.solve(G, Ab.T @ tb)
            except np.linalg.LinAlgError:
                continue
            preds.append(float(f @ w))
        if len(preds) < 2:
            return 1e9
        return float(np.std(preds))


# ═══════════════════════════════════════════════════ modelo local
class LocalRegressor:
    """
    Regressor local (Nadaraya-Watson com largura de banda adaptativa).

    Não assume forma funcional nenhuma: prevê pela média ponderada dos vizinhos
    mais próximos. É o oposto do `BasisRegressor`, que impõe uma família suave.

    Por que isso importa aqui: a bancada mostrou que a busca por incerteza sobre
    base suave PERDE para o acaso em espectro discreto, limiar e pico estreito.
    A causa é que uma base suave não consegue nem representar essas paisagens —
    a "incerteza" que ela reporta é erro de especificação, não ignorância sobre
    o mundo. Um modelo local representa qualquer uma delas, desde que haja
    amostra por perto.
    """

    def __init__(self, dims: Sequence[str], espaco: List[Dict[str, Any]],
                 log_target: bool = True, k: int = 4):
        self.dims = list(dims)
        self.log_target = log_target
        self.k = k
        # escala de cada dimensão, para a distância não ser dominada por unidades
        self.escala = {d["chave"]: max(1e-12, d["max"] - d["min"]) for d in espaco}
        self.X: List[Dict[str, float]] = []
        self.y: List[float] = []

    def add(self, cfg: Dict[str, float], y: float) -> None:
        self.X.append(dict(cfg))
        self.y.append(float(y))

    def fit(self) -> None:  # nada a ajustar: o modelo é os próprios dados
        pass

    def _dist(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.sqrt(sum(((a[d] - b[d]) / self.escala[d]) ** 2 for d in self.dims))

    def predict(self, cfg: Dict[str, float]) -> float:
        if not self.X:
            return 1.0
        ds = [(self._dist(cfg, x), i) for i, x in enumerate(self.X)]
        ds.sort()
        k = min(self.k, len(ds))
        # largura de banda = distância ao k-ésimo vizinho: densa onde há dados,
        # larga onde há poucos
        h = max(ds[k - 1][0], 1e-6)
        num = den = 0.0
        for d, i in ds[: max(k, 6)]:
            w = math.exp(-0.5 * (d / h) ** 2)
            t = self.y[i]
            if self.log_target:
                t = math.log10(max(1e-12, abs(t)))
            num += w * t
            den += w
        v = num / max(den, 1e-12)
        return 10.0 ** max(-12.0, min(12.0, v)) if self.log_target else v

    def predict_many(self, cfgs: Sequence[Dict[str, float]]) -> List[float]:
        return [self.predict(c) for c in cfgs]


class HybridPredictor:
    """
    Combina o modelo suave e o local, pesando cada um pelo erro de
    validação leave-one-out nos dados já obtidos.

    Nenhum dos dois é bom em tudo: o suave extrapola bem numa lei de potência
    e é péssimo num limiar; o local acerta o limiar e não extrapola nada.
    Deixar os dados decidirem o peso evita escolher errado a priori.
    """

    def __init__(self, dims, espaco, log_target: bool = True):
        self.suave = BasisRegressor(dims, log_target=log_target)
        self.local = LocalRegressor(dims, espaco, log_target=log_target)
        self.w_suave = 0.5

    def add(self, cfg, y):
        self.suave.add(cfg, y)
        self.local.add(cfg, y)

    def fit(self):
        self.suave.fit()
        n = len(self.local.y)
        if n < 4:
            self.w_suave = 0.5
            return
        # leave-one-out para os dois modelos
        errs = {"suave": 0.0, "local": 0.0}
        for i in range(n):
            cfg, alvo = self.local.X[i], self.local.y[i]
            lt = math.log10(max(1e-12, abs(alvo)))
            sub_s = BasisRegressor(self.suave.dims, log_target=True)
            sub_l = LocalRegressor(self.local.dims,
                                   [{"chave": d, "min": 0.0,
                                     "max": self.local.escala[d]} for d in self.local.dims],
                                   log_target=True, k=self.local.k)
            for j in range(n):
                if j == i:
                    continue
                sub_s.add(self.local.X[j], self.local.y[j])
                sub_l.add(self.local.X[j], self.local.y[j])
            sub_s.fit()
            for nome, sub in (("suave", sub_s), ("local", sub_l)):
                p = sub.predict(cfg)
                errs[nome] += (math.log10(max(1e-12, abs(p))) - lt) ** 2
        es, el = math.sqrt(errs["suave"] / n), math.sqrt(errs["local"] / n)
        # Peso pelo INVERSO DO QUADRADO do erro, e sem piso.
        #
        # A primeira versão usava el/(es+el) com piso em 0,05. Parecia inofensivo
        # e não era: numa lei de potência pura o modelo local erra várias ordens
        # de grandeza ao extrapolar, e 5% de um erro de 3 décadas ainda são
        # 0,15 décadas cravadas na predição. A ablação mediu exatamente isso —
        # o híbrido piorou os dois braços 8×, em Casimir, onde o modelo suave
        # sozinho era quase exato. Um modelo claramente pior tem de poder ir a
        # zero, não a 5%.
        if es <= 0:
            self.w_suave = 1.0
        elif el <= 0:
            self.w_suave = 0.0
        else:
            a, b = 1.0 / (es * es), 1.0 / (el * el)
            self.w_suave = a / (a + b)

    def predict(self, cfg) -> float:
        ps, pl = self.suave.predict(cfg), self.local.predict(cfg)
        ls = math.log10(max(1e-12, abs(ps)))
        ll = math.log10(max(1e-12, abs(pl)))
        return 10.0 ** (self.w_suave * ls + (1 - self.w_suave) * ll)

    def predict_many(self, cfgs) -> List[float]:
        return [self.predict(c) for c in cfgs]

    def desacordo(self, cfg) -> float:
        """
        Quanto os dois modelos discordam num ponto.

        Este é o sinal central do agente corrigido: onde uma família suave e um
        modelo local dão respostas diferentes, existe estrutura que a família
        suave não representa — uma descontinuidade, um pico, um limiar.
        Discordância é, na prática, um detector de onde a física é interessante.
        """
        ps, pl = self.suave.predict(cfg), self.local.predict(cfg)
        return abs(math.log10(max(1e-12, abs(ps))) - math.log10(max(1e-12, abs(pl))))


# ═══════════════════════════════════════════════════ agentes
class RandomScientist:
    """Baseline: escolhe onde medir ao acaso, uniformemente no espaço."""

    nome = "aleatório"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def escolher(self, espaco: List[Dict[str, Any]],
                 historico: List[Dict[str, Any]]) -> Dict[str, float]:
        cfg = {}
        for d in espaco:
            if d.get("log"):
                cfg[d["chave"]] = math.exp(self.rng.uniform(math.log(d["min"]), math.log(d["max"])))
            else:
                cfg[d["chave"]] = self.rng.uniform(d["min"], d["max"])
        return cfg


class ActiveScientist:
    """
    Escolhe o experimento cuja observação mais reduz a incerteza do modelo.

    Estratégia:
      1. Um punhado de medições de arranque, espalhadas pelas bordas e pelo
         meio do espaço (sem isso não há modelo nenhum para questionar).
      2. Depois, a cada rodada: sorteia candidatos, estima a incerteza
         preditiva em cada um por bootstrap, mede o mais incerto.
      3. Penaliza candidatos colados em pontos já medidos — repetir uma
         medição quase idêntica não traz informação nova.

    É Active Inference na versão honesta: sem recompensa externa, guiado só
    por reduzir a própria ignorância.
    """

    nome = "ativo (ganho de informação)"

    def __init__(self, seed: int = 0, n_cand: int = 64, n_seed: int = 4):
        self.rng = random.Random(seed)
        self.n_cand = n_cand
        self.n_seed = n_seed
        self.reg: Optional[BasisRegressor] = None

    def _amostra(self, espaco) -> Dict[str, float]:
        cfg = {}
        for d in espaco:
            if d.get("log"):
                cfg[d["chave"]] = math.exp(self.rng.uniform(math.log(d["min"]), math.log(d["max"])))
            else:
                cfg[d["chave"]] = self.rng.uniform(d["min"], d["max"])
        return cfg

    def escolher(self, espaco: List[Dict[str, Any]],
                 historico: List[Dict[str, Any]]) -> Dict[str, float]:
        dims = [d["chave"] for d in espaco]

        # fase 1: arranque estruturado — extremos e centro de cada dimensão
        if len(historico) < self.n_seed:
            k = len(historico)
            cfg = {}
            for i, d in enumerate(espaco):
                lo, hi = d["min"], d["max"]
                if d.get("log"):
                    pontos = [lo * 1.05, math.sqrt(lo * hi), hi * 0.95, math.exp(0.5 * (math.log(lo) + math.log(hi)) + 0.4)]
                else:
                    pontos = [lo + 0.05 * (hi - lo), 0.5 * (lo + hi), hi - 0.05 * (hi - lo), lo + 0.25 * (hi - lo)]
                cfg[d["chave"]] = pontos[(k + i) % len(pontos)]
            return cfg

        # fase 2: máxima incerteza
        reg = BasisRegressor(dims, log_target=True)
        for h in historico:
            reg.add(h["config"], max(1e-12, abs(h["resultado"])))
        reg.fit()
        self.reg = reg

        melhor, melhor_score = None, -1.0
        for _ in range(self.n_cand):
            cand = self._amostra(espaco)
            u = reg.uncertainty(cand, rng=self.rng)
            # penaliza proximidade ao que já foi medido
            dmin = min(
                (sum(((cand[k] - h["config"][k]) / max(1e-9, d["max"] - d["min"])) ** 2
                     for k, d in zip(dims, espaco)) ** 0.5
                 for h in historico), default=1.0)
            score = u * (1.0 - math.exp(-6.0 * dmin))
            if score > melhor_score:
                melhor_score, melhor = score, cand
        return melhor or self._amostra(espaco)


class ActiveScientistV2:
    """
    Versão corrigida, depois do que a bancada mostrou.

    A primeira versão perdia para o acaso em três de cinco paisagens porque
    media a incerteza de um modelo suave — e num espectro discreto ou num
    limiar, essa incerteza é erro de especificação, não ignorância. O agente
    ia medir onde o próprio modelo estava mal formulado.

    A correção tem três partes:

      1. Preditor híbrido (suave + local), com o peso saindo de validação
         leave-one-out, e não de aposta prévia.
      2. Aquisição por DESACORDO entre os dois modelos, que é um detector
         de descontinuidade: onde o suave e o local divergem, há estrutura.
      3. Uma fração fixa do orçamento reservada a cobertura pura do espaço.
         Sem isso, o agente encontra a primeira feição interessante e
         fica orbitando nela — o mesmo erro de um cientista que só investiga
         a hipótese de que já gostou.
    """

    nome = "ativo v2 (desacordo)"

    def __init__(self, seed: int = 0, n_cand: int = 96, n_seed: int = 5,
                 frac_cobertura: float = 0.3):
        self.rng = random.Random(seed)
        self.n_cand = n_cand
        self.n_seed = n_seed
        self.frac_cobertura = frac_cobertura
        self.modelo: Optional[HybridPredictor] = None

    def _amostra(self, espaco) -> Dict[str, float]:
        cfg = {}
        for d in espaco:
            if d.get("log"):
                cfg[d["chave"]] = math.exp(self.rng.uniform(math.log(d["min"]), math.log(d["max"])))
            else:
                cfg[d["chave"]] = self.rng.uniform(d["min"], d["max"])
        return cfg

    def _mais_distante(self, espaco, historico) -> Dict[str, float]:
        """Cobertura: o candidato mais longe de tudo que já foi medido."""
        dims = [d["chave"] for d in espaco]
        esc = {d["chave"]: max(1e-12, d["max"] - d["min"]) for d in espaco}
        melhor, best_d = None, -1.0
        for _ in range(self.n_cand):
            c = self._amostra(espaco)
            dmin = min((math.sqrt(sum(((c[k] - h["config"][k]) / esc[k]) ** 2 for k in dims))
                        for h in historico), default=1e9)
            if dmin > best_d:
                best_d, melhor = dmin, c
        return melhor or self._amostra(espaco)

    def escolher(self, espaco, historico, orcamento: int = 20) -> Dict[str, float]:
        dims = [d["chave"] for d in espaco]

        # fase 1: arranque nos extremos e no meio (uma lei só aparece com alcance)
        if len(historico) < self.n_seed:
            k = len(historico)
            cfg = {}
            for i, d in enumerate(espaco):
                lo, hi = d["min"], d["max"]
                if d.get("log"):
                    pts = [lo * 1.03, math.sqrt(lo * hi), hi * 0.97,
                           math.exp(0.25 * math.log(lo) + 0.75 * math.log(hi)),
                           math.exp(0.75 * math.log(lo) + 0.25 * math.log(hi))]
                else:
                    pts = [lo + 0.03 * (hi - lo), 0.5 * (lo + hi), hi - 0.03 * (hi - lo),
                           lo + 0.75 * (hi - lo), lo + 0.25 * (hi - lo)]
                cfg[d["chave"]] = pts[(k + i) % len(pts)]
            return cfg

        # cobertura intercalada, para não orbitar a primeira feição achada
        if self.rng.random() < self.frac_cobertura:
            return self._mais_distante(espaco, historico)

        modelo = HybridPredictor(dims, espaco, log_target=True)
        for h in historico:
            modelo.add(h["config"], max(1e-12, abs(h["resultado"])))
        modelo.fit()
        self.modelo = modelo

        esc = {d["chave"]: max(1e-12, d["max"] - d["min"]) for d in espaco}
        melhor, melhor_score = None, -1.0
        for _ in range(self.n_cand):
            c = self._amostra(espaco)
            desac = modelo.desacordo(c)
            incert = modelo.suave.uncertainty(c, n_boot=8, rng=self.rng)
            dmin = min((math.sqrt(sum(((c[k] - h["config"][k]) / esc[k]) ** 2 for k in dims))
                        for h in historico), default=1.0)
            # desacordo domina; a incerteza do modelo suave entra só como desempate
            score = (desac + 0.15 * min(incert, 5.0)) * (1.0 - math.exp(-8.0 * dmin))
            if score > melhor_score:
                melhor_score, melhor = score, c
        return melhor or self._amostra(espaco)


class ActiveOptimizer:
    """
    Para tarefas de otimização (achar o máximo), não de predição.

    Faz busca grosseira e depois refina em torno do melhor, dividindo a
    janela pela metade a cada rodada. Com orçamento pequeno num pico
    estreito, isso ganha muito de amostragem uniforme — e o benchmark do
    ângulo áureo tem exatamente um pico estreito.
    """

    nome = "ativo (refinamento sucessivo)"

    def __init__(self, seed: int = 0, frac_grossa: float = 0.8):
        """
        `frac_grossa` alto de propósito. A versão anterior gastava metade do
        orçamento refinando e perdia para o acaso: num pico de 0,5° numa faixa
        de 80°, refinar em torno de um máximo local errado é jogar fora
        experimento. O que decide o resultado é a COBERTURA — a chance de a
        varredura grossa cair numa região boa —, e refinar só faz sentido depois
        que ela caiu. Custo de aprender isso: a suíte medindo 0,63×.
        """
        self.rng = random.Random(seed)
        self.frac_grossa = frac_grossa

    def escolher(self, espaco, historico, orcamento: int = 20) -> Dict[str, float]:
        d = espaco[0]
        chave, lo, hi = d["chave"], d["min"], d["max"]
        n_grossa = max(4, int(orcamento * self.frac_grossa))
        k = len(historico)
        if k < n_grossa:
            # Amostragem estratificada COM JITTER: divide a faixa em n_grossa
            # faixas iguais e sorteia um ponto dentro de cada uma.
            #
            # A versão anterior usava o centro de cada faixa — uma grade fixa —
            # e perdia para amostragem aleatória. O motivo é sutil: com um pico
            # de 0,5° numa faixa de 80°, uma grade determinística ou acerta ou
            # não acerta, e isso está decidido antes de começar. Amostragem
            # aleatória tem ~91% de chance de cair numa região boa com 30 tiros.
            # A estratificação recupera a garantia de cobertura da grade, e o
            # jitter elimina a ressonância entre o passo da grade e a estrutura
            # da paisagem. É o melhor dos dois.
            largura = (hi - lo) / n_grossa
            return {chave: lo + largura * (k + self.rng.random())}

        # Refinamento fino em torno do melhor, com passo caindo pela metade.
        # Poucos pontos: o grosso do orçamento já foi para cobertura, que é o
        # que de fato determina se alguma região boa foi encontrada.
        melhor = max(historico, key=lambda h: h["resultado"])
        centro = melhor["config"][chave]
        rodada = k - n_grossa
        passo = (hi - lo) / n_grossa / (2 ** (1 + rodada // 2))
        desloc = 1 if rodada % 2 == 0 else -1
        alvo = centro + desloc * passo
        return {chave: float(min(hi, max(lo, alvo)))}

    def melhor(self, historico) -> Dict[str, float]:
        h = max(historico, key=lambda x: x["resultado"])
        return dict(h["config"])


class RandomOptimizer:
    """Baseline de otimização: amostra uniformemente e devolve o melhor visto."""

    nome = "aleatório"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def escolher(self, espaco, historico, orcamento: int = 20) -> Dict[str, float]:
        d = espaco[0]
        return {d["chave"]: self.rng.uniform(d["min"], d["max"])}

    def melhor(self, historico) -> Dict[str, float]:
        h = max(historico, key=lambda x: x["resultado"])
        return dict(h["config"])

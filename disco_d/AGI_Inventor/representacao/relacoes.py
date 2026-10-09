"""
relacoes — explicar MUDANCAS de movimento com estados relativos
=================================================================

Lado do APRENDIZ (PRE_REGISTRO_etapa2.md). Quando a inercia falha, o que
distingue o futuro diferente? Para cada entidade em movimento, o residuo da
inercia r = pos(t+1) - (pos(t) + v(t)) e previsto por duas arvores:
  INDEPENDENTE  so o estado causal da propria entidade
  RELACIONAL    + as 2 entidades mais proximas em t, medidas ate a CAIXA delas
                (contato), e a velocidade delas
Nenhuma entidade tem nome; a relacao que importar sai porque separa futuros.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import percepcao as P

LONGE = 99.0


def _ate_borda(p, borda):
    """Vetor ate o PIXEL de borda mais proximo do outro (contato); a caixa mente em U."""
    d = borda.astype(float) - p
    k = int(np.argmin((d ** 2).sum(1)))
    return d[k]


def amostras(ents, ts, acoes=None):
    """X_ind, X_rel, residuo r, e (quadro, entidade, parceira) de cada amostra."""
    causal = [P.estado_causal(T, ts) for T in ents]
    por_q = defaultdict(list)
    for i, T in enumerate(ents):
        Z, ok = causal[i]
        for j, k in enumerate(T["q"]):
            v = Z[j, 2:] if ok[j] else np.zeros(2)
            por_q[int(k)].append((i, T["borda"][j], v))
    Xi, Xr, R, info = [], [], [], []
    for i, T in enumerate(ents):
        if len(T["q"]) < 10 or not P.em_movimento(T, ts):
            continue
        Z, ok = causal[i]
        q = T["q"]
        for j in range(len(q) - 1):
            if not ok[j] or q[j + 1] != q[j] + 1:
                continue
            dt = ts[q[j + 1]] - ts[q[j]]
            p0 = T["pos"][j]
            r = T["pos"][j + 1] - (p0 + Z[j, 2:] * dt)
            outros = []
            for (o, cx, vo) in por_q[int(q[j])]:
                if o == i:
                    continue
                d = _ate_borda(p0, cx)
                outros.append((np.hypot(*d), d, vo, o))
            outros.sort(key=lambda x: x[0])
            rel = []
            for m in range(2):
                if m < len(outros):
                    _dn, d, vo, _o = outros[m]
                    rel += [d[0], d[1], vo[0], vo[1]]
                else:
                    rel += [LONGE, LONGE, 0.0, 0.0]
            a = [acoes[int(q[j])] if acoes is not None else -1.0]   # acao OBSERVADA
            Xi.append(np.concatenate([Z[j], a]))
            Xr.append(np.concatenate([Z[j], a, rel]))
            R.append(r)
            info.append((int(q[j]), i, outros[0][3] if outros else -1, p0.copy(), Z[j, 2:] * dt))
    return (np.array(Xi), np.array(Xr), np.array(R), info)


class ModeloEvento:
    """
    Como a proposta descreve: reunir os estados antes das FALHAS da inercia e
    CONTRASTAR com transicoes sem falha (classificador, eventos pesados por igual);
    o efeito da falha e aprendido so nos eventos. Previsao: efeito se P(falha) > 0,5.
    """

    def __init__(self, X, R, limiar=1.5):
        from sklearn.tree import DecisionTreeClassifier
        falha = np.linalg.norm(R, axis=1) > limiar
        self.clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                          class_weight="balanced", random_state=0)
        self.clf.fit(X, falha)
        self.reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3, random_state=0)
        self.reg.fit(X[falha], R[falha]) if falha.sum() >= 3 else None
        self.ok = falha.sum() >= 3
        self.feature_importances_ = self.clf.feature_importances_

    def predict(self, X):
        if not self.ok:
            return np.zeros((len(X), 2))
        p = self.clf.predict_proba(X)[:, list(self.clf.classes_).index(True)]             if True in self.clf.classes_ else np.zeros(len(X))
        return self.reg.predict(X) * (p > 0.5)[:, None]


def arvore(X, R):
    return ModeloEvento(X, R)


def erro(mod, X, R):
    return np.linalg.norm(mod.predict(X) - R, axis=1)


def ic_blocos(dif, rng, bloco=50, reps=1000):
    """IC95 da media de `dif` (na ordem do tempo), por blocos."""
    bl = [dif[i:i + bloco] for i in range(0, len(dif), bloco)]
    if len(bl) < 2:
        return (float("nan"), float("nan"))
    ms = sorted(np.concatenate([bl[k] for k in rng.integers(len(bl), size=len(bl))]).mean()
                for _ in range(reps))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps)]


NOMES_REL = ["u", "v", "du", "dv", "acao", "dx1", "dy1", "vx1", "vy1", "dx2", "dy2", "vx2",
             "vy2"]

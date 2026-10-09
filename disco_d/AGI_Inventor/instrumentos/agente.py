"""
agente — coleta com sondas, decisao por fitted Q-iteration, e as duas avaliacoes
==================================================================================

Um BRACO (U, C, I ou P = I-placebo) gasta o orcamento jogando e sondando; o
Laboratorio decide que experiencias rodar e, nos bracos I/P, cria sensores,
experiencias e variaveis. O decisor e IGUAL para todos; so muda a representacao.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import ambiente as A
import instrumentos as I

SONDA_CADA = 16
GAMA = 0.97


def _um(nA, a, n):
    M = np.zeros((n, nA), np.float32)
    M[np.arange(n), a] = 1
    return M


def fqi(trans, lab, nA, rng, iters=5, n_max=60000):
    idx = rng.choice(len(trans), min(n_max, len(trans)), replace=False)
    S = np.stack([trans[i][0] for i in idx]).astype(np.float32)
    Aa = np.array([trans[i][1] for i in idx])
    R = np.clip(np.array([trans[i][2] for i in idx], np.float32), -1, 1)
    S2 = np.stack([trans[i][3] for i in idx]).astype(np.float32)
    D = np.array([trans[i][4] for i in idx], np.float32)
    nv = len(lab.vars)
    F, F2 = lab.rep(S, nv), lab.rep(S2, nv)
    X = np.concatenate([F, _um(nA, Aa, len(Aa))], 1)
    y, m = R, None
    for it in range(iters):
        m = ExtraTreesRegressor(n_estimators=15, min_samples_leaf=10, max_features=0.3,
                                n_jobs=2, random_state=it).fit(X, y)
        qn = np.stack([m.predict(np.concatenate([F2, _um(nA, np.full(len(F2), a), len(F2))], 1))
                       for a in range(nA)], 1).max(1)
        y = R + GAMA * (1 - D) * qn
    m.n_vars = nv                                # a representacao que este decisor conhece
    return m


def escolhe_acao(Q, lab, r0, nA):
    f = lab.rep(r0[None], Q.n_vars)
    X = np.concatenate([np.repeat(f, nA, 0), np.eye(nA, dtype=np.float32)], 1)
    return int(np.argmax(Q.predict(X)))


def coleta(jogo, braco, semente, orcamento=100_000, log=print):
    t0 = time.time()
    rng = np.random.default_rng(semente)
    mundo = A.Mundo(jogo, semente, 0.0, orcamento=orcamento)
    lab = I.Laboratorio(mundo.nA, rng, braco)
    trans, Q = [], None
    o = mundo.reinicia()
    oa = o.copy()
    passo, prox_fqi = 0, orcamento // 2
    retornos, ret = [], 0.0
    try:
        while True:
            if passo % SONDA_CADA == 0:
                r0 = A.base(o, oa)
                pids = lab.escolhe(r0)
                salvo = mundo.salva()
                difs = []
                for pid in pids:
                    mundo.restaura(salvo)
                    ramo = []
                    oH, R, _fim = I.executa(mundo, lab.progs[pid], o, oa, ramo, A.base)
                    trans += [(s.astype(np.float16), a, r, s2.astype(np.float16), d)
                              for s, a, r, s2, d in ramo]
                    dif = (oH - o).reshape(21, 2, 16, 2).mean((1, 3)).ravel()
                    lab.registra(passo, r0, int(pid), I.neutro(o, oH, R), dif)
                    difs.append(dif)
                mundo.restaura(salvo)
                lab.registra_par(difs[0], difs[1])
                if len(lab.reg) % 400 == 0:
                    lab.revisa()
            eps = max(0.1, 1.0 - mundo.passos / (0.6 * orcamento))
            r0 = A.base(o, oa)
            if Q is None or rng.random() < eps:
                a = int(rng.integers(mundo.nA))
            else:
                a = escolhe_acao(Q, lab, r0, mundo.nA)
            o2, r, d = mundo.passo(a)
            trans.append((r0.astype(np.float16), a, r, A.base(o2, o).astype(np.float16), d))
            ret += r
            oa, o = o, o2
            passo += 1
            if d:
                retornos.append(ret)
                ret = 0.0
                o = mundo.reinicia()
                oa = o.copy()
            if mundo.passos >= prox_fqi and Q is None:
                lab.revisa()
                Q = fqi(trans, lab, mundo.nA, rng)
                log("    [%s/%s] FQI no meio (%d passos, %d transicoes, %.0f s)"
                    % (jogo, braco, mundo.passos, len(trans), time.time() - t0))
    except A.Orcamento:
        pass
    lab.revisa()
    Q = fqi(trans, lab, mundo.nA, rng)
    return {"lab": lab, "Q": Q, "nA": mundo.nA, "retornos_coleta": retornos,
            "n_sondas": len(lab.pares), "passos_jogo": passo, "t_coleta_s": time.time() - t0}


def avalia_pontos(jogo, res, semente, episodios=5, teto=10_000, aparencia=None, eps=0.05):
    rng = np.random.default_rng(semente + 500)
    mundo = A.Mundo(jogo, semente + 500, 0.25, aparencia=aparencia)
    pts = []
    for _ in range(episodios):
        o = mundo.reinicia()
        oa = o.copy()
        tot, n = 0.0, 0
        while n < teto:
            r0 = A.base(o, oa)
            a = int(rng.integers(res["nA"])) if rng.random() < eps else \
                escolhe_acao(res["Q"], res["lab"], r0, res["nA"])
            o2, r, d = mundo.passo(a)
            tot += r
            n += 1
            oa, o = o, o2
            if d:
                break
        pts.append(tot)
    return pts


def antecipacao(jogo, res, semente, n=200):
    """Episodios NOVOS: prever o desfecho neutro de experiencias de L0 com a representacao."""
    lab = res["lab"]
    X = lab.rep(np.stack([r[1] for r in lab.reg]))
    Y = np.stack([r[3] for r in lab.reg])
    P = np.array([r[2] for r in lab.reg])
    modelos = {}
    for p in range(lab.n_L0):
        m = P == p
        if m.sum() < 8:
            modelos[p] = ("media", Y.mean(0))
            continue
        Xm = np.concatenate([X[m], np.ones((m.sum(), 1))], 1)
        B = np.linalg.solve(Xm.T @ Xm + 1.0 * np.eye(Xm.shape[1]), Xm.T @ Y[m])
        modelos[p] = ("lin", B)
    rng = np.random.default_rng(semente + 999)
    mundo = A.Mundo(jogo, semente + 999, 0.0)
    o = mundo.reinicia()
    oa = o.copy()
    erros, passo = [], 0
    while len(erros) < n:
        if passo % 25 == 0 and passo > 0:
            r0 = A.base(o, oa)
            f = np.concatenate([lab.rep(r0[None])[0], [1.0]])
            salvo = mundo.salva()
            for p in rng.choice(lab.n_L0, 2, replace=False):
                mundo.restaura(salvo)
                oH, R, _ = I.executa(mundo, lab.progs[p], o, oa, [], A.base)
                y = I.neutro(o, oH, R)
                tipo, B = modelos[p]
                yp = B if tipo == "media" else f @ B
                erros.append(float(((y - yp) ** 2).mean()))
            mundo.restaura(salvo)
        o2, _r, d = mundo.passo(int(rng.integers(mundo.nA)))
        oa, o = o, o2
        passo += 1
        if d:
            o = mundo.reinicia()
            oa = o.copy()
    return float(np.mean(erros))

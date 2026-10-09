"""
novaphys.inventa — inventar a grandeza que faz a lei caber
============================================================

DE ONDE VEM
-------------
`orbita3`: com h no caderno, a lei de Einstein sai 10/10 como UM termo,
-0,030 h^2 r^-4. Sem h, vira dois termos, (v^2 - vr^2)/r^2, e o fraco e
subdeterminado: lei exata em 7/10, e as 3 falhas desabam fora da faixa medida.

Kepler resolveu problema desse tipo inventando a grandeza: a area varrida por
unidade de tempo e constante (h conservado). A lei seguinte, Newton, ficou
simples POR CAUSA dela. O sistema nao inventa grandezas — usa as que eu dou.
Toda abstracao entregue vira o teto (padrao medido seis vezes).

O MECANISMO
-------------
FASE 1 — INVENTAR. O agente recebe so TRAJETORIAS: posicao e velocidade de 20
corpos ao longo do tempo, com ruido. Procura a combinacao linear
    q = SOMA w_i m_i(estado)
que fica CONSTANTE ao longo de cada orbita e VARIA entre orbitas. Para cada
subconjunto de ate 3 monomios, o melhor w e o menor autovetor generalizado de
(variancia DENTRO das trajetorias, variancia TOTAL) — lambda e a fracao da
variancia que sobra dentro das orbitas.

A regua e a de sempre, em bits: saber que q e constante por orbita comprime
    poupanca = (N/2) log2(1/lambda) - (n_traj/2) log2(N/n_traj)
             - log2(3) - log2 C(m,k) - (k-1)/2 log2 N      (estrutura)
Escolhida no treino, validada em 20 trajetorias NOVAS (aninhado).

O alfabeto e meu: monomios de grau 1 e 2 em (x, y, vx, vy), mais r, 1/r, 1/r^2 —
17 monomios, 833 subconjuntos. h = x vy - y vx esta la como PALAVRA de duas
letras; a energia (v^2/2 - 1/r) como palavra de tres. Eu nao digo qual procurar.

FASE 2 — USAR. A grandeza inventada entra no caderno como variavel "q", e a
descoberta da lei roda de novo no mundo einstein SEM h, com os MESMOS dados de
laboratorio e as mesmas dobras da `orbita3` (semente 20260915) — comparacao
pareada com os 7/10 de la.

CONTROLES
-----------
  permutado  os pontos das trajetorias sorteados entre as trajetorias: nenhuma
             grandeza pode ser constante por orbita
  placebo    no lugar de q, uma grandeza NAO conservada feita do mesmo alfabeto:
             p = x vx + y vy (= r vr). Se o placebo ajudar tanto quanto q, o ganho
             e "mais variaveis", nao "a grandeza certa"
  arrasto    (descritivo) mundo com atrito, -0,05 v: nada e conservado de verdade

O QUE NAO E NOVO, DITO ANTES
------------------------------
Descobrir leis de conservacao a partir de trajetorias ja foi feito (AI Poincare,
Liu & Tegmark 2021; metodos estilo SINDy). O que este degrau mede e o LACO: o
conceito inventado entra no vocabulario e se paga na descoberta seguinte, contra
placebo.

PRE-REGISTRO, ANTES DE RODAR — semente 20260916, 10 sementes
--------------------------------------------------------------
  I0  INVENCAO: no mundo einstein, a grandeza aceita tem |corr(q, h)| >= 0,999 em
      estados novos, em >= 9/10
  I1  NULO: no permutado, nenhuma grandeza aceita em >= 9/10
  I2  USO: com q no caderno (sem h), a lei sai como UM termo  c q^2 r^-4 r^  e
      extrapola com R2 >= 0,98, em >= 9/10
  I3  PLACEBO: com p no lugar de q, extrapolacao R2 >= 0,98 em <= 8/10
      (linha de base pareada da `orbita3`: 7/10)
  I4  REPRODUCAO: o braco sem extra reproduz os 7/10 da `orbita3`

PREVISAO: I0 10/10 — h e exata em duas letras e so tem o ruido de medida; a
energia tambem sera quase conservada, mas a correcao de Einstein (A h^2/3r^3)
nao esta no alfabeto, entao perde. I1 10/10. I2 10/10 — com q = c h, e a condicao
"com h" disfarcada, que deu 10/10. I3 ~7/10. I4 exato.
O ponto duro e I0. Se I0 passa, I2 e quase automatico — o que o experimento
realmente testa e se o agente acha a grandeza SEM eu apontar.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import time
from math import comb, log2

import numpy as np
from scipy.linalg import eigh, LinAlgError

from novaphys import orbita as O
from novaphys import orbita2 as O2
from novaphys import orbita3 as O3

SEMENTE = 20260916
N_TRAJ = 20
N_PTS = 25
T_MAX = 6.0
SIG = 0.002
GAMA = 0.05
MAX_K = 3


# ─── o mundo em movimento ───────────────────────────────────────────────────

def acel(mundo, S):
    if mundo == "arrasto":
        return O.newton(S) - GAMA * S[:, 2:]
    return O.newton(S) + O.anomalia(mundo, S)


def _deriv(mundo, S):
    return np.concatenate([S[:, 2:], acel(mundo, S)], 1)


def integra(mundo, S0, ts, dt=0.005):
    S, t, out = S0.copy(), 0.0, [S0.copy()]
    for tn in ts[1:]:
        while t < tn - 1e-12:
            h = min(dt, tn - t)
            k1 = _deriv(mundo, S)
            k2 = _deriv(mundo, S + h / 2 * k1)
            k3 = _deriv(mundo, S + h / 2 * k2)
            k4 = _deriv(mundo, S + h * k3)
            S = S + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out.append(S.copy())
    return np.stack(out, 1)


def trajetorias(mundo, rng, n):
    m = 3 * n
    r = rng.uniform(0.8, 2.5, m)
    th = rng.uniform(0, 2 * np.pi, m)
    v = rng.uniform(0.8, 1.2, m) / np.sqrt(r)
    ang = rng.uniform(-0.4, 0.4, m)
    sentido = np.where(rng.random(m) < 0.3, -1.0, 1.0)
    vt, vr = v * np.cos(ang) * sentido, v * np.sin(ang)
    c, s = np.cos(th), np.sin(th)
    S0 = np.stack([r * c, r * s, vr * c - vt * s, vr * s + vt * c], 1)
    T = integra(mundo, S0, np.linspace(0, T_MAX, N_PTS))
    rr = np.hypot(T[..., 0], T[..., 1])
    ok = (rr.min(1) > 0.35) & (rr.max(1) < 8)
    T = T[ok][:n]
    return T + rng.normal(0, SIG, T.shape)


# ─── o alfabeto ──────────────────────────────────────────────────────────────

def alfabeto(S):
    x, y, vx, vy = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    r = np.hypot(x, y)
    base = {"x": x, "y": y, "vx": vx, "vy": vy}
    ks = list(base)
    nomes, cols = list(ks), [base[k] for k in ks]
    for i in range(4):
        for j in range(i, 4):
            nomes.append(f"{ks[i]}*{ks[j]}")
            cols.append(base[ks[i]] * base[ks[j]])
    for nm, c in (("r", r), ("1/r", 1 / r), ("1/r^2", 1 / r ** 2)):
        nomes.append(nm)
        cols.append(c)
    return nomes, np.stack(cols, -1)


def _covs(F):
    n, p, m = F.shape
    G = F.reshape(-1, m)
    G = G - G.mean(0)
    Tc = G.T @ G / len(G)
    Fc = F - F.mean(1, keepdims=True)
    W = np.einsum("npi,npj->ij", Fc, Fc) / (n * p)
    return W, Tc


def _lam(Q):
    dentro = ((Q - Q.mean(1, keepdims=True)) ** 2).mean()
    total = ((Q - Q.mean()) ** 2).mean()
    return dentro / total if total > 0 else 1.0


def poupanca(lam, N, ntraj):
    return 0.5 * N * log2(1 / max(lam, 1e-300)) - 0.5 * ntraj * log2(N / ntraj)


def estrutura(m, k, N):
    return log2(MAX_K) + log2(comb(m, k)) + 0.5 * (k - 1) * log2(N)


def inventa(Ttr, Tho):
    nomes, Ftr = alfabeto(Ttr)
    _, Fho = alfabeto(Tho)
    m = len(nomes)
    W, Tc = _covs(Ftr)
    Ntr, Nho = Ftr.shape[0] * Ftr.shape[1], Fho.shape[0] * Fho.shape[1]
    cands = []
    for k in range(1, MAX_K + 1):
        for s in itertools.combinations(range(m), k):
            s = list(s)
            Ts = Tc[np.ix_(s, s)]
            ev = np.linalg.eigvalsh(Ts)
            if ev.min() <= 1e-9 * max(ev.max(), 1e-300):
                continue
            try:
                vals, vecs = eigh(W[np.ix_(s, s)], Ts)
            except LinAlgError:
                continue
            w = vecs[:, 0]
            g = poupanca(max(vals[0], 1e-300), Ntr, Ftr.shape[0]) - estrutura(m, k, Ntr)
            cands.append((g, s, w, vals[0]))
    cands.sort(key=lambda c: -c[0])
    top = []
    for g, s, w, lam in cands[:3]:
        lho = _lam(Fho[..., s] @ w)
        top.append((g, s, w, lam, lho, poupanca(lho, Nho, Fho.shape[0])))
    g, s, w, lam, lho, gho = top[0]
    aceito = g > 0 and gho > 0
    return aceito, top, nomes


def formula(s, w, nomes):
    w = w / np.abs(w).max()
    return " ".join(f"{c:+.3f} {nomes[i]}" for c, i in zip(w, s))


def grandeza(s, w, Tref):
    """A funcao estado -> q, na escala do treino."""
    _, F = alfabeto(Tref)
    esc = (F[..., s] @ w).std()
    return lambda S: (alfabeto(S)[1][..., s] @ w) / esc


# ─── fase 2: a lei, com e sem a grandeza ─────────────────────────────────────

def _lei(k, extras, S2, S3):
    O.EXTRAS.clear()
    O.EXTRAS.update(extras)
    try:
        rng = np.random.default_rng(O3.SEMENTE + 1000 * k + 97 * 2 + 500)
        S = O.estados(rng, O.N_OBS)
        D = O.mede("einstein", S, rng)
        ac = O3.descobre(S, D, False, rng, "gauss")
        rx = O2._r2(O.anomalia("einstein", S3), O.preve(ac, S3, False))
        r2 = O2._r2(O.anomalia("einstein", S2), O.preve(ac, S2, False))
        return ac, r2, rx
    finally:
        O.EXTRAS.clear()


def _um_termo_q(ac):
    return (len(ac) == 1 and ac[0][0] == "lei" and ac[0][1] == "r^"
            and ac[0][2] == ("r", "q") and tuple(ac[0][3]) == (-4, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=10)
    a = ap.parse_args()
    n = a.sementes
    S2 = O.estados(np.random.default_rng(O3.SEMENTE + 5), 2000)
    S3 = O2.estados_ext(np.random.default_rng(O3.SEMENTE + 6), 2000)
    Sh = O.estados(np.random.default_rng(SEMENTE + 9), 2000)
    h_ref = Sh[:, 0] * Sh[:, 3] - Sh[:, 1] * Sh[:, 2]

    print("=" * 76)
    print("INVENTA — achar a grandeza conservada, e usa-la na lei seguinte")
    print("=" * 76)
    t0 = time.time()
    c = {"i0": 0, "i1": 0, "base": 0, "q": 0, "q1": 0, "p": 0}
    for k in range(n):
        print(f"\n  ===== semente {k} =====")
        Qk = None
        for mundo in ("einstein", "permutado", "arrasto"):
            rng = np.random.default_rng(SEMENTE + 1000 * k + 7 * len(mundo))
            fonte = "einstein" if mundo == "permutado" else mundo
            Ttr = trajetorias(fonte, rng, N_TRAJ)
            Tho = trajetorias(fonte, rng, N_TRAJ)
            if mundo == "permutado":
                for T in (Ttr, Tho):
                    flat = T.reshape(-1, 4)
                    flat[:] = flat[rng.permutation(len(flat))]
            aceito, top, nomes = inventa(Ttr, Tho)
            print(f"    [{mundo}] {'INVENTOU' if aceito else 'nada'}")
            for g, s, w, lam, lho, gho in top:
                print(f"        {formula(s, w, nomes):42s} lambda={lam:.1e} "
                      f"fora={lho:.1e}  [{g:.0f} bits, fora {gho:.0f}]")
            if mundo == "einstein" and aceito:
                _g, s, w, *_ = top[0]
                Qk = grandeza(s, w, Ttr)
                cr = abs(np.corrcoef(Qk(Sh), h_ref)[0, 1])
                print(f"        |corr(q, h)| = {cr:.5f}")
                c["i0"] += cr >= 0.999
            if mundo == "permutado" and not aceito:
                c["i1"] += 1

        ac, _r2, rx = _lei(k, {}, S2, S3)
        c["base"] += rx >= 0.98
        print(f"    lei SEM extra      extrap={rx:6.3f}   {O.escreve(ac)}")
        if Qk is not None:
            ac, _r2, rx = _lei(k, {"q": Qk}, S2, S3)
            c["q"] += rx >= 0.98
            c["q1"] += _um_termo_q(ac) and rx >= 0.98
            print(f"    lei COM q inventado extrap={rx:6.3f}   {O.escreve(ac)}")
        pl = {"p": lambda S: S[:, 0] * S[:, 2] + S[:, 1] * S[:, 3]}
        ac, _r2, rx = _lei(k, pl, S2, S3)
        c["p"] += rx >= 0.98
        print(f"    lei COM placebo p   extrap={rx:6.3f}   {O.escreve(ac)}")

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("=" * 76)
    print(f"I0 {'.' if c['i0'] >= n - 1 else 'FALHOU .'} inventou h (|corr|>=0,999): "
          f"{c['i0']}/{n}")
    print(f"I1 {'.' if c['i1'] >= n - 1 else 'FALHOU .'} permutado sem invencao: "
          f"{c['i1']}/{n}")
    print(f"I2 {'.' if c['q1'] >= n - 1 else 'FALHOU .'} com q: UM termo q^2 r^-4 e "
          f"extrapola: {c['q1']}/{n}   (extrapola de qualquer jeito: {c['q']}/{n})")
    print(f"I3 {'.' if c['p'] <= 8 else 'FALHOU .'} placebo extrapola: {c['p']}/{n}")
    print(f"I4 {'.' if c['base'] == 7 else 'DIFERE .'} linha de base sem extra: "
          f"{c['base']}/{n}  (orbita3 deu 7/10)")


if __name__ == "__main__":
    main()

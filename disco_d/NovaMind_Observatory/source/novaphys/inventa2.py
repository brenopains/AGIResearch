"""
novaphys.inventa2 — invencao sobre invencao: o laco composto
==============================================================

DE ONDE VEM
-------------
`inventa.py`: de trajetorias, o agente inventou q = -x vy + y vx (o momento
angular, 10/10) e q pagou na lei seguinte (7/10 -> 10/10, placebo 7/10).

A pergunta agora e a da ciencia que ACUMULA: um conceito inventado torna possivel
inventar o PROXIMO? No mundo einstein a energia conservada e

    E = v^2/2 - 1/r - (A/3) h^2 / r^3

O ultimo termo — a correcao relativistica da energia — exige h. Com o alfabeto
original (monomios de grau <= 2 em x, y, vx, vy e potencias de r), h^2/r^3 NAO e
escrevivel: h^2 e de grau 4. Sem o conceito, a melhor "energia" que o agente pode
escrever e a de Newton, v^2/2 - 1/r, que so e APROXIMADAMENTE conservada aqui.

O MECANISMO
-------------
Fase 1 (a de `inventa`): inventa q das trajetorias.
Fase 2: busca uma SEGUNDA grandeza conservada, com ate 4 monomios, que traga
informacao NOVA: cada monomio e residualizado contra um polinomio em q (1, q, q^2,
q^3). O que e funcao de q some; redescobrir h fica impossivel.

Tres alfabetos para a fase 2, todos com a base de 18 (os 17 + 1/r^3, a MESMA para
os tres):
    A  base                                   (sem conceito novo)
    B  base + q^a r^b                         (o conceito inventado)
    C  base + p^a r^b,  p = x vx + y vy       (placebo: mesma forma, nao conservado)
    a em {1, 2};  b em {1, -1, -2, -3}

A regra "o conceito novo se combina com as potencias de r ja existentes" e MINHA —
e uma escolha de desenho, dita aqui.

PRE-REGISTRO, ANTES DE RODAR — semente 20260917, 10 sementes
--------------------------------------------------------------
  C0  com B, o agente ESCREVE a energia de Einstein: o subconjunto escolhido e
      {vx^2, vy^2, 1/r, q^2/r^3}, com as razoes vy^2/vx^2 e (1/r)/vx^2 a 10% de
      1 e -2, e q^2 r^-3 / vx^2 a 25% de -(2A/3) c^2  (c = h/q) — em >= 9/10
  C1  sem o conceito (A): a melhor segunda grandeza e so aproximada:
      lambda_A >= 10 lambda_B em >= 9/10
  C2  placebo (C): lambda_C >= 10 lambda_B em >= 9/10
  N   trajetorias embaralhadas (q aplicado a elas): nenhuma segunda grandeza
      aceita em >= 9/10

PREVISAO: C0 8-10/10 (o termo q^2/r^3 e ~2% da energia; o coeficiente pode sair
ruidoso). C1 10/10. C2 10/10. N 10/10.

RESULTADO (ruido 0,002): C0 9/10 — escreveu +0,50 vx^2 +0,50 vy^2 -1,00/r
-0,013 q^2 r^-3, a correcao relativistica da energia. N 10/10. MAS C1 e C2 FALHARAM
0/10: lambda_A / lambda_B ficou entre 1,0 e 1,25, nao 10. Sem o conceito, A e C
escreveram um SUBSTITUTO: -0,005/r^3 ou -0,017/r^2 (h^2 trocado pela media) —
epiciclo de novo. Previ 10x sem calcular a magnitude: a variacao da correcao
dentro de uma orbita e menor que o ruido de medida da energia (v*sigma ~ 0,002).
Dentro do braco B, q^2/r^3 e 1/r^3 estavam os dois disponiveis e q^2/r^3 venceu
9/10 — ha sinal, com margem pequena.

SEGUIMENTO DECLARADO, ANTES DE RODAR: mesma semente, ruido 10x menor (0,0002).
Pergunta: a vantagem do conceito e limitada pela qualidade do dado?
Previsao: lambda_A / lambda_B >= 10 em >= 9/10 e C0 10/10. Se nao, o limite
nao e o ruido e eu nao entendo o que e.

RESULTADO DO SEGUIMENTO: C0 10/10. lambda_B caiu ao piso do ruido (~1e-5);
lambda_A travou em ~1e-4 — o erro sistematico do substituto (h^2 pela media).
Razao A/B: mediana ~7,7 (2,3 a 21,6), contra ~1,1 com ruido 0,002. O portao de
10x passou so 2/10: FALHOU como registrado, na direcao que a hipotese preve
(razao ~ 1/sigma^2 ate o piso sistematico). O limite era o dado, e o meu 10x
estava na mediana esperada, nao abaixo dela.
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
from novaphys import inventa as I

SEMENTE = 20260917
MAX_K2 = 4
POT = ((1, "r"), (-1, "r^-1"), (-2, "r^-2"), (-3, "r^-3"))


def base(S):
    nomes, F = I.alfabeto(S)
    r = np.hypot(S[..., 0], S[..., 1])
    return nomes + ["1/r^3"], np.concatenate([F, (1 / r ** 3)[..., None]], -1)


def com_conceito(S, nome, f):
    nomes, F = base(S)
    q = f(S)
    r = np.hypot(S[..., 0], S[..., 1])
    extra_n, extra_c = [], []
    for a in (1, 2):
        for b, rn in POT:
            extra_n.append(f"{nome}^{a}*{rn}")
            extra_c.append(q ** a * r ** b)
    return nomes + extra_n, np.concatenate([F, np.stack(extra_c, -1)], -1)


def _poli(q):
    return np.stack([np.ones_like(q), q, q ** 2, q ** 3], -1)


def residualiza(F, q, beta=None):
    m = F.shape[-1]
    G = F.reshape(-1, m)
    X = _poli(q.reshape(-1))
    if beta is None:
        beta, *_ = np.linalg.lstsq(X, G, rcond=None)
    return (G - X @ beta).reshape(F.shape), beta


def estrutura(m, k, N):
    return log2(MAX_K2) + log2(comb(m, k)) + 0.5 * (k - 1) * log2(N)


def segunda(Ftr, Fho, qtr, qho):
    Rtr, beta = residualiza(Ftr, qtr)
    Rho, _ = residualiza(Fho, qho, beta)
    m = Ftr.shape[-1]
    W, Tc = I._covs(Rtr)
    Ntr = Ftr.shape[0] * Ftr.shape[1]
    Nho = Fho.shape[0] * Fho.shape[1]
    cands = []
    for k in range(1, MAX_K2 + 1):
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
            lam = max(vals[0], 1e-300)
            g = I.poupanca(lam, Ntr, Ftr.shape[0]) - estrutura(m, k, Ntr)
            cands.append((g, s, vecs[:, 0], lam))
    cands.sort(key=lambda c: -c[0])
    g, s, w, lam = cands[0]
    lho = I._lam(Rho[..., s] @ w)
    aceito = g > 0 and I.poupanca(lho, Nho, Fho.shape[0]) > 0
    return aceito, s, w, lam, lho, g


def escreveu_E(s, w, nomes, c):
    alvo = {"vx*vx", "vy*vy", "1/r", "q^2*r^-3"}
    if {nomes[i] for i in s} != alvo:
        return False
    k = {nomes[i]: wi for i, wi in zip(s, w)}
    a = k["vx*vx"]
    rq = -(2 * O.A_EIN / 3) * c ** 2
    return (abs(k["vy*vy"] / a - 1) <= 0.10 and abs(k["1/r"] / a + 2) <= 0.20
            and abs(k["q^2*r^-3"] / a - rq) <= 0.25 * abs(rq))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=10)
    ap.add_argument("--sig", type=float, default=I.SIG)
    a = ap.parse_args()
    n = a.sementes
    I.SIG = a.sig
    Sh = O.estados(np.random.default_rng(SEMENTE + 9), 2000)
    h_ref = Sh[:, 0] * Sh[:, 3] - Sh[:, 1] * Sh[:, 2]
    placebo = lambda S: S[..., 0] * S[..., 2] + S[..., 1] * S[..., 3]

    print("=" * 76)
    print("INVENTA 2 — o conceito inventado permite inventar o proximo?")
    print("=" * 76)
    t0 = time.time()
    c0 = c1 = c2 = nn = 0
    for k in range(n):
        rng = np.random.default_rng(SEMENTE + 1000 * k)
        Ttr = I.trajetorias("einstein", rng, I.N_TRAJ)
        Tho = I.trajetorias("einstein", rng, I.N_TRAJ)
        aceito, top, nomes1 = I.inventa(Ttr, Tho)
        print(f"\n  ===== semente {k} =====")
        if not aceito:
            print("    fase 1 nao inventou nada — semente perdida")
            continue
        _g, s1, w1, *_ = top[0]
        Q = I.grandeza(s1, w1, Ttr)
        cq = float((Q(Sh) @ h_ref) / (Q(Sh) @ Q(Sh)))
        print(f"    fase 1: q = {I.formula(s1, w1, nomes1)}   (h = {cq:.3f} q)")

        lams = {}
        for braco, fz in (("A", None), ("B", ("q", Q)), ("C", ("p", placebo))):
            if fz is None:
                nm, Ftr = base(Ttr)
                _, Fho = base(Tho)
            else:
                nm, Ftr = com_conceito(Ttr, *fz)
                _, Fho = com_conceito(Tho, *fz)
            ac2, s, w, lam, lho, g = segunda(Ftr, Fho, Q(Ttr), Q(Tho))
            lams[braco] = lam
            ok = braco == "B" and escreveu_E(s, w, nm, cq)
            if ok:
                c0 += 1
            print(f"    [{braco}] {'aceita' if ac2 else 'nada':6s} lambda={lam:.1e} "
                  f"fora={lho:.1e}  {I.formula(s, w / w[[nm[i] for i in s].index('vx*vx')] if 'vx*vx' in [nm[i] for i in s] else w, nm)}"
                  f"{'   <- ENERGIA DE EINSTEIN' if ok else ''}")
        c1 += lams["A"] >= 10 * lams["B"]
        c2 += lams["C"] >= 10 * lams["B"]

        # nulo: trajetorias embaralhadas, com o q ja inventado aplicado a elas
        Tp, Tq = Ttr.copy(), Tho.copy()
        for T in (Tp, Tq):
            flat = T.reshape(-1, 4)
            flat[:] = flat[rng.permutation(len(flat))]
        nm, Fp = com_conceito(Tp, "q", Q)
        _, Fq = com_conceito(Tq, "q", Q)
        acn, *_ = segunda(Fp, Fq, Q(Tp), Q(Tq))
        nn += not acn
        print(f"    [nulo] {'ACEITOU (falso)' if acn else 'nada'}")

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("=" * 76)
    print(f"C0 {'.' if c0 >= n - 1 else 'FALHOU .'} escreveu a energia de Einstein com q: {c0}/{n}")
    print(f"C1 {'.' if c1 >= n - 1 else 'FALHOU .'} sem o conceito, 10x pior: {c1}/{n}")
    print(f"C2 {'.' if c2 >= n - 1 else 'FALHOU .'} placebo 10x pior: {c2}/{n}")
    print(f"N  {'.' if nn >= n - 1 else 'FALHOU .'} nulo sem segunda grandeza: {nn}/{n}")


if __name__ == "__main__":
    main()

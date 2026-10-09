"""
novaphys.orbita — Netuno, Vulcano, o Sol achatado ou Einstein? Escrever a equacao
===================================================================================

A HISTORIA QUE ESTE MUNDO REPETE
----------------------------------
1846: a orbita de Urano desviava de Newton. Le Verrier supos um planeta escondido,
calculou onde estaria, e Netuno estava la. Objeto oculto, lei intacta.

1859: a orbita de Mercurio tambem desviava (o perielio avanca 43"/seculo a mais).
Le Verrier aplicou a MESMA logica e propos Vulcano, um planeta dentro da orbita de
Mercurio. Nunca foi achado. Dicke propos depois um Sol ACHATADO (quadrupolo), que
da uma forca extra ~ 1/r^4. A resposta certa era outra: MUDAR A LEI. Einstein, 1915.

A mesma anomalia, tres causas. O que separa Einstein do Sol achatado e sutil: os
dois dao um termo ~ 1/r^4, mas o de Einstein e  -3GM h^2/(c^2 r^4)  — depende do
MOMENTO ANGULAR do corpo, isto e, da velocidade. O achatado nao.

O MUNDO
---------
Plano, G*M = 1. A TEORIA VELHA E DADA: a = -r^ / r^2. Um "laboratorio" mede a
aceleracao de N corpos em estados (posicao, velocidade) variados, com ruido. Cinco
mundos:

    nulo       Newton puro + ruido
    permutado  o residuo do mundo einstein EMBARALHADO entre os estados (mesma
               distribuicao, nenhuma ligacao com o estado) — o nulo do projeto
    einstein   + ( -A h^2 / r^4 ) r^
    oblato     + ( -K / r^4 ) r^                (Sol achatado)
    vulcano    + atracao de uma massa oculta m em p, fora da regiao observada

O AGENTE
----------
Revisao de teoria por MDL, a mesma regua do nucleo ARC com numeros reais:
residuo = medido - teoria velha. Sobre o residuo, uma PILHA de modificacoes, cada
uma escolhida pelo ganho em bits, parando quando nada mais paga:

  - TERMO DE LEI:   c * PRODUTO(var_i ^ e_i) * direcao
                    ate 3 fatores, expoentes em {-4..-1, 1..4}, direcao em
                    {r^, t^ (tangencial), v^}; variaveis do "caderno":
                    x, y, vx, vy, r, v, vr (vel. radial), th (angulo) e h
  - OBJETO OCULTO:  massa m em (px, py), ajustada por minimos quadrados

  custo = bits da estrutura + 0,5 log2(N) por parametro real
          + (N/2) log2(erro quadratico)          (codigo gaussiano do residuo)

O custo da estrutura inclui log2 do numero de candidatos: e a correcao de
Bonferroni escrita em bits. Com ~136 mil termos candidatos, o termo espurio
precisa pagar ~20 bits.

A SAIDA E UMA EQUACAO ESCRITA, por exemplo:   a = -r^/r^2  - 0.0300 h^2 r^-4 r^

O QUE ESTE DEGRAU NAO TESTA, DITO ANTES
------------------------------------------
Eu entrego o caderno de variaveis, INCLUSIVE h. Redescobrir lei conhecida com
regressao simbolica ja foi feito (AI Feynman, Udrescu & Tegmark 2020; Lemos et al.
2022 com dados reais do sistema solar). O que este degrau mede e (1) se a regua
separa as TRES causas da mesma anomalia e (2) onde ela quebra. A condicao SEM h no
caderno e a primeira sonda de gap.

PRE-REGISTRO, ANTES DE RODAR — 5 sementes por condicao
--------------------------------------------------------
  O0  NULOS: nulo e permutado nao aceitam modificacao nenhuma em >= 4/5 cada
  O1  EINSTEIN (caderno com h): modificacao de LEI, R2 >= 0,98 sobre a anomalia
      verdadeira em estados novos, e o termo usa h — em >= 4/5
  O2  OBLATO: LEI, R2 >= 0,98, e NENHUM termo usa h — em >= 4/5
  O3  VULCANO: a primeira modificacao e OBJETO, a menos de 0,5 da posicao real —
      em >= 4/5; e einstein/oblato NUNCA comecam por objeto
  O4  SONDA DE GAP, caderno SEM h: einstein com R2 >= 0,98 em >= 4/5?
      Sem h o termo exato nao e um monomio. Mas h^2/r^4 = (v^2 - vr^2)/r^2 —
      e decomponivel em DOIS monomios, e a pilha poderia achar os dois em sequencia.

SEMENTE PRE-REGISTRADA: 20260913

PREVISAO: O0, O1, O2 passam (5/5) — com o caderno certo isto e facil, e esse e o
ponto. O3 passa em >= 4/5. O4 passa por DECOMPOSICAO (v^2 r^-2 primeiro, vr^2 r^-2
depois) em >= 4/5 — se eu estiver certo, esta sonda NAO revela o gap, e ele tera
que ser buscado no degrau seguinte (so posicoes, sem forca medida).
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")    # um nucleo so: nao travar a maquina

import argparse
import itertools
import time
from math import comb, log2

import numpy as np
from scipy.optimize import least_squares

SEMENTE = 20260913
A_EIN = 0.03
K_OBL = 0.03
M_VUL = 0.08
P_VUL = np.array([4.0, 1.5])
SIGMA = 0.004
N_OBS = 400
EXPS = (-4, -3, -2, -1, 1, 2, 3, 4)
MAX_FAT = 3
MAX_TERMOS = 4
PISO = 1e-3
EXTRAS = {}      # nome -> funcao(S); preenchido por `inventa.py`, vazio no resto
DIR_EXTRA = False   # direcao n^ (v girada); ligada so pela `generalidade.py`
DIRS_EXTRA = {}     # nome -> funcao(S) -> (N, 2); gramatica crescida pela `gramatica.py`
A_MAG = 0.06


# ─── o mundo ─────────────────────────────────────────────────────────────────

def estados(rng, n):
    r = rng.uniform(0.7, 3.0, n)
    th = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0.6, 1.3, n) / np.sqrt(r)
    ang = rng.uniform(-0.6, 0.6, n)
    sentido = np.where(rng.random(n) < 0.3, -1.0, 1.0)
    vt, vr = v * np.cos(ang) * sentido, v * np.sin(ang)
    c, s = np.cos(th), np.sin(th)
    return np.stack([r * c, r * s, vr * c - vt * s, vr * s + vt * c], 1)


def newton(S):
    rv = S[:, :2]
    r = np.linalg.norm(rv, axis=1, keepdims=True)
    return -rv / r ** 3


def _obj_base(S, p):
    d = S[:, :2] - p
    dn = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 0.2)
    return -d / dn ** 3


def anomalia(mundo, S):
    x, y, vx, vy = S.T
    r = np.hypot(x, y)
    h = x * vy - y * vx
    rh = S[:, :2] / r[:, None]
    if mundo == "einstein":
        return -(A_EIN * h ** 2 / r ** 4)[:, None] * rh
    if mundo == "oblato":
        return -(K_OBL / r ** 4)[:, None] * rh
    if mundo == "vulcano":
        return M_VUL * _obj_base(S, P_VUL)
    if mundo == "magnetico":
        # forca PERPENDICULAR a velocidade (nao realiza trabalho: conserva a
        # energia E = v^2/2 - 1/r) com intensidade proporcional a E; nao e central,
        # entao h NAO se conserva. O conceito necessario aqui e E, nao h.
        E = 0.5 * (vx ** 2 + vy ** 2) - 1 / r
        v = np.hypot(vx, vy)
        return (A_MAG * E)[:, None] * np.stack([-vy / v, vx / v], 1)
    return np.zeros((len(S), 2))


def mede(mundo, S, rng):
    fonte = "einstein" if mundo == "permutado" else mundo
    D = anomalia(fonte, S) + rng.normal(0, SIGMA, (len(S), 2))
    if mundo == "permutado":
        D = D[rng.permutation(len(S))]
    return D                      # medido - teoria velha


# ─── o caderno de variaveis ─────────────────────────────────────────────────

def caderno(S, com_h):
    x, y, vx, vy = S.T
    r = np.hypot(x, y)
    V = {"x": x, "y": y, "vx": vx, "vy": vy, "r": r, "v": np.hypot(vx, vy),
         "vr": (x * vx + y * vy) / r, "th": np.arctan2(y, x)}
    if com_h:
        V["h"] = x * vy - y * vx
    for nome, f in EXTRAS.items():      # grandezas que o PROPRIO agente inventou
        V[nome] = f(S)
    nomes = list(V)
    return nomes, np.stack([V[n] for n in nomes])


def direcoes(S):
    x, y, vx, vy = S.T
    r, v = np.hypot(x, y), np.hypot(vx, vy)
    out = {"r^": np.stack([x / r, y / r], 1),
           "t^": np.stack([-y / r, x / r], 1),
           "v^": np.stack([vx / v, vy / v], 1)}
    if DIR_EXTRA:      # v girada 90 graus; desligado em tudo que ja foi medido
        out["n^"] = np.stack([-vy / v, vx / v], 1)
    for nome, f in DIRS_EXTRA.items():   # direcoes que o PROPRIO agente criou
        out[nome] = f(S)
    return out


_CANDS = {}


def candidatos(P):
    if P not in _CANDS:
        meta, linhas = [], []
        for k in range(1, MAX_FAT + 1):
            for vs in itertools.combinations(range(P), k):
                for es in itertools.product(EXPS, repeat=k):
                    e = np.zeros(P)
                    e[list(vs)] = es
                    linhas.append(e)
                    meta.append((k, vs, es))
        _CANDS[P] = (np.array(linhas), meta)
    return _CANDS[P]


def _F(E, X):
    la = np.log(np.maximum(np.abs(X), PISO))
    neg = (X < 0).astype(float)
    par = (np.abs(E) % 2) @ neg
    return (1 - 2 * (par % 2)) * np.exp(E @ la)


# ─── a regua ─────────────────────────────────────────────────────────────────

def bits_termo(P, k, neff):
    return (1 + log2(3 + DIR_EXTRA + len(DIRS_EXTRA)) + log2(MAX_FAT)
            + log2(comb(P, k) * len(EXPS) ** k) + 0.5 * log2(neff))


def bits_objeto(neff):
    return 1 + 1.5 * log2(neff)


def ganho(sse, red, neff, estrutura):
    depois = max(sse - red, 1e-300)
    return 0.5 * neff * log2(sse / depois) - estrutura


# ─── o agente ────────────────────────────────────────────────────────────────

def _melhor_termo(R, X, dirs, P, neff):
    E, meta = candidatos(P)
    sse = (R ** 2).sum()
    proj = {d: (R * u).sum(1) for d, u in dirs.items()}
    cust = np.array([bits_termo(P, k, neff) for k in range(1, MAX_FAT + 1)])
    kv = np.array([m[0] for m in meta])
    melhor = None
    for c0 in range(0, len(E), 8000):
        F = _F(E[c0:c0 + 8000], X)
        den = (F ** 2).sum(1)
        ok = np.isfinite(den) & (den > 0)
        for d, pj in proj.items():
            num = F @ pj
            red = np.where(ok, num ** 2 / np.where(ok, den, 1), 0)
            red = np.minimum(red, sse * (1 - 1e-12))
            g = 0.5 * neff * np.log2(sse / (sse - red)) - cust[kv[c0:c0 + 8000] - 1]
            i = int(np.argmax(g))
            if melhor is None or g[i] > melhor[0]:
                melhor = (float(g[i]), c0 + i, d, float(num[i] / den[i]))
    return melhor


def _melhor_objeto(R, S, neff):
    sse = (R ** 2).sum()
    melhor = None
    grade = np.arange(-6, 6.01, 0.5)
    for px in grade:
        for py in grade:
            p = np.array([px, py])
            b = _obj_base(S, p)
            den = (b ** 2).sum()
            m = (R * b).sum() / den
            red = m * m * den
            if melhor is None or red > melhor[0]:
                melhor = (red, m, p)
    _, m0, p0 = melhor
    fit = least_squares(lambda q: (R - q[0] * _obj_base(S, q[1:])).ravel(),
                        np.array([m0, *p0]))
    m, px, py = fit.x
    red = sse - ((R - m * _obj_base(S, np.array([px, py]))) ** 2).sum()
    return ganho(sse, red, neff, bits_objeto(neff)), (m, px, py)


def descobre(S, D, com_h):
    nomes, X = caderno(S, com_h)
    P = len(nomes)
    E, meta = candidatos(P)
    dirs = direcoes(S)
    neff = D.size
    R = D.copy()
    aceitos = []
    for _ in range(MAX_TERMOS):
        gt, i, d, c = _melhor_termo(R, X, dirs, P, neff)
        go, (m, px, py) = _melhor_objeto(R, S, neff)
        if max(gt, go) <= 0:
            break
        if gt >= go:
            k, vs, es = meta[i]
            t = ("lei", d, tuple(nomes[j] for j in vs), es, c, gt)
            R = R - (c * _F(E[i:i + 1], X)[0])[:, None] * dirs[d]
        else:
            t = ("objeto", m, px, py, go)
            R = R - m * _obj_base(S, np.array([px, py]))
        aceitos.append(t)
    return aceitos


def preve(aceitos, S, com_h):
    nomes, X = caderno(S, com_h)
    idx = {n: j for j, n in enumerate(nomes)}
    dirs = direcoes(S)
    pred = np.zeros((len(S), 2))
    for t in aceitos:
        if t[0] == "lei":
            _, d, vs, es, c, _g = t
            e = np.zeros(len(nomes))
            for v, x in zip(vs, es):
                e[idx[v]] = x
            pred += (c * _F(e[None, :], X)[0])[:, None] * dirs[d]
        else:
            _, m, px, py, _g = t
            pred += m * _obj_base(S, np.array([px, py]))
    return pred


def escreve(aceitos):
    partes = ["a = -r^/r^2"]
    for t in aceitos:
        if t[0] == "lei":
            _, d, vs, es, c, g = t
            fat = " ".join(f"{v}^{e}" for v, e in zip(vs, es))
            partes.append(f"{c:+.4f} {fat} {d}   [{g:.0f} bits]")
        else:
            _, m, px, py, g = t
            partes.append(f"{m:+.4f} (massa oculta em ({px:.2f}, {py:.2f}))"
                          f"   [{g:.0f} bits]")
    return "\n        ".join(partes)


# ─── o experimento ───────────────────────────────────────────────────────────

CONDICOES = [("nulo", True), ("permutado", True), ("einstein", True),
             ("oblato", True), ("vulcano", True),
             ("einstein", False), ("oblato", False)]


def episodio(mundo, com_h, k):
    # indice fixo por mundo: hash() de str muda entre processos no Python
    iw = ("nulo", "permutado", "einstein", "oblato", "vulcano").index(mundo)
    rng = np.random.default_rng(SEMENTE + 1000 * k + 97 * iw
                                + (0 if com_h else 500))
    S = estados(rng, N_OBS)
    D = mede(mundo, S, rng)
    aceitos = descobre(S, D, com_h)
    S2 = estados(np.random.default_rng(SEMENTE + 77 + k), 2000)
    alvo = anomalia(mundo if mundo != "permutado" else "nulo", S2)
    pred = preve(aceitos, S2, com_h)
    tot = (alvo ** 2).sum()
    r2 = 1 - ((alvo - pred) ** 2).sum() / tot if tot > 0 else float("nan")
    return aceitos, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=5)
    a = ap.parse_args()
    n = a.sementes

    print("=" * 76)
    print("ORBITA — Netuno, Vulcano, o Sol achatado ou Einstein?")
    print("=" * 76)
    t0 = time.time()
    res = {}
    for mundo, com_h in CONDICOES:
        rot = f"{mundo}{'' if com_h else ' (sem h)'}"
        print(f"\n  --- {rot} ---")
        res[(mundo, com_h)] = []
        for k in range(n):
            aceitos, r2 = episodio(mundo, com_h, k)
            res[(mundo, com_h)].append((aceitos, r2))
            print(f"    s{k}  R2={r2:6.3f}   {escreve(aceitos)}")
    print(f"\n  ({time.time()-t0:.0f}s)\n")

    def primeiro(ac):
        return ac[0][0] if ac else "nada"

    def usa_h(ac):
        return any(t[0] == "lei" and "h" in t[2] for t in ac)

    print("=" * 76)
    o0 = all(sum(1 for ac, _ in res[(m, True)] if not ac) >= n - 1
             for m in ("nulo", "permutado"))
    print(f"O0 {'.' if o0 else 'FALHOU .'} nulos sem modificacao: "
          + ", ".join(f"{m} {sum(1 for ac, _ in res[(m, True)] if not ac)}/{n}"
                      for m in ("nulo", "permutado")))
    e1 = sum(1 for ac, r2 in res[("einstein", True)]
             if primeiro(ac) == "lei" and r2 >= 0.98 and usa_h(ac))
    print(f"O1 {'.' if e1 >= n - 1 else 'FALHOU .'} einstein: lei com h, "
          f"R2>=0,98 em {e1}/{n}")
    o2 = sum(1 for ac, r2 in res[("oblato", True)]
             if primeiro(ac) == "lei" and r2 >= 0.98 and not usa_h(ac))
    print(f"O2 {'.' if o2 >= n - 1 else 'FALHOU .'} oblato: lei sem h, "
          f"R2>=0,98 em {o2}/{n}")
    v3 = sum(1 for ac, _ in res[("vulcano", True)]
             if primeiro(ac) == "objeto"
             and np.hypot(ac[0][2] - P_VUL[0], ac[0][3] - P_VUL[1]) < 0.5)
    trai = sum(1 for m in ("einstein", "oblato") for ac, _ in res[(m, True)]
               if primeiro(ac) == "objeto")
    print(f"O3 {'.' if v3 >= n - 1 and trai == 0 else 'FALHOU .'} vulcano: "
          f"objeto no lugar certo em {v3}/{n}; leis confundidas com objeto: {trai}")
    o4 = sum(1 for ac, r2 in res[("einstein", False)] if r2 >= 0.98)
    print(f"O4 {'.' if o4 >= n - 1 else 'GAP .'} einstein SEM h no caderno: "
          f"R2>=0,98 em {o4}/{n}")


if __name__ == "__main__":
    main()

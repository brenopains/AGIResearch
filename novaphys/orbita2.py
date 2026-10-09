"""
novaphys.orbita2 — o nulo pegou a regua, e epiciclo nao e lei
===============================================================

O QUE A PRIMEIRA RODADA MOSTROU (`orbita.py`, semente 20260913)
-----------------------------------------------------------------
Passou: einstein 5/5 escreveu  -0,030 h^2 r^-4 r^  (verdade -0,03); oblato 5/5
escreveu  -0,030 r^-4 r^  SEM h; vulcano 5/5 achou a massa a ~0,1 do lugar, com
a massa certa a 5%. Nenhuma lei foi confundida com objeto.

FALHOU O0: no mundo PERMUTADO o agente aceitou modificacao em 4 de 5. Dois
defeitos da MINHA regua, nenhum do mundo:
  1. a busca da posicao do objeto (625 pontos de grade + refinamento continuo)
     nao pagava bit nenhum — comparacoes multiplas de graca; e aceitou MASSA
     NEGATIVA;
  2. termos "espinho" (vx^-4 th^3, x^-1 vr^-1 h^4) explodem onde uma variavel
     passa perto de zero e compram UM ponto discrepante. O codigo gaussiano com
     um sigma so e mal especificado para cauda pesada, e o residuo permutado
     tem cauda pesada.

E O4 "PASSOU" PELA LETRA E ERROU NO CONTEUDO. Sem h no caderno, a lei exata e
(v^2 - vr^2)/r^2, dois monomios. O agente escreveu
    -0,0265 r^-2 v^2  +  0,06 vr^4  (+ as vezes v^4)
coeficiente enviesado e forma errada. R2 0,994-0,997 dentro da faixa observada,
porque o portao de R2 >= 0,98 aceita APROXIMACAO. Isso e EPICICLO: sem a grandeza
certa, empilhar correcoes que ajustam bem onde se olhou. A causa mecanica e a
selecao GULOSA sem reajuste: o primeiro termo absorve parte do segundo e o
coeficiente nunca e corrigido (cegueira do ganho imediato, de novo).

O que separa lei de epiciclo nao e o ajuste — e a EXTRAPOLACAO. Ptolomeu ajustava
bem; so nao previa fora do que tinha visto.

O CONSERTO, ESCOLHIDO PELO PRINCIPIO E NAO PELO SINTOMA
---------------------------------------------------------
  a. GANHO FORA DA AMOSTRA: toda modificacao so e aceita se melhorar o erro em
     metade dos dados ajustada na OUTRA metade (duas dobras). E o MDL
     prequencial (Dawid, Rissanen): o modelo tem que prever o que nao viu. Ataca
     os dois defeitos de uma vez — espinho nao prevê o ponto que nao viu, e a
     busca de posicao passa a ser paga pela validacao, nao por uma constante que
     eu escolheria.
  b. SELECAO ORTOGONAL com REAJUSTE CONJUNTO (OLS progressivo): o ganho de um
     candidato e medido depois de remover dele o que os termos ja aceitos
     explicam, e todos os coeficientes sao reajustados juntos a cada passo.
Nao adicionei prior de massa positiva: seria consertar pelo sintoma.

PRE-REGISTRO, ANTES DE RODAR — semente nova 20260914, 5 por condicao
----------------------------------------------------------------------
  E0  DIAGNOSTICO sobre os resultados JA REGISTRADOS da `orbita.py` (sem
      reajuste): R2 de EXTRAPOLACAO (r em [3, 6], velocidade radial ate 1 rad —
      fora de tudo que foi medido). Previsao: einstein COM h >= 0,98 (lei);
      einstein SEM h < 0,9 (epiciclo).
  O0' nulo e permutado sem modificacao em >= 4/5 cada
  O1'-O3' os mesmos criterios da primeira rodada, nao podem regredir
  O5  einstein SEM h, com o conserto: EXTRAPOLACAO R2 >= 0,98 em >= 4/5 — isto
      e, a lei de verdade, (v^2 - vr^2)/r^2, e nao epiciclo

PREVISAO: E0 como escrito. O0' passa. O1'-O3' passam. O5: 3 a 5 de 5 — a
selecao ortogonal deve achar a decomposicao, mas eu ja errei o mecanismo do O4
uma vez hoje.

CALIBRAGEM: na primeira rodada previ O0 passando — errei, otimista, de novo.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from math import log2

import numpy as np
from scipy.optimize import least_squares

from novaphys import orbita as O

SEMENTE = 20260914
TOP = 15


def estados_ext(rng, n):
    """Fora de tudo que foi medido: mais longe e mais radial."""
    r = rng.uniform(3.0, 6.0, n)
    th = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0.6, 1.3, n) / np.sqrt(r)
    ang = rng.uniform(-1.0, 1.0, n)
    sentido = np.where(rng.random(n) < 0.3, -1.0, 1.0)
    vt, vr = v * np.cos(ang) * sentido, v * np.sin(ang)
    c, s = np.cos(th), np.sin(th)
    return np.stack([r * c, r * s, vr * c - vt * s, vr * s + vt * c], 1)


def _r2(alvo, pred):
    tot = (alvo ** 2).sum()
    return 1 - ((alvo - pred) ** 2).sum() / tot if tot > 0 else float("nan")


def _fit_obj(R, S):
    melhor = None
    grade = np.arange(-6, 6.01, 0.5)
    for px in grade:
        for py in grade:
            p = np.array([px, py])
            b = O._obj_base(S, p)
            den = (b ** 2).sum()
            m = (R * b).sum() / den
            if melhor is None or m * m * den > melhor[0]:
                melhor = (m * m * den, m, p)
    _, m0, p0 = melhor
    fit = least_squares(lambda q: (R - q[0] * O._obj_base(S, q[1:])).ravel(),
                        np.array([m0, *p0]))
    return fit.x[0], fit.x[1:].copy()


def _ajusta(bases, D, idx):
    M = np.stack([b[idx].ravel() for b in bases], 1)
    c, *_ = np.linalg.lstsq(M, D[idx].ravel(), rcond=None)
    return c


def _sse_fora(D, bases_por_dobra, dobras):
    """Erro fora da dobra: ajusta em `tr`, mede em `te`."""
    tot = 0.0
    for (tr, te), bases in zip(dobras, bases_por_dobra):
        if not bases:
            tot += (D[te] ** 2).sum()
            continue
        c = _ajusta(bases, D, tr)
        M = np.stack([b[te].ravel() for b in bases], 1)
        tot += ((D[te].ravel() - M @ c) ** 2).sum()
    return tot


def _ortonormal(bases):
    Q = []
    for b in bases:
        v = b.copy()
        for q in Q:
            v = v - (v * q).sum() * q
        nv = np.sqrt((v ** 2).sum())
        if nv > 1e-12:
            Q.append(v / nv)
    return Q


def _top_termos(R, Q, X, dirs, P, neff):
    E, meta = O.candidatos(P)
    sse = (R ** 2).sum()
    kv = np.array([m[0] for m in meta])
    cust = np.array([O.bits_termo(P, k, neff) for k in range(1, O.MAX_FAT + 1)])
    top = []
    for c0 in range(0, len(E), 8000):
        F = O._F(E[c0:c0 + 8000], X)
        den = (F ** 2).sum(1)
        ok = np.isfinite(den) & (den > 0)
        for d, u in dirs.items():
            num = F @ (R * u).sum(1)
            dp = den.copy()
            for q in Q:
                dp -= (F @ (q * u).sum(1)) ** 2
            ok2 = ok & (dp > 1e-9 * np.where(ok, den, 1))
            red = np.where(ok2, num ** 2 / np.where(ok2, dp, 1), 0)
            red = np.minimum(red, sse * (1 - 1e-12))
            g = 0.5 * neff * np.log2(sse / (sse - red)) - cust[kv[c0:c0 + 8000] - 1]
            j = np.argpartition(-g, TOP)[:TOP]
            top += [(float(g[i]), c0 + int(i), d) for i in j]
    top.sort(reverse=True)
    return top[:TOP]


def descobre(S, D, com_h, rng):
    nomes, X = O.caderno(S, com_h)
    P = len(nomes)
    E, meta = O.candidatos(P)
    dirs = O.direcoes(S)
    neff = D.size
    perm = rng.permutation(len(S))
    A, B = perm[: len(S) // 2], perm[len(S) // 2:]
    dobras = ((A, B), (B, A))

    estr, bases, ganhos = [], [], []
    R = D.copy()
    for _ in range(O.MAX_TERMOS):
        antes = _sse_fora(D, (bases, bases), dobras)
        melhor = None
        # termos de lei: os TOP pela selecao ortogonal, julgados FORA da amostra
        for _g, i, d in _top_termos(R, _ortonormal(bases), X, dirs, P, neff):
            b = O._F(E[i:i + 1], X)[0][:, None] * dirs[d]
            depois = _sse_fora(D, (bases + [b], bases + [b]), dobras)
            k = meta[i][0]
            gh = (0.5 * neff * log2(antes / depois)
                  - (O.bits_termo(P, k, neff) - 0.5 * log2(neff)))
            if melhor is None or gh > melhor[0]:
                melhor = (gh, ("lei", d, i), b)
        # objeto oculto: posicao ajustada DENTRO de cada dobra
        bd = []
        for tr, _te in dobras:
            _m, p = _fit_obj(R[tr], S[tr])
            bd.append(bases + [O._obj_base(S, p)])
        depois = _sse_fora(D, bd, dobras)
        go = 0.5 * neff * log2(antes / depois) - 1
        if melhor is None or go > melhor[0]:
            _m, p = _fit_obj(R, S)
            melhor = (go, ("objeto", p), O._obj_base(S, p))
        if melhor[0] <= 0:
            break
        estr.append(melhor[1])
        bases.append(melhor[2])
        ganhos.append(melhor[0])
        c = _ajusta(bases, D, np.arange(len(S)))
        M = np.stack([b.ravel() for b in bases], 1)
        R = D - (M @ c).reshape(D.shape)

    aceitos = []
    if estr:
        c = _ajusta(bases, D, np.arange(len(S)))
        for e, ci, g in zip(estr, c, ganhos):
            if e[0] == "lei":
                _k, vs, es = meta[e[2]]
                aceitos.append(("lei", e[1], tuple(nomes[j] for j in vs), es,
                                float(ci), g))
            else:
                aceitos.append(("objeto", float(ci), e[1][0], e[1][1], g))
    return aceitos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=5)
    a = ap.parse_args()
    n = a.sementes
    S2 = O.estados(np.random.default_rng(SEMENTE + 5), 2000)
    S3 = estados_ext(np.random.default_rng(SEMENTE + 6), 2000)

    print("=" * 76)
    print("ORBITA 2 — ganho fora da amostra, selecao ortogonal, extrapolacao")
    print("=" * 76)

    print("\n  E0 . diagnostico dos resultados JA REGISTRADOS (orbita.py, gulosa):")
    for com_h in (True, False):
        linha = []
        for k in range(n):
            ac, _r = O.episodio("einstein", com_h, k)
            linha.append(_r2(O.anomalia("einstein", S3), O.preve(ac, S3, com_h)))
        print(f"    einstein {'com h' if com_h else 'SEM h'}: extrapolacao R2 = "
              + "  ".join(f"{x:.3f}" for x in linha))

    t0 = time.time()
    res = {}
    for mundo, com_h in O.CONDICOES:
        rot = f"{mundo}{'' if com_h else ' (sem h)'}"
        print(f"\n  --- {rot} ---")
        res[(mundo, com_h)] = []
        iw = ("nulo", "permutado", "einstein", "oblato", "vulcano").index(mundo)
        for k in range(n):
            rng = np.random.default_rng(SEMENTE + 1000 * k + 97 * iw
                                        + (0 if com_h else 500))
            S = O.estados(rng, O.N_OBS)
            D = O.mede(mundo, S, rng)
            ac = descobre(S, D, com_h, rng)
            fonte = "nulo" if mundo == "permutado" else mundo
            r2 = _r2(O.anomalia(fonte, S2), O.preve(ac, S2, com_h))
            rx = _r2(O.anomalia(fonte, S3), O.preve(ac, S3, com_h))
            res[(mundo, com_h)].append((ac, r2, rx))
            print(f"    s{k}  R2={r2:6.3f}  extrap={rx:6.3f}   {O.escreve(ac)}")
    print(f"\n  ({time.time()-t0:.0f}s)\n")

    def primeiro(ac):
        return ac[0][0] if ac else "nada"

    def usa_h(ac):
        return any(t[0] == "lei" and "h" in t[2] for t in ac)

    print("=" * 76)
    lim = n - 1
    lim0 = {m: sum(1 for ac, _, _ in res[(m, True)] if not ac)
            for m in ("nulo", "permutado")}
    print(f"O0' {'.' if min(lim0.values()) >= lim else 'FALHOU .'} nulos limpos: "
          + ", ".join(f"{m} {v}/{n}" for m, v in lim0.items()))
    e1 = sum(1 for ac, r2, _ in res[("einstein", True)]
             if primeiro(ac) == "lei" and r2 >= 0.98 and usa_h(ac))
    print(f"O1' {'.' if e1 >= lim else 'FALHOU .'} einstein lei com h: {e1}/{n}")
    o2 = sum(1 for ac, r2, _ in res[("oblato", True)]
             if primeiro(ac) == "lei" and r2 >= 0.98 and not usa_h(ac))
    print(f"O2' {'.' if o2 >= lim else 'FALHOU .'} oblato lei sem h: {o2}/{n}")
    v3 = sum(1 for ac, _, _ in res[("vulcano", True)]
             if primeiro(ac) == "objeto"
             and np.hypot(ac[0][2] - O.P_VUL[0], ac[0][3] - O.P_VUL[1]) < 0.5)
    trai = sum(1 for m in ("einstein", "oblato") for ac, _, _ in res[(m, True)]
               if primeiro(ac) == "objeto")
    print(f"O3' {'.' if v3 >= lim and trai == 0 else 'FALHOU .'} vulcano objeto "
          f"{v3}/{n}; leis confundidas com objeto {trai}")
    o5 = sum(1 for _, _, rx in res[("einstein", False)] if rx >= 0.98)
    print(f"O5  {'.' if o5 >= lim else 'FALHOU .'} einstein SEM h extrapola "
          f"(lei, nao epiciclo): {o5}/{n}")


if __name__ == "__main__":
    main()

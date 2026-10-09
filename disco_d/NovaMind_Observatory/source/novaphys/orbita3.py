"""
novaphys.orbita3 — o nulo ainda passa: selecao contaminada ou codigo errado?
=============================================================================

ONDE ESTAMOS
--------------
`orbita2` (semente 20260914): O1'-O3' passaram 5/5; O5 passou 4/5 — sem h no
caderno, a lei EXATA  -0,03 v^2 r^-2 + 0,03 vr^2 r^-2  saiu em 3/5 e extrapola
com R2 0,998-1,000, contra -3,3 a 0,8 do epiciclo da rodada 1. Mas O0' FALHOU:
o mundo permutado ainda aceitou termos "espinho" em 2/5 (x^-2 t^, vx^-4 vr^4 v^).

DUAS CAUSAS POSSIVEIS, E ESTA RODADA SEPARA AS DUAS
-----------------------------------------------------
  1. SELECAO CONTAMINADA. Em `orbita2` os 15 candidatos sao escolhidos com TODOS
     os dados e so depois validados fora da amostra. A validacao julga uma
     escolha que ja viu a metade de teste. O remedio de livro e VALIDACAO
     ANINHADA: em cada dobra, o PROCEDIMENTO inteiro (escolher o termo e ajustar)
     roda so na metade de treino, e e julgado na outra. Aceita-se uma modificacao
     se o procedimento, assim avaliado, melhora fora da amostra; o termo final e o
     que o mesmo procedimento escolhe com todos os dados.
  2. CODIGO ERRADO PARA CAUDA PESADA. O residuo permutado tem cauda pesada (a
     anomalia de einstein cresce como r^-4). O codigo gaussiano cobra o QUADRADO
     do erro, e entao um termo que acerta um ou dois pontos enormes compra muitos
     bits. Um codigo de LAPLACE cobra o erro absoluto: e o codigo robusto padrao.

Dois bracos, os dois com validacao aninhada:
    ANINHADO           aninhado + codigo gaussiano      (testa so a causa 1)
    ANINHADO+LAPLACE   aninhado + codigo de Laplace     (causa 1 e 2)

PRE-REGISTRO, ANTES DE RODAR — semente nova 20260915, 10 por condicao
-----------------------------------------------------------------------
Por braco:
  O0  nulo e permutado sem modificacao em >= 9/10 cada
  O1  einstein: lei com h, R2 >= 0,98, em >= 9/10
  O2  oblato: lei sem h, R2 >= 0,98, em >= 9/10
  O3  vulcano: objeto a < 0,5 do lugar em >= 9/10; leis nunca viram objeto
  O5  einstein SEM h: extrapolacao R2 >= 0,98 em >= 8/10
  e, descritivo: quantas vezes a lei EXATA sem h (os dois termos, coeficientes
  opostos a 20% de 0,03, nada mais) foi escrita.

Leitura:
  ANINHADO passa O0             -> era contaminacao da selecao
  so ANINHADO+LAPLACE passa O0  -> era o codigo; a contaminacao nao bastava
  nenhum passa                  -> ha uma terceira causa, e eu nao a entendo

PREVISAO: ANINHADO sozinho NAO passa O0 (permutado 7-8/10): a contaminacao
existe mas os espinhos vivem da cauda pesada. ANINHADO+LAPLACE passa O0
(>= 9/10). O1-O3 passam nos dois. O5 passa nos dois; lei exata em 5-8 de 10.

CALIBRAGEM HOJE: previ O0 passando duas vezes seguidas e errei as duas.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from math import log2

import numpy as np

from novaphys import orbita as O
from novaphys import orbita2 as O2

SEMENTE = 20260915
BRACOS = ("aninhado", "aninhado+laplace")


def _cod(Rm, codigo):
    return (Rm ** 2).sum() if codigo == "gauss" else np.abs(Rm).sum()


def _bits(antes, depois, n, codigo):
    depois = max(depois, 1e-300)
    return (0.5 if codigo == "gauss" else 1.0) * n * log2(antes / depois)


def _resid(D, bases, tr, alvo):
    """Residuo em `alvo` do modelo conjunto ajustado em `tr`."""
    if not bases:
        return D[alvo]
    c = O2._ajusta(bases, D, tr)
    M = np.stack([b[alvo].ravel() for b in bases], 1)
    return (D[alvo].ravel() - M @ c).reshape(-1, 2)


def _escolhe(D, bases, idx, S, X, dirs, P, E):
    """O PROCEDIMENTO de escolha, usando so as linhas `idx`. Devolve a base
    escolhida (sobre todos os estados) e o ganho dentro de `idx`, ou None."""
    n = len(idx) * 2
    R = _resid(D, bases, idx, idx)
    sse = (R ** 2).sum()
    Q = O2._ortonormal([b[idx] for b in bases])
    top = O2._top_termos(R, Q, X[:, idx], {d: u[idx] for d, u in dirs.items()},
                         P, n)
    g_lei, i, d = top[0]
    m, p = O2._fit_obj(R, S[idx])
    red = sse - ((R - m * O._obj_base(S[idx], p)) ** 2).sum()
    g_obj = O.ganho(sse, red, n, O.bits_objeto(n))
    if max(g_lei, g_obj) <= 0:
        return None
    if g_lei >= g_obj:
        return g_lei, ("lei", d, i), O._F(E[i:i + 1], X)[0][:, None] * dirs[d]
    return g_obj, ("objeto", p), O._obj_base(S, p)


def descobre(S, D, com_h, rng, codigo):
    nomes, X = O.caderno(S, com_h)
    P = len(nomes)
    E, meta = O.candidatos(P)
    dirs = O.direcoes(S)
    neff = D.size
    perm = rng.permutation(len(S))
    A, B = perm[: len(S) // 2], perm[len(S) // 2:]
    tudo = np.arange(len(S))

    estr, bases, ganhos = [], [], []
    for _ in range(O.MAX_TERMOS):
        # o procedimento, julgado fora da amostra: escolhe em tr, mede em te
        antes = depois = 0.0
        for tr, te in ((A, B), (B, A)):
            r0 = _resid(D, bases, tr, te)
            antes += _cod(r0, codigo)
            esc = _escolhe(D, bases, tr, S, X, dirs, P, E)
            depois += _cod(_resid(D, bases + [esc[2]], tr, te) if esc else r0,
                           codigo)
        g_fora = _bits(antes, depois, neff, codigo)
        if g_fora <= 0:
            break
        esc = _escolhe(D, bases, tudo, S, X, dirs, P, E)
        if esc is None:
            break
        estr.append(esc[1])
        bases.append(esc[2])
        ganhos.append(g_fora)

    aceitos = []
    if estr:
        c = O2._ajusta(bases, D, tudo)
        for e, ci, g in zip(estr, c, ganhos):
            if e[0] == "lei":
                _k, vs, es = meta[e[2]]
                aceitos.append(("lei", e[1], tuple(nomes[j] for j in vs), es,
                                float(ci), g))
            else:
                aceitos.append(("objeto", float(ci), e[1][0], e[1][1], g))
    return aceitos


def exata_sem_h(ac):
    leis = [t for t in ac if t[0] == "lei"]
    if len(leis) != 2 or len(ac) != 2:
        return False
    alvo = {(("r", "v"), (-2, 2)): -O.A_EIN, (("r", "vr"), (-2, 2)): O.A_EIN}
    achou = {(t[2], tuple(t[3])): t[4] for t in leis if t[1] == "r^"}
    return (set(achou) == set(alvo)
            and all(abs(achou[k] - v) <= 0.2 * abs(v) for k, v in alvo.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=10)
    a = ap.parse_args()
    n = a.sementes
    S2 = O.estados(np.random.default_rng(SEMENTE + 5), 2000)
    S3 = O2.estados_ext(np.random.default_rng(SEMENTE + 6), 2000)

    print("=" * 76)
    print("ORBITA 3 — validacao aninhada; codigo gaussiano x Laplace")
    print("=" * 76)
    t0 = time.time()
    res = {}
    for braco in BRACOS:
        codigo = "laplace" if "laplace" in braco else "gauss"
        print(f"\n######## {braco} ########")
        for mundo, com_h in O.CONDICOES:
            rot = f"{mundo}{'' if com_h else ' (sem h)'}"
            print(f"\n  --- {rot} ---")
            iw = ("nulo", "permutado", "einstein", "oblato", "vulcano").index(mundo)
            lst = []
            for k in range(n):
                rng = np.random.default_rng(SEMENTE + 1000 * k + 97 * iw
                                            + (0 if com_h else 500))
                S = O.estados(rng, O.N_OBS)
                D = O.mede(mundo, S, rng)
                ac = descobre(S, D, com_h, rng, codigo)
                fonte = "nulo" if mundo == "permutado" else mundo
                r2 = O2._r2(O.anomalia(fonte, S2), O.preve(ac, S2, com_h))
                rx = O2._r2(O.anomalia(fonte, S3), O.preve(ac, S3, com_h))
                lst.append((ac, r2, rx))
                print(f"    s{k}  R2={r2:6.3f}  extrap={rx:6.3f}   {O.escreve(ac)}")
            res[(braco, mundo, com_h)] = lst
    print(f"\n  ({time.time()-t0:.0f}s)\n")

    def primeiro(ac):
        return ac[0][0] if ac else "nada"

    def usa_h(ac):
        return any(t[0] == "lei" and "h" in t[2] for t in ac)

    print("=" * 76)
    for braco in BRACOS:
        R = {(m, h): res[(braco, m, h)] for m, h in O.CONDICOES}
        print(f"\n  {braco}")
        l0 = {m: sum(1 for ac, _, _ in R[(m, True)] if not ac)
              for m in ("nulo", "permutado")}
        ok0 = min(l0.values()) >= n - 1
        print(f"    O0 {'.' if ok0 else 'FALHOU .'} nulos limpos: "
              + ", ".join(f"{m} {v}/{n}" for m, v in l0.items()))
        e1 = sum(1 for ac, r2, _ in R[("einstein", True)]
                 if primeiro(ac) == "lei" and r2 >= 0.98 and usa_h(ac))
        print(f"    O1 {'.' if e1 >= n - 1 else 'FALHOU .'} einstein lei com h: {e1}/{n}")
        o2 = sum(1 for ac, r2, _ in R[("oblato", True)]
                 if primeiro(ac) == "lei" and r2 >= 0.98 and not usa_h(ac))
        print(f"    O2 {'.' if o2 >= n - 1 else 'FALHOU .'} oblato lei sem h: {o2}/{n}")
        v3 = sum(1 for ac, _, _ in R[("vulcano", True)]
                 if primeiro(ac) == "objeto"
                 and np.hypot(ac[0][2] - O.P_VUL[0], ac[0][3] - O.P_VUL[1]) < 0.5)
        trai = sum(1 for m in ("einstein", "oblato") for ac, _, _ in R[(m, True)]
                   if primeiro(ac) == "objeto")
        print(f"    O3 {'.' if v3 >= n - 1 and trai == 0 else 'FALHOU .'} vulcano "
              f"objeto {v3}/{n}; leis confundidas com objeto {trai}")
        o5 = sum(1 for _, _, rx in R[("einstein", False)] if rx >= 0.98)
        ex = sum(1 for ac, _, _ in R[("einstein", False)] if exata_sem_h(ac))
        print(f"    O5 {'.' if o5 >= n - 2 else 'FALHOU .'} einstein SEM h extrapola: "
              f"{o5}/{n}   (lei EXATA escrita: {ex}/{n})")


if __name__ == "__main__":
    main()

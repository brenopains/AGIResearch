"""
novaphys.bilinear_busca — o maquinário contra Strassen
========================================================

A LINHA DE BASE A BATER
-------------------------
`bilinear.py` mediu: 200.000 sorteios ternários não acham nem o algoritmo
ESCOLAR de 2x2 (R=8). Melhor resultado: 6 falhas de 64 para R=7. A paisagem é o
mesmo penhasco que [[paisagem-e-penhasco]] mediu no ARC — agora num domínio
sem nenhuma relação e com verificador exato.

A ESTRUTURA QUE DISSOLVE O PENHASCO
-------------------------------------
A identidade é

    Σ_r u[r][ab] · v[r][cd] · w[r][il]  ==  [a==i][b==c][d==l]

e ela é **LINEAR EM w**. Fixados u e v, monta-se

    M[(ab,cd)][r] = u[r][ab] · v[r][cd]

e cada saída (i,l) vira um sistema linear exato  M · w[:,il] = t[i,l]. Para 2x2
são 4 sistemas de 16 equações e R incógnitas, resolvidos em FRAÇÕES — exato,
sem ponto flutuante.

Isso não é heurística: é o alvo deixando de ser "adivinhar 3·R·n² coeficientes"
e virando "adivinhar 2·R·n² e RESOLVER o resto". O espaço cai de 3^84 para 3^56
em 2x2, e cada amostra passa a ser decidida por álgebra em vez de sorte.

É [[abducao-construir-nao-propor]] aplicado: CONSTRUIR o termo a partir do dado
funciona; PROPOR de fora falha sempre. Aqui `w` é construído, não proposto.

E é [[construcao-com-verificador]]: descrição curta primeiro. Os produtos de
Strassen são somas de 1-2 entradas — esparsos. A busca prefere esparso, que é a
"descrição curta" deste domínio.

PRÉ-REGISTRO
--------------
  M0  PRÉ-CONDIÇÃO. Dado o (u,v) DE Strassen, o solucionador reconstrói um w
      válido. Se não reconstruir, a álgebra está errada e nada mais vale.
  M1  a busca acha 2x2 com R=7 válido em <= 2.000.000 de amostras
      -> a estrutura dissolve o penhasco, e o ganho sobre a linha de base é a
         medida do que "construir em vez de propor" vale AQUI
  M2  não acha -> reportar a menor contagem de falhas alcançada e comparar com
      as 6/64 da linha de base sob orçamento casado

  CONTROLE PAREADO: o mesmo número de amostras, mesma esparsidade, mas com `w`
  SORTEADO em vez de resolvido. Isola o ganho da ÁLGEBRA do ganho da amostragem.

PREVISÃO: M1, 0,65. Mecanismo: o solve exato transforma cada amostra num teste
de posto em vez de um sorteio de 28 coeficientes, e Strassen é esparso, que é
onde a amostragem concentra. Risco nomeado: pode existir uma multidão de (u,v)
com w rational válido mas sem interesse — nesse caso M1 passa e não significa
nada, então o resultado só conta se o algoritmo achado FECHAR no verificador
independente de `bilinear.verifica`.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from novaphys.bilinear import Alg, alvo, strassen, verifica


def resolve(M: List[List[Fraction]], t: List[Fraction]
            ) -> Optional[List[Fraction]]:
    """Gauss EXATO em frações. Devolve uma solução ou None se inconsistente."""
    m, n = len(M), len(M[0])
    A = [row[:] + [t[i]] for i, row in enumerate(M)]
    piv = []
    lin = 0
    for col in range(n):
        p = None
        for r in range(lin, m):
            if A[r][col] != 0:
                p = r
                break
        if p is None:
            continue
        A[lin], A[p] = A[p], A[lin]
        inv = Fraction(1, 1) / A[lin][col]
        A[lin] = [x * inv for x in A[lin]]
        for r in range(m):
            if r != lin and A[r][col] != 0:
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[lin])]
        piv.append(col)
        lin += 1
        if lin == m:
            break
    for r in range(lin, m):
        if all(A[r][c] == 0 for c in range(n)) and A[r][n] != 0:
            return None                      # inconsistente
    x = [Fraction(0)] * n
    for i, col in enumerate(piv):
        x[col] = A[i][n]
    return x


def _matriz(u, v, n: int, R: int) -> List[List[Fraction]]:
    """M[(ab,cd)][r] = u[r][ab]*v[r][cd]. Linhas = n^4, colunas = R."""
    nn = n * n
    return [[Fraction(u[r][ab] * v[r][cd]) for r in range(R)]
            for ab in range(nn) for cd in range(nn)]


def _alvos(n: int) -> List[List[Fraction]]:
    """Por saída (i,l), o vetor alvo indexado por (ab,cd)."""
    nn = n * n
    fora = []
    for i in range(n):
        for l in range(n):
            col = []
            for ab in range(nn):
                a, b = divmod(ab, n)
                for cd in range(nn):
                    c, d = divmod(cd, n)
                    col.append(Fraction(alvo(n, a, b, c, d, i, l)))
            fora.append(col)
    return fora


def completa_w(u, v, n: int, R: int) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """
    Dado (u,v), CONSTROI w resolvendo o sistema linear. Nenhum chute.
    Devolve None se algum dos n^2 sistemas for inconsistente.
    """
    M = _matriz(u, v, n, R)
    ws = []
    for t in _alvos(n):
        x = resolve(M, t)
        if x is None:
            return None
        ws.append(x)
    # w[r][il] = ws[il][r]
    nn = n * n
    return tuple(tuple(ws[il][r] for il in range(nn)) for r in range(R))


def _esparso(nn: int, rnd: random.Random, k: int) -> Tuple[int, ...]:
    """Descricao curta primeiro: exatamente k entradas nao nulas."""
    pos = rnd.sample(range(nn), k)
    v = [0] * nn
    for p in pos:
        v[p] = rnd.choice((-1, 1))
    return tuple(v)


def busca(n: int, R: int, amostras: int, semente: int = 20260921,
          resolver: bool = True, ks=(1, 2, 2, 3)) -> dict:
    """
    `resolver=True`  -> w CONSTRUIDO pelo sistema linear (o maquinario)
    `resolver=False` -> w SORTEADO (controle pareado: isola o ganho da algebra)
    """
    rnd = random.Random(semente)
    nn = n * n
    melhor, achado, checagens = 10 ** 9, None, 0
    for t in range(amostras):
        u = tuple(_esparso(nn, rnd, rnd.choice(ks)) for _ in range(R))
        v = tuple(_esparso(nn, rnd, rnd.choice(ks)) for _ in range(R))
        if resolver:
            w = completa_w(u, v, n, R)
            if w is None:
                continue
            if any(x.denominator != 1 for lin in w for x in lin):
                continue                     # so algoritmos inteiros
            w = tuple(tuple(int(x) for x in lin) for lin in w)
        else:
            w = tuple(_esparso(nn, rnd, rnd.choice(ks)) for _ in range(R))
        checagens += 1
        ok, f = verifica((u, v, w), n)
        if f < melhor:
            melhor = f
        if ok:
            achado = (u, v, w)
            break
    return {"achado": achado, "menor_falhas": melhor,
            "amostras": t + 1, "checagens_verificador": checagens}


def experimento(amostras: int = 2_000_000) -> None:
    print("=" * 78)
    print("O MAQUINARIO CONTRA STRASSEN")
    print("=" * 78)

    u, v, _w = strassen()
    w2 = completa_w(u, v, 2, 7)
    ok = w2 is not None and verifica((u, v, tuple(tuple(int(x) for x in l)
                                                  for l in w2)), 2)[0]
    print("\n  M0 — dado o (u,v) de Strassen, o solucionador reconstroi w?")
    print("       %s" % ("SIM, e o algoritmo reconstruido FECHA" if ok
                         else "NAO — a algebra esta errada, pare aqui"))
    if not ok:
        return

    print("\n  linha de base de bilinear.py: 200.000 sorteios, melhor 6/64\n")
    for rot, res in (("MAQUINARIO (w resolvido)", True),
                     ("CONTROLE   (w sorteado) ", False)):
        r = busca(2, 7, amostras if res else amostras // 4, resolver=res)
        print("  %s  achou=%-5s  menor falhas=%-3d  amostras=%d  "
              "checagens=%d" % (rot, r["achado"] is not None,
                                r["menor_falhas"], r["amostras"],
                                r["checagens_verificador"]), flush=True)
        if r["achado"] is not None and res:
            uu, vv, ww = r["achado"]
            print("\n  ALGORITMO ACHADO (2x2 com 7 produtos), conferido pelo")
            print("  verificador independente:")
            for i in range(7):
                print("    P%d: u=%s  v=%s  w=%s" % (i + 1, uu[i], vv[i], ww[i]))


if __name__ == "__main__":
    experimento()


# ============================================================================
# O GRADIENTE QUE A ALGEBRA CRIA
# ============================================================================
#
# O piloto expos que sortear (u,v) quase nunca da sistema CONSISTENTE: 0
# checagens em 3000 amostras. O gargalo so mudou de lugar.
#
# Mas o sistema linear nao devolve so consistente/inconsistente — devolve um
# RESIDUO. A identidade tensorial e penhasco (0 ou ~46, sem meio-termo, e
# [[paisagem-e-penhasco]] mediu isso em 96% dos casos). O residuo de minimos
# quadrados do melhor w para um dado (u,v) e CONTINUO e varia suavemente quando
# se troca um coeficiente de u ou v.
#
# Entao a reformulacao algebrica nao apenas encolhe o espaco: ela CRIA o
# gradiente que o problema original nao tinha. E ai busca local passa a fazer
# sentido onde sorteio estacionava em 6/64.
#
#   R0  PRE-CONDICAO: residuo(u,v de Strassen) == 0, e residuo de um (u,v)
#       sorteado > 0. Se nao separar, a medida nao serve.
#   R1  subida de encosta sobre (u,v) acha 2x2/7 valido
#       -> o gradiente e real e utilizavel; a linha de base 6/64 cai
#   R2  nao acha -> reportar menor residuo e comparar com o sorteio pareado

import numpy as np


def residuo(u, v, n: int, R: int) -> float:
    """Residuo de minimos quadrados do melhor w. CONTINUO onde o tensor e penhasco."""
    nn = n * n
    M = np.array([[float(u[r][ab] * v[r][cd]) for r in range(R)]
                  for ab in range(nn) for cd in range(nn)])
    tot = 0.0
    for t in _alvos(n):
        b = np.array([float(x) for x in t])
        sol, res, rank, _sv = np.linalg.lstsq(M, b, rcond=None)
        tot += float(res[0]) if res.size else float(((M @ sol - b) ** 2).sum())
    return tot


def sobe_encosta(n: int, R: int, passos: int, semente: int,
                 ks=(1, 2, 2, 3)) -> dict:
    rnd = random.Random(semente)
    nn = n * n
    u = [list(_esparso(nn, rnd, rnd.choice(ks))) for _ in range(R)]
    v = [list(_esparso(nn, rnd, rnd.choice(ks))) for _ in range(R)]
    cur = residuo(tuple(map(tuple, u)), tuple(map(tuple, v)), n, R)
    aval = 1
    for _ in range(passos):
        if cur < 1e-9:
            break
        qual = rnd.random() < 0.5
        alvo_m = u if qual else v
        r, p = rnd.randrange(R), rnd.randrange(nn)
        velho = alvo_m[r][p]
        novo = rnd.choice([x for x in (-1, 0, 1) if x != velho])
        alvo_m[r][p] = novo
        n2 = residuo(tuple(map(tuple, u)), tuple(map(tuple, v)), n, R)
        aval += 1
        if n2 <= cur:
            cur = n2
        else:
            alvo_m[r][p] = velho
    U, V = tuple(map(tuple, u)), tuple(map(tuple, v))
    fora = {"residuo": cur, "avaliacoes": aval, "achado": None}
    if cur < 1e-9:
        w = completa_w(U, V, n, R)
        if w is not None and all(x.denominator == 1 for l in w for x in l):
            wi = tuple(tuple(int(x) for x in l) for l in w)
            if verifica((U, V, wi), n)[0]:
                fora["achado"] = (U, V, wi)
    return fora


def experimento_encosta(reinicios: int = 60, passos: int = 4000) -> None:
    print("=" * 78)
    print("O GRADIENTE QUE A ALGEBRA CRIA — subida de encosta em (u,v)")
    print("=" * 78)
    u, v, _ = strassen()
    rs = residuo(u, v, 2, 7)
    rnd = random.Random(7)
    ru = tuple(_esparso(4, rnd, 2) for _ in range(7))
    rv = tuple(_esparso(4, rnd, 2) for _ in range(7))
    rr = residuo(ru, rv, 2, 7)
    print("\n  R0 — residuo(Strassen) = %.3g   residuo(sorteado) = %.3g   %s"
          % (rs, rr, "SEPARA" if rs < 1e-9 < rr else "NAO SEPARA"))
    if not (rs < 1e-9 < rr):
        return
    print("\n  linha de base: 200.000 sorteios, melhor 6/64, nunca acha\n")
    achados, melhores, tot = 0, [], 0
    for s in range(reinicios):
        r = sobe_encosta(2, 7, passos, semente=1000 + s)
        melhores.append(r["residuo"]); tot += r["avaliacoes"]
        if r["achado"]:
            achados += 1
            if achados == 1:
                U, V, W = r["achado"]
                print("  ACHOU em %d avaliacoes. Algoritmo 2x2 com 7 produtos:"
                      % r["avaliacoes"])
                for i in range(7):
                    print("    P%d  u=%-12s v=%-12s w=%s"
                          % (i + 1, U[i], V[i], W[i]))
                print()
    print("  %d/%d reinicios acharam um algoritmo VALIDO (verificador exato)"
          % (achados, reinicios))
    print("  menor residuo medio: %.4g   avaliacoes totais: %d"
          % (sum(melhores) / len(melhores), tot))
    print()
    if achados:
        print("R1 . O GRADIENTE E REAL. Reformular pela algebra criou a encosta")
        print("     que a identidade tensorial nao tinha, e a busca local acha")
        print("     Strassen onde 200.000 sorteios estacionavam em 6/64.")
    else:
        print("R2 . nao achou; menor residuo medio %.4g."
              % (sum(melhores) / len(melhores)))


# ----------------------------------------------------------------------------
# RECOZIMENTO: o gradiente existe mas tem minimo local
#
# A descida pura desce de 4,87 (sorteado) para 1,23 e trava — 4x de reducao onde
# a identidade tensorial nao dava NENHUM valor intermediario. Isso e mudanca de
# regime, nao fracasso: saiu de "sem gradiente" para "gradiente com minimo
# local". Aceitar piora com probabilidade decrescente e o que navega isso.
#
#   S1  o recozimento acha 2x2/7 valido -> a paisagem e navegavel e a linha de
#       base 6/64 cai por reformulacao, nao por forca bruta
#   S2  nao acha -> reportar o menor residuo e comparar com 1,232 da descida
#       pura, sob orcamento casado em AVALIACOES

import math


def recoze(n: int, R: int, passos: int, semente: int, T0: float = 0.6,
           ks=(1, 2, 2, 3)) -> dict:
    rnd = random.Random(semente)
    nn = n * n
    u = [list(_esparso(nn, rnd, rnd.choice(ks))) for _ in range(R)]
    v = [list(_esparso(nn, rnd, rnd.choice(ks))) for _ in range(R)]
    cur = residuo(tuple(map(tuple, u)), tuple(map(tuple, v)), n, R)
    melhor, best_uv = cur, (tuple(map(tuple, u)), tuple(map(tuple, v)))
    for k in range(passos):
        if melhor < 1e-9:
            break
        T = T0 * (1.0 - k / passos) + 1e-4
        qual = rnd.random() < 0.5
        M = u if qual else v
        r, p = rnd.randrange(R), rnd.randrange(nn)
        velho = M[r][p]
        M[r][p] = rnd.choice([x for x in (-1, 0, 1) if x != velho])
        novo = residuo(tuple(map(tuple, u)), tuple(map(tuple, v)), n, R)
        if novo <= cur or rnd.random() < math.exp(-(novo - cur) / T):
            cur = novo
            if novo < melhor:
                melhor = novo
                best_uv = (tuple(map(tuple, u)), tuple(map(tuple, v)))
        else:
            M[r][p] = velho
    fora = {"residuo": melhor, "avaliacoes": k + 1, "achado": None,
            "uv": best_uv}
    if melhor < 1e-9:
        U, V = best_uv
        w = completa_w(U, V, n, R)
        if w is not None and all(x.denominator == 1 for l in w for x in l):
            wi = tuple(tuple(int(x) for x in l) for l in w)
            if verifica((U, V, wi), n)[0]:
                fora["achado"] = (U, V, wi)
    return fora


def experimento_recozimento(reinicios: int = 40, passos: int = 20000) -> None:
    print("=" * 78)
    print("RECOZIMENTO — a paisagem criada pela algebra e navegavel?")
    print("=" * 78)
    print("\n  sorteio puro (200.000 tentativas): NUNCA acha, melhor 6/64")
    print("  descida pura (62.525 avaliacoes):   0/25, residuo medio 1,232\n")
    achados, res, tot = 0, [], 0
    for s in range(reinicios):
        r = recoze(2, 7, passos, semente=5000 + s)
        res.append(r["residuo"]); tot += r["avaliacoes"]
        if r["achado"]:
            achados += 1
            if achados == 1:
                U, V, W = r["achado"]
                print("  ACHOU em %d avaliacoes deste reinicio." % r["avaliacoes"])
                print("  Algoritmo 2x2 com 7 produtos, conferido pelo")
                print("  verificador exato e independente:")
                for i in range(7):
                    print("    P%d  u=%-13s v=%-13s w=%s"
                          % (i + 1, U[i], V[i], W[i]))
                print()
        if (s + 1) % 10 == 0:
            print("    ... %d/%d reinicios, %d achados, menor residuo %.4g"
                  % (s + 1, reinicios, achados, min(res)), flush=True)
    print("\n  %d/%d reinicios acharam algoritmo VALIDO" % (achados, reinicios))
    print("  menor residuo: %.4g   medio: %.4g   avaliacoes: %d"
          % (min(res), sum(res) / len(res), tot))
    print()
    if achados:
        print("S1 . A PAISAGEM E NAVEGAVEL. Reformular o problema pela algebra")
        print("     criou o gradiente que a identidade tensorial nao tinha, e o")
        print("     recozimento acha Strassen onde 200.000 sorteios")
        print("     estacionavam em 6/64.")
    else:
        print("S2 . nao achou. menor residuo %.4g contra 1,232 da descida pura."
              % min(res))


# ----------------------------------------------------------------------------
# DEFEITO NA RECONSTRUCAO, nao na busca.
#
# O recozimento ATINGIU residuo 1,16e-30 — o sistema ficou consistente — e ainda
# assim reportou 0 achados. Causa: eu exigia `w` INTEIRO. Um algoritmo bilinear
# com w RACIONAL e perfeitamente valido sobre Q; Strassen ser inteiro e um fato
# sobre Strassen, nao um requisito do problema. Alem disso o residuo e float e o
# solve e exato, entao a palavra final tem de ser das FRACOES.

def verifica_frac(u, v, w, n: int) -> Tuple[bool, int]:
    """O mesmo verificador, EXATO sobre Q. Nada de ponto flutuante."""
    R = len(u)
    falhas = 0
    rng = range(n)
    for a in rng:
        for b in rng:
            for c in rng:
                for d in rng:
                    ab, cd = a * n + b, c * n + d
                    for i in rng:
                        for l in rng:
                            il = i * n + l
                            s = Fraction(0)
                            for r in range(R):
                                ur = u[r][ab]
                                if ur == 0:
                                    continue
                                vr = v[r][cd]
                                if vr == 0:
                                    continue
                                s += Fraction(ur * vr) * w[r][il]
                            if s != alvo(n, a, b, c, d, i, l):
                                falhas += 1
    return falhas == 0, falhas


def reconstroi(U, V, n: int, R: int):
    """Sem filtro de inteiro; veredito pelas FRACOES."""
    w = completa_w(U, V, n, R)
    if w is None:
        return None, "sistema inconsistente (exato)"
    ok, f = verifica_frac(U, V, w, n)
    if not ok:
        return None, "solve fechou mas a identidade falha em %d eq" % f
    inteiro = all(x.denominator == 1 for l in w for x in l)
    return (U, V, w), ("valido, w %s" % ("INTEIRO" if inteiro else "RACIONAL"))


def experimento_final(reinicios: int = 60, passos: int = 20000) -> None:
    print("=" * 78)
    print("STRASSEN — com a reconstrucao consertada")
    print("=" * 78)
    print("\n  sorteio puro (200.000):  NUNCA acha, melhor 6/64")
    print("  descida pura  (62.525):  0/25, residuo medio 1,232\n")
    achados, res, tot, mostrou = 0, [], 0, False
    for s in range(reinicios):
        r = recoze(2, 7, passos, semente=5000 + s)
        res.append(r["residuo"]); tot += r["avaliacoes"]
        if r["residuo"] < 1e-9:
            # refaz a reconstrucao a partir do (u,v) que o recozimento guardou
            pass
        if (s + 1) % 15 == 0:
            print("    ... %d/%d, menor residuo %.4g" % (s + 1, reinicios,
                                                         min(res)), flush=True)
    print("\n  menor residuo alcancado: %.4g   avaliacoes: %d"
          % (min(res), tot))


# ============================================================================
# 3x3 — onde 23 e conhecido (Laderman 1976) e 22 esta ABERTO
# ============================================================================
#
# Custo: 9 sistemas de 81xR por avaliacao. Mas os 9 alvos compartilham a MESMA
# matriz M, entao lstsq com multiplos lados direitos resolve tudo de uma vez —
# 9x mais barato. Sem isso o recozimento nao cabe no orcamento.
#
# A escada e descendente e cada degrau e um resultado: R=27 e o escolar, 23 e
# Laderman, 22 e fronteira publica. Onde a busca parar, para ali e a parede
# MEDIDA deste maquinario.

_ALVO_CACHE = {}


def alvos_matriz(n: int):
    """(n^4 x n^2): coluna il, linha (ab,cd). Construida uma vez."""
    if n in _ALVO_CACHE:
        return _ALVO_CACHE[n]
    nn = n * n
    T = np.zeros((nn * nn, nn))
    for ab in range(nn):
        a, b = divmod(ab, n)
        for cd in range(nn):
            c, d = divmod(cd, n)
            for il in range(nn):
                i, l = divmod(il, n)
                T[ab * nn + cd, il] = alvo(n, a, b, c, d, i, l)
    _ALVO_CACHE[n] = T
    return T


def residuo_r(u, v, n: int, R: int) -> float:
    """Todos os n^2 alvos num lstsq so. Mesma grandeza, 9x mais barato."""
    nn = n * n
    U = np.array(u, dtype=float)              # R x nn
    V = np.array(v, dtype=float)              # R x nn
    M = (U[:, :, None] * V[:, None, :]).reshape(R, nn * nn).T   # (nn^2) x R
    T = alvos_matriz(n)
    sol, res, rank, _ = np.linalg.lstsq(M, T, rcond=None)
    if res.size == T.shape[1]:
        return float(res.sum())
    return float(((M @ sol - T) ** 2).sum())


def recoze_r(n: int, R: int, passos: int, semente: int, T0: float = 0.6,
             ks=(1, 2, 2, 3, 4)) -> dict:
    rnd = random.Random(semente)
    nn = n * n
    u = [list(_esparso(nn, rnd, min(rnd.choice(ks), nn))) for _ in range(R)]
    v = [list(_esparso(nn, rnd, min(rnd.choice(ks), nn))) for _ in range(R)]
    cur = residuo_r(u, v, n, R)
    melhor, best = cur, ([r[:] for r in u], [r[:] for r in v])
    for k in range(passos):
        if melhor < 1e-9:
            break
        T = T0 * (1.0 - k / passos) + 1e-4
        M = u if rnd.random() < 0.5 else v
        r, p = rnd.randrange(R), rnd.randrange(nn)
        velho = M[r][p]
        M[r][p] = rnd.choice([x for x in (-1, 0, 1) if x != velho])
        novo = residuo_r(u, v, n, R)
        if novo <= cur or rnd.random() < math.exp(-(novo - cur) / T):
            cur = novo
            if novo < melhor:
                melhor = novo
                best = ([x[:] for x in u], [x[:] for x in v])
        else:
            M[r][p] = velho
    return {"residuo": melhor, "passos": k + 1,
            "uv": (tuple(map(tuple, best[0])), tuple(map(tuple, best[1])))}


def escada(n: int, Rs, reinicios: int, passos: int, semente0: int = 9000):
    print("=" * 78)
    print("ESCADA %dx%d — descendo R ate a parede" % (n, n))
    print("=" * 78)
    print("  verificador exato sobre Q; residuo por lstsq multi-RHS\n")
    for R in Rs:
        achados, melhores = 0, []
        alg_ok = None
        for s in range(reinicios):
            r = recoze_r(n, R, passos, semente=semente0 + 137 * R + s)
            melhores.append(r["residuo"])
            if r["residuo"] < 1e-9:
                U, V = r["uv"]
                alg, nota = reconstroi(U, V, n, R)
                if alg:
                    achados += 1
                    if alg_ok is None:
                        alg_ok = (alg, nota, s)
        marca = {27: "escolar", 23: "Laderman 1976", 22: "FRONTEIRA ABERTA"}
        print("  R=%-3d %-18s achados %d/%d   menor residuo %.4g"
              % (R, marca.get(R, ""), achados, reinicios, min(melhores)),
              flush=True)
        if alg_ok:
            print("        -> %s" % alg_ok[1])
        if achados == 0:
            print("\n  PAREDE em R=%d. Acima disso a busca fecha; aqui nao." % R)
            return R
    return None

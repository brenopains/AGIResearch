"""
novaphys.invariante_arc — a operacao que achou h nas orbitas, levada ao ARC
=============================================================================

A PERGUNTA DO USUARIO
-----------------------
"Que prova temos de que estamos generalizando com os substratos certos?" Nao da
para aplicar o sabor de uma maca a fisica quantica. O que transfere entre dominios,
em humanos, sao RELACOES e OPERACOES, nao atributos (Gentner; Gick & Holyoak). E
"procure o que se conserva" e uma das operacoes que atravessa toda a fisica.

O TESTE
---------
Na fisica, a operacao foi: entre muitas combinacoes de grandezas, ache a que fica
CONSTANTE ao longo de cada trajetoria e VARIA entre trajetorias, julgada em bits.
Aqui, a "trajetoria" e um par de treino do ARC: entrada -> saida. Uma invariante e
uma combinacao de grandezas da grade que a TRANSFORMACAO conserva:
    Q(saida) = Q(entrada)   em todos os pares de treino.

O que e o mesmo e o que nao e, dito antes: o PRINCIPIO e a REGUA (bits) sao os
mesmos. A implementacao e a versao DISCRETA — grade e inteira, entao "conservar"
e igualdade exata, e o autovetor vira enumeracao de pesos inteiros pequenos. O
alfabeto de grandezas da grade e meu, como o da fisica:
    contagem de cada cor (10), altura, largura, area, numero de cores, numero de
    objetos (componentes 4-conexos de mesma cor, fora do fundo), celulas
    ocupadas, altura e largura da caixa envolvente do que nao e fundo   (18)
Combinacoes: 1 grandeza (peso 1); 2 grandezas (pesos {1,2} x {+-1,+-2}); 3
grandezas (pesos +-1). Ganho em bits:
    SOMA_pares log2(area_saida + 1)  -  log2(numero de candidatos)
(saber Q da saida poupa codificar um numero entre 0 e a area; escolher entre K
candidatas custa log2 K).

A PROVA: O PAR DE TESTE, CONTRA UM NULO
-----------------------------------------
A invariante foi achada so no treino. Se ela for estrutura real, vale no par de
TESTE (que o agente nunca viu). NULO: as saidas trocadas por saidas de OUTRAS
tarefas (mesmo sorteio para treino e teste) — qualquer "invariante" ali e
coincidencia, e a taxa com que ela vale no teste e a linha de base do acaso.

PRE-REGISTRO, ANTES DE RODAR — 400 tarefas de avaliacao, semente do nulo 20260920
------------------------------------------------------------------------------------
  A0  PRECISAO: entre as tarefas com invariante aceita, a MAIS SIMPLES vale no par
      de teste em >= 70%
  A1  CONTRA O NULO: precisao real >= precisao nula + 30 pontos
  A2  CONTEUDO, NAO SO TAMANHO: repetindo A0-A1 SEM altura/largura/area (que
      qualquer tarefa de "mesmo tamanho" conserva), a precisao real ainda fica
      >= nula + 20 pontos

PREVISAO: A0 passa (~85%); A1 passa (nulo ~20-30%); A2 passa por menos margem.
Se A1 passar, e a primeira evidencia de que uma OPERACAO aprendida na fisica acha
estrutura verdadeira noutro substrato. Nao e ainda que ela AJUDE a resolver o ARC
— isso e a fase seguinte (a invariante como CRITICA das respostas do nucleo).
"""

from __future__ import annotations

import argparse
import itertools
import random
from collections import Counter
from math import log2

from novaphys.arc import carregar, dims, fundo
from novaphys.eter import PASTA_AVAL

SEMENTE_NULO = 20260920
NOMES = [f"cor{c}" for c in range(10)] + ["alt", "larg", "area", "ncores", "nobj",
                                           "ocup", "bb_alt", "bb_larg"]
TAMANHO = {"alt", "larg", "area"}


def _objetos(g, bg):
    h, w = dims(g)
    visto = [[False] * w for _ in range(h)]
    n = 0
    for i in range(h):
        for j in range(w):
            if visto[i][j] or g[i][j] == bg:
                continue
            n += 1
            c = g[i][j]
            pilha = [(i, j)]
            visto[i][j] = True
            while pilha:
                y, x = pilha.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not visto[yy][xx] \
                            and g[yy][xx] == c:
                        visto[yy][xx] = True
                        pilha.append((yy, xx))
    return n


def grandezas(g):
    h, w = dims(g)
    cnt = Counter(x for lin in g for x in lin)
    bg = fundo(g)
    oc = [(i, j) for i in range(h) for j in range(w) if g[i][j] != bg]
    if oc:
        bb_a = max(i for i, _ in oc) - min(i for i, _ in oc) + 1
        bb_l = max(j for _, j in oc) - min(j for _, j in oc) + 1
    else:
        bb_a = bb_l = 0
    return ([cnt.get(c, 0) for c in range(10)]
            + [h, w, h * w, len(cnt), _objetos(g, bg), len(oc), bb_a, bb_l])


def candidatos(usar):
    out = [((i,), (1,)) for i in usar]
    for i, j in itertools.combinations(usar, 2):
        for a in (1, 2):
            for b in (1, -1, 2, -2):
                if a == 2 and abs(b) == 2:
                    continue
                out.append(((i, j), (a, b)))
    for i, j, k in itertools.combinations(usar, 3):
        for b in (1, -1):
            for c in (1, -1):
                out.append(((i, j, k), (1, b, c)))
    return out


def Q(f, cand):
    return sum(p * f[i] for i, p in zip(*cand))


def acha(treino, cands):
    F = [(grandezas(a), grandezas(b), dims(b)) for a, b in treino]
    custo = log2(len(cands))
    aceitas = []
    for cand in cands:
        if all(Q(fa, cand) == Q(fb, cand) for fa, fb, _ in F):
            # trivial: constante em TODAS as grades (nao distingue nada)
            if len({Q(fa, cand) for fa, _, _ in F}) == 1 and len(F) > 1 \
                    and all(Q(fa, cand) == 0 for fa, _, _ in F):
                continue
            ganho = sum(log2(h * w + 1) for _, _, (h, w) in F) - custo
            if ganho > 0:
                aceitas.append((len(cand[0]), -ganho, cand))
    aceitas.sort()
    return aceitas


def texto(cand):
    return " ".join(f"{p:+d}*{NOMES[i]}" for i, p in zip(*cand))


def mede(tarefas, usar, rotulo):
    cands = candidatos(usar)
    com = vale = 0
    exemplos = []
    for nome, tr, te in tarefas:
        ac = acha(tr, cands)
        if not ac:
            continue
        com += 1
        cand = ac[0][2]
        ok = all(Q(grandezas(a), cand) == Q(grandezas(b), cand) for a, b in te)
        vale += ok
        if len(exemplos) < 6 and ok:
            exemplos.append(f"{nome}: {texto(cand)}")
    prec = vale / max(com, 1)
    print(f"  {rotulo:34s} com invariante {com:3d}/400   vale no teste "
          f"{vale:3d}/{com:<3d} = {100*prec:5.1f}%")
    return prec, exemplos


def nulo(tarefas):
    rng = random.Random(SEMENTE_NULO)
    saidas = [b for _n, tr, te in tarefas for _a, b in tr + te]
    fora = []
    for nome, tr, te in tarefas:
        tr2 = [(a, rng.choice(saidas)) for a, _b in tr]
        te2 = [(a, rng.choice(saidas)) for a, _b in te]
        fora.append((nome, tr2, te2))
    return fora


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tarefas = carregar(PASTA_AVAL)
    nul = nulo(tarefas)
    todas = list(range(len(NOMES)))
    sem_tam = [i for i, n in enumerate(NOMES) if n not in TAMANHO]
    print("=" * 76)
    print("INVARIANTE ARC — a operacao das orbitas, no par entrada -> saida")
    print("=" * 76)
    pr, ex = mede(tarefas, todas, "REAL, alfabeto inteiro")
    pn, _ = mede(nul, todas, "NULO, alfabeto inteiro")
    pr2, ex2 = mede(tarefas, sem_tam, "REAL, sem alt/larg/area")
    pn2, _ = mede(nul, sem_tam, "NULO, sem alt/larg/area")
    print("\n  exemplos (real, sem tamanho) que valeram no teste:")
    for e in ex2:
        print(f"    {e}")
    print("")
    print("=" * 76)
    print(f"A0 {'.' if pr >= 0.70 else 'FALHOU .'} precisao real {100*pr:.1f}% (limiar 70%)")
    print(f"A1 {'.' if pr >= pn + 0.30 else 'FALHOU .'} real - nulo = "
          f"{100*(pr-pn):+.1f} pontos (limiar +30)")
    print(f"A2 {'.' if pr2 >= pn2 + 0.20 else 'FALHOU .'} sem tamanho: real - nulo = "
          f"{100*(pr2-pn2):+.1f} pontos (limiar +20)")


if __name__ == "__main__":
    main()

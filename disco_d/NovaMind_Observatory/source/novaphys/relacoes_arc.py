"""
novaphys.relacoes_arc — a entrada como manual: o sistema acha os PARAMETROS da saida
=====================================================================================

DE ONDE VEM
-------------
Duas barreiras medidas no ARC, que sao uma so:
  - 53% das tarefas nao resolvidas (199/373) pedem FORMAS NOVAS — criar, nao
    transformar (`inventario.py`);
  - o compositor, que trata a entrada como fonte de parametros, acerta 100% das
    4 tarefas que suas familias escritas a mao cobrem, e nada fora delas.
Cada familia de desenho escrita por mim e o alfabeto entregue de novo.

A operacao que achou h nas orbitas ja atravessou para o ARC (`invariante_arc`:
invariantes entrada->saida valem no teste 97% x 16% no nulo). Aqui ela e
generalizada de IGUALDADE para RELACAO: uma grandeza da SAIDA como funcao de
grandezas da ENTRADA. "Altura da saida = numero de objetos da entrada", "celulas
da cor 3 na saida = 2 x celulas da cor 1 na entrada". Sao exatamente os
parametros que o compositor le — so que achados pelo sistema, nao escritos por
mim.

ESTA E A ETAPA 1: medir quanto da ESPECIFICACAO da saida as relacoes achadas no
treino acertam no teste. E condicao necessaria para construir. A etapa 2
(construir uma saida que satisfaca as relacoes, com o treino como verificador) vem
depois, se esta mostrar que ha o que construir.

O MECANISMO
-------------
Grandezas de uma grade: as 18 do `invariante_arc` (contagem das 10 cores, altura,
largura, area, n de cores, n de objetos, ocupadas, caixa). Para cada grandeza da
saida, relacoes candidatas com as da entrada (familias minhas, ditas aqui):
    constante            Qs = c
    afim                 Qs = a Qe + b,  a em {1, 2, 3, 4, 1/2, 1/3}, |b| <= 30
    soma / diferenca     Qs = Qe_i +- Qe_k
    produto              Qs = Qe_i x Qe_k
Aceita-se a relacao que vale EXATAMENTE em todos os pares de treino e mais poupa
bits: SOMA_pares log2(901) (codificar a grandeza sem relacao) menos o custo da
estrutura. Nulo: saidas trocadas por saidas de outras tarefas.

PRE-REGISTRO, ANTES DE RODAR — 400 tarefas de avaliacao, nulo com semente 20261014
------------------------------------------------------------------------------------
Nas tarefas com assinatura formas_novas (critério do `inventario.assinaturas`):
  R1  as dimensoes da saida de teste (altura E largura) saem certas em >= 60%, e o
      nulo fica em <= 20%
  R2  a especificacao de CONTAGEM completa (dimensoes + as 10 contagens de cor)
      sai certa em >= 25%, e o nulo em <= 5%
PREVISAO: R1 passa (~70% — dimensao costuma ser igual a da entrada ou constante).
R2 FALHA (~15%): contar as cores da saida exige saber o que se desenha, e
relacoes de uma ou duas grandezas nao devem bastar na maioria.

RESULTADO:
                  n    dimensoes certas   contagem completa   nulo
  formas_novas   213       79,3%               4,7%            0% / 0%
  todas as 400   400       75,8%               7,8%            0% / 0%
R1 PASSOU (79,3%, nulo 0%). R2 FALHOU (4,7%) — como previsto, e pior (previ 15%).
O manual global da a TELA (4 em 5 tarefas de criacao) mas quase nunca o CONTEUDO:
quanto de cada cor vai na saida nao e funcao de uma ou duas grandezas globais da
entrada. O conteudo depende de ESTRUTURA — quais objetos, onde.

DIAGNOSTICO POR GRANDEZA (formas_novas, certa no teste): area 89%, ncores 83%,
alt 82%, larg 80% — mas celulas OCUPADAS 23%, n de OBJETOS 30%, fundo (cor 0) 29%.
Das 10 contagens de cor, a moda e 8 certas (61 tarefas): tipicamente erra a cor
DESENHADA e o fundo (que e o complemento dela). Parte dos acertos de cor e trivial
(cores ausentes; a familia "produto" codifica zero barato). O que falta e
exatamente o CONTEUDO: quantos objetos, quantas celulas desenhadas.
Etapa 2: `estrutura_arc.py` — relacoes por OBJETO.
"""

from __future__ import annotations

import argparse
import itertools
import random
from fractions import Fraction
from math import comb, log2

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys.inventario import assinaturas
from novaphys import invariante_arc as IA

SEMENTE_NULO = 20261014
NQ = len(IA.NOMES)
AS = (Fraction(1), Fraction(2), Fraction(3), Fraction(4), Fraction(1, 2), Fraction(1, 3))
BITS_VAL = log2(901)
CORES = list(range(10))
DIM = (IA.NOMES.index("alt"), IA.NOMES.index("larg"))


def _candidatas(Fe, Fs, j):
    """Relacoes para a grandeza j da saida que valem em TODOS os pares, com custo."""
    n = len(Fe)
    alvo = [f[j] for f in Fs]
    out = []
    if len(set(alvo)) == 1:
        out.append((BITS_VAL + 2, ("const", alvo[0])))
    for i in range(NQ):
        for a in AS:
            b = alvo[0] - a * Fe[0][i]
            if b.denominator != 1 or abs(b) > 30:
                continue
            if all(a * Fe[p][i] + b == alvo[p] for p in range(n)):
                out.append((log2(NQ) + log2(len(AS)) + log2(61) + 2, ("afim", i, a, int(b))))
    for i, k in itertools.combinations(range(NQ), 2):
        if all(Fe[p][i] + Fe[p][k] == alvo[p] for p in range(n)):
            out.append((log2(comb(NQ, 2) * 2) + 2, ("soma", i, k)))
        for s in (1, -1):
            if all(s * (Fe[p][i] - Fe[p][k]) == alvo[p] for p in range(n)):
                out.append((log2(comb(NQ, 2) * 2) + 2, ("dif", i, k, s)))
    for i, k in itertools.combinations_with_replacement(range(NQ), 2):
        if all(Fe[p][i] * Fe[p][k] == alvo[p] for p in range(n)):
            out.append((log2(comb(NQ, 2) + NQ) + 2, ("prod", i, k)))
    return out


def _aplica(rel, fe):
    t = rel[0]
    if t == "const":
        return rel[1]
    if t == "afim":
        v = rel[2] * fe[rel[1]] + rel[3]
        return int(v) if v.denominator == 1 else None
    if t == "soma":
        return fe[rel[1]] + fe[rel[2]]
    if t == "dif":
        return rel[3] * (fe[rel[1]] - fe[rel[2]])
    return fe[rel[1]] * fe[rel[2]]


def relacoes(tr):
    Fe = [[Fraction(x) for x in IA.grandezas(a)] for a, _b in tr]
    Fs = [[Fraction(x) for x in IA.grandezas(b)] for _a, b in tr]
    ganho_sem = len(tr) * BITS_VAL
    rels = {}
    for j in range(NQ):
        cs = _candidatas(Fe, Fs, j)
        if not cs:
            continue
        custo, rel = min(cs, key=lambda c: c[0])
        if ganho_sem - custo > 0:
            rels[j] = rel
    return rels


def mede(tarefas, filtro):
    n = dims_ok = cont_ok = 0
    for nome, tr, te in tarefas:
        if filtro is not None and nome not in filtro:
            continue
        n += 1
        rels = relacoes(tr)
        okd = okc = True
        for a, b in te:
            fe = [Fraction(x) for x in IA.grandezas(a)]
            fs = IA.grandezas(b)
            for j in DIM:
                okd &= j in rels and _aplica(rels[j], fe) == fs[j]
            for j in CORES:
                okc &= j in rels and _aplica(rels[j], fe) == fs[j]
        dims_ok += okd
        cont_ok += okd and okc
    return n, dims_ok, cont_ok


def nulo(tarefas):
    rng = random.Random(SEMENTE_NULO)
    saidas = [b for _n, tr, te in tarefas for _a, b in tr + te]
    return [(nome, [(a, rng.choice(saidas)) for a, _b in tr],
             [(a, rng.choice(saidas)) for a, _b in te]) for nome, tr, te in tarefas]


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tarefas = carregar(PASTA_AVAL)
    novas = {nome for nome, tr, _te in tarefas if "formas_novas" in assinaturas(tr)}
    nul = nulo(tarefas)
    print("=" * 76)
    print("RELACOES ARC — a entrada como manual: parametros da saida achados no treino")
    print("=" * 76)
    for rot, conj in (("formas_novas", novas), ("todas as 400", None)):
        n, d, c = mede(tarefas, conj)
        nn, dn, cn = mede(nul, conj)
        print(f"  {rot:13s} n={n:3d}   dimensoes certas {d:3d} ({100*d/n:4.1f}%)   "
              f"contagem completa {c:3d} ({100*c/n:4.1f}%)   |  nulo: "
              f"{100*dn/max(nn,1):4.1f}% / {100*cn/max(nn,1):4.1f}%")
    n, d, c = mede(tarefas, novas)
    nn, dn, cn = mede(nul, novas)
    print("")
    print("=" * 76)
    print(f"R1 {'.' if d / n >= 0.60 and dn / nn <= 0.20 else 'FALHOU .'} dimensoes: "
          f"{100*d/n:.1f}% (>= 60%), nulo {100*dn/nn:.1f}% (<= 20%)")
    print(f"R2 {'.' if c / n >= 0.25 and cn / nn <= 0.05 else 'FALHOU .'} contagem completa: "
          f"{100*c/n:.1f}% (>= 25%), nulo {100*cn/nn:.1f}% (<= 5%)")


if __name__ == "__main__":
    main()

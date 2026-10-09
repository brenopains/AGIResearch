"""
mente_web.inferencia — a REGRA MAIS CURTA do que acontece (o principio do nucleo do ARC)
=========================================================================================
O nucleo do ARC (AGI 2.0) escolhe, entre explicacoes, a que descreve os exemplos com
menos bits (MDL: bits do programa + bits do que o programa nao explica). Aqui o mesmo
principio roda sobre o que ELA vive: a sequencia de verbos visuais de um video ("como
a cena muda" quadro a quadro). Candidatas, da mais simples para a mais rica:

  constante(a)      "sempre o mesmo verbo"
  inercia           "o que acabou de acontecer continua"
  ciclo(a b ..)     "um padrao que se repete" (periodo 2 a 6)
  markov            "depois de X costuma vir Y" (tabela tirada do proprio video)

A regra escolhida PREVE o proximo verbo antes de ele acontecer. Conta como inferencia
so o que bate as duas apostas burras: "o mais frequente" e "repete o ultimo".
Limite declarado: o DSL do ARC (grades) e o inventor de grandezas (trajetorias) pedem
objetos rastreados, que ela ainda nao extrai; isso e o proximo degrau, nao esta aqui.
"""
from __future__ import annotations

import collections
import math

EPS = 0.05


def _bits(seq, prever, K):
    b = 0.0
    for i in range(1, len(seq)):
        b += -math.log2(1 - EPS) if prever(seq[:i]) == seq[i] else -math.log2(EPS / max(K - 1, 1))
    return b


def melhor_regra(seq, K):
    """-> (descricao, funcao historico->proximo, bits). seq: lista de inteiros (verbos)."""
    n = len(seq)
    if n < 4:
        return None
    lk = math.log2(max(K, 2))
    cands = []
    for a in set(seq):
        cands.append((("constante", (a,)), (lambda h, a=a: a), lk))
    cands.append((("inercia", ()), (lambda h: h[-1]), 1.0))
    for L in range(2, 7):
        if n < 2 * L:
            break
        pad = tuple(collections.Counter(seq[k::L]).most_common(1)[0][0] for k in range(L))
        if len(set(pad)) == 1:
            continue
        cands.append((("ciclo", pad), (lambda h, pad=pad: pad[len(h) % len(pad)]), L * lk + math.log2(6)))
    tab = collections.defaultdict(collections.Counter)
    for x, y in zip(seq, seq[1:]):
        tab[x][y] += 1
    tabela = {x: c.most_common(1)[0][0] for x, c in tab.items()}
    comum = collections.Counter(seq).most_common(1)[0][0]
    cands.append((("markov", tuple(sorted(tabela.items()))), (lambda h, t=tabela, c=comum: t.get(h[-1], c)),
                  2 * lk * len(tabela)))
    melhor = None
    for desc, f, modelo in cands:
        tot = modelo + _bits(seq, f, K)
        if melhor is None or tot < melhor[2]:
            melhor = (desc, f, tot)
    return melhor


def descreve(desc):
    nome, args = desc
    if nome == "constante":
        return f"sempre o verbo {args[0]}"
    if nome == "inercia":
        return "o que acabou de acontecer continua"
    if nome == "ciclo":
        return "ciclo " + " → ".join(str(a) for a in args)
    return "depois de X vem Y (" + ", ".join(f"{a}→{b}" for a, b in args[:4]) + ("…" if len(args) > 4 else "") + ")"

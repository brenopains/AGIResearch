"""
novaphys.procedencia — onde o rastro morre
============================================

A PERGUNTA
------------
O usuário propõe escalar datasets multimodais e gerais, **com procedência**, de
modo a identificar de qual fonte veio cada nutriente. A proposta é boa e a
procedência é a parte que a torna científica — sem ela não dá para separar
"esta fonte alimentou" de "esta fonte envenenou", e [[variedade-paga]] (mais
fontes melhoram o ranking) e [[integrar-exige-dose-e-afinidade]] (ruído não
dilui, ENVENENA) medem coisas diferentes que seriam escaladas juntas.

Mas antes de coletar volume, uma medida barata: **a cadeia atravessa?**

    fonte -> representação -> fragmento executável -> teste independente -> reuso

O volume útil é o que atravessa. Se o rastro morre num degrau, terabytes
empilham no degrau anterior.

O QUE JÁ EXISTE, CONFERIDO NO CÓDIGO ANTES DE CONSTRUIR
---------------------------------------------------------
`Acervo.guarda` já grava `rotulo` (procedência), `bruto`, `lexico` (o nutriente),
`conta` e o vetor HDC. E o fragmento EXECUTÁVEL também já existe: um léxico é um
segmentador — é o que [[mdl-descobre-palavras]] e [[descobre-andar]] rodam.
Minha previsão anterior, de que o rastro morria entre representação e executável,
estava errada antes de medir.

OS CINCO DEGRAUS, E COMO CADA UM É CONTADO
--------------------------------------------
  1 FONTE        o material existe e é lido
  2 REPRESENTA   o léxico extraído é maior que o alfabeto (senão não houve
                 extração: [[limiar-de-dado-e-lei]] mediu que abaixo do limiar
                 o léxico É o alfabeto, zero e não fração)
  3 EXECUTÁVEL   o léxico vira segmentador e roda sem erro
  4 TESTE INDEP  o segmentador comprime dado RETIDO da MESMA fonte melhor que
                 o trivial (alfabeto puro)
  5 REUSO        o segmentador de A comprime dado de B melhor que o trivial de
                 B — atravessa a fonte

PRÉ-REGISTRO
--------------
  W0  PRÉ-CONDIÇÃO. As fontes são distinguíveis: o léxico de cada uma difere
      das outras. Se todos os léxicos forem iguais, não há procedência a medir.
  W1  o funil tem um degrau onde a sobrevivência cai para ~0 -> esse degrau é o
      endereço, e coletar volume antes dele é desperdício
  W2  todos os degraus sobrevivem -> a cadeia atravessa e escalar dado tem
      fundamento medido; a lista de domínios do usuário vira plano de coleta

PREVISÃO: o rastro morre no degrau 5, REUSO. Mecanismo:
[[integrar-exige-dose-e-afinidade]] já mediu que o ganho de +0,095 que parecia
transferência entre domínios era 32,6% de vocabulário compartilhado, e que
stdlib em inglês dá −0,057. Se estiver certo, os degraus 1-4 passam e o 5 fica
perto de zero exceto entre fontes que compartilham vocabulário de superfície —
o que seria procedência funcionando e revelando que "geral" não atravessa.
"""

from __future__ import annotations

import io
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from novaphys.lingua import _viterbi, segmenta

# `lingua` nao exporta alfabeto; os modulos que a usam o declaram.
ALFA = "abcdefghijklmnopqrstuvwxyz"


def _bits_lex(s: str, lex: set) -> float:
    """Bits para descrever `s` com este léxico. O fragmento EXECUTÁVEL rodando."""
    prob = {w: 1.0 / max(len(lex), 1) for w in lex}
    ped = _viterbi(s, lex, prob)
    from math import log2
    return sum(log2(max(len(lex), 2)) for _ in ped)


def _bits_trivial(s: str) -> float:
    from math import log2
    return len(s) * log2(max(len(set(s)), 2))


def funil(fontes: Dict[str, Tuple[str, str]], rodadas: int = 300,
          cand: int = 24) -> None:
    """
    `fontes[rotulo] = (treino, retido)` — texto já normalizado ao alfabeto.
    Conta quantas fontes sobrevivem a cada degrau, e reporta POR FONTE.
    """
    print("=" * 78)
    print("PROCEDENCIA — onde o rastro morre")
    print("=" * 78)
    print("  fonte -> representa -> executavel -> teste indep -> reuso\n")

    lex: Dict[str, set] = {}
    print("  %-14s %8s %8s   degrau 1  2  3" % ("fonte", "chars", "|lexico|"))
    print("  " + "-" * 56)
    for rot, (tr, _ho) in fontes.items():
        d1 = len(tr) > 0
        try:
            _t, l, _a, _b = segmenta(tr, rodadas=rodadas, cand=cand)
            l = {w for w in l if len(w) > 1}
        except Exception:
            l = set()
        d2 = len(l) > 0                      # lexico acima do alfabeto
        d3 = False
        if d2:
            try:
                _ = _bits_lex(tr[:2000], l | set(ALFA))
                d3 = True
            except Exception:
                d3 = False
        lex[rot] = l | set(ALFA) if d3 else set()
        print("  %-14s %8d %8d        %s  %s  %s"
              % (rot, len(tr), len(l), "ok" if d1 else "--",
                 "ok" if d2 else "--", "ok" if d3 else "--"), flush=True)

    # W0: as fontes sao distinguiveis?
    vivos = [r for r in fontes if lex[r]]
    print("\n  W0 — os lexicos diferem entre fontes?")
    for i, a in enumerate(vivos):
        for b in vivos[i + 1:]:
            la, lb = lex[a] - set(ALFA), lex[b] - set(ALFA)
            j = len(la & lb) / max(len(la | lb), 1)
            print("      %-12s x %-12s  Jaccard %.3f" % (a, b, j))

    print("\n  degrau 4 — TESTE INDEPENDENTE (dado retido da MESMA fonte)")
    print("  %-14s %10s %10s %8s" % ("fonte", "trivial", "com lexico", "ganho"))
    print("  " + "-" * 48)
    d4 = {}
    for rot in vivos:
        _tr, ho = fontes[rot]
        if not ho:
            continue
        bt, bl = _bits_trivial(ho), _bits_lex(ho, lex[rot])
        d4[rot] = (bt - bl) / max(bt, 1e-9)
        print("  %-14s %10.0f %10.0f %+7.1f%%"
              % (rot, bt, bl, 100 * d4[rot]), flush=True)

    print("\n  degrau 5 — REUSO (lexico de A no dado retido de B)")
    print("  %-14s" % "lexico de \\ em", end="")
    for b in vivos:
        print(" %10s" % b[:10], end="")
    print()
    print("  " + "-" * (14 + 11 * len(vivos)))
    atravessa = 0
    total = 0
    for a in vivos:
        print("  %-14s" % a[:14], end="")
        for b in vivos:
            _tr, ho = fontes[b]
            if not ho:
                print(" %10s" % "-", end="")
                continue
            bt = _bits_trivial(ho)
            g = (bt - _bits_lex(ho, lex[a])) / max(bt, 1e-9)
            if a != b:
                total += 1
                atravessa += (g > 0.02)
            print(" %+9.1f%%" % (100 * g), end="")
        print(flush=True)

    print("\n" + "=" * 78)
    print("  degrau 4 (mesma fonte):  %d de %d com ganho > 2%%"
          % (sum(1 for v in d4.values() if v > 0.02), len(d4)))
    print("  degrau 5 (fonte cruzada): %d de %d com ganho > 2%%"
          % (atravessa, total))
    print()
    if total and atravessa <= total * 0.2:
        print("W1 . O RASTRO MORRE NO DEGRAU 5. O fragmento atravessa a extracao,")
        print("     a execucao e o teste na propria fonte, e NAO atravessa para")
        print("     outra fonte. Coletar volume geral empilha no degrau 4.")
    elif total and atravessa >= total * 0.6:
        print("W2 . A CADEIA ATRAVESSA. Escalar dado tem fundamento medido.")
    else:
        print("parcial: %d/%d atravessam — procedencia diz QUAIS." % (atravessa, total))


# ============================================================================
# ADAPTACAO: o rastro morre no reuso RIGIDO. Atravessa se adaptar?
# ============================================================================
#
# O funil mediu reuso RIGIDO — aplicar o lexico de A em B sem tocar nele — e deu
# 1 de 12. Mas ha tres niveis acima disso, e nenhum foi testado:
#
#   REPARAM      lexico de A, probabilidades reestimadas em B
#   SUBSTITUI    lexico de A filtrado ao que de fato ocorre em B
#   REESTRUTURA  uniao de A com o lexico proprio de B, podada pelo que paga
#
# E a pergunta que interessa nao e "adaptado bate rigido" (bate trivialmente,
# ja que ve dado de B). E: **adaptar a experiencia de A bate APRENDER DO ZERO
# em B, quando B tem pouco dado?** Essa e a afirmacao de experiencia acumulada:
# o que se aprendeu antes reduz o dado necessario depois.
#
# Por isso a dose entra: todos os bracos veem a MESMA fatia de B.
#
#   X0  PRE-CONDICAO: com dose grande, DO ZERO ganha de tudo (senao a medida
#       esta invertida e nada e interpretavel)
#   X1  com dose PEQUENA, algum braco adaptado bate DO ZERO em >= 2/3 dos pares
#       -> experiencia anterior reduz o dado necessario; a intuicao do usuario
#          se sustenta e o gargalo e adaptacao, nao coleta
#   X2  nenhum bate -> o fragmento e especifico da fonte em qualquer nivel de
#       adaptacao, e coletar volume geral nao converte em capacidade

from math import log2


def _probs(s: str, lex: set) -> Dict[str, float]:
    """Probabilidades estimadas em `s` com este lexico. Laplace."""
    prob = {w: 1.0 / max(len(lex), 1) for w in lex}
    ped = _viterbi(s, lex, prob)
    c = Counter(ped)
    tot = sum(c.values()) or 1
    return {w: (c.get(w, 0) + 0.5) / (tot + 0.5 * len(lex)) for w in lex}


def _bits_prob(s: str, lex: set, prob: Dict[str, float]) -> float:
    """Bits reais: -log2 da probabilidade de cada pedaco escolhido."""
    ped = _viterbi(s, lex, prob)
    return sum(-log2(max(prob.get(w, 1e-12), 1e-12)) for w in ped)


def _uniforme(lex: set) -> Dict[str, float]:
    return {w: 1.0 / max(len(lex), 1) for w in lex}


def _poda(lex: set, alvo: str, base: set) -> set:
    """REESTRUTURA: tira do lexico o que nao paga em `alvo`. Guloso, 1 passada."""
    cur = set(lex)
    b = _bits_prob(alvo, cur, _probs(alvo, cur))
    for w in sorted(lex - base, key=len, reverse=True):
        tent = cur - {w}
        if len(tent) <= len(base):
            break
        p = _probs(alvo, tent)
        bt = _bits_prob(alvo, tent, p)
        if bt <= b:
            cur, b = tent, bt
    return cur


def adaptacao(fontes: Dict[str, Tuple[str, str]], doses=(600, 2400, 9600),
              rodadas: int = 300, cand: int = 24) -> None:
    base = set(ALFA)
    print("=" * 78)
    print("ADAPTACAO — reparametrizar, substituir, reestruturar")
    print("=" * 78)
    print("  todos os bracos veem a MESMA fatia de B. A pergunta e se adaptar")
    print("  a experiencia de A bate APRENDER DO ZERO em B.\n")

    lexA = {}
    for rot, (tr, _ho) in fontes.items():
        try:
            _t, l, _a, _b = segmenta(tr, rodadas=rodadas, cand=cand)
            l = {w for w in l if len(w) > 1}
        except Exception:
            l = set()
        if l:
            lexA[rot] = l | base
    vivos = sorted(lexA)

    for dose in doses:
        print("  DOSE = %d chars de B" % dose)
        print("  %-12s %-12s %8s %8s %8s %8s %8s"
              % ("de A", "para B", "trivial", "zero", "reparam", "substit",
                 "reestr"))
        print("  " + "-" * 72)
        venc = Counter()
        pares = 0
        for b in vivos:
            trB, hoB = fontes[b]
            fatia = trB[:dose]
            try:
                _t, lb, _x, _y = segmenta(fatia, rodadas=rodadas, cand=cand)
                lb = {w for w in lb if len(w) > 1} | base
            except Exception:
                lb = base
            bt = _bits_trivial(hoB)
            bz = _bits_prob(hoB, lb, _probs(fatia, lb))
            for a in vivos:
                if a == b:
                    continue
                pares += 1
                la = lexA[a]
                br = _bits_prob(hoB, la, _probs(fatia, la))          # REPARAM
                ls = {w for w in la - base if w in fatia} | base
                bs = _bits_prob(hoB, ls, _probs(fatia, ls))          # SUBSTITUI
                lu = _poda(la | lb, fatia, base)
                bu = _bits_prob(hoB, lu, _probs(fatia, lu))          # REESTRUTURA
                melhor = min(br, bs, bu)
                nome = ("reparam" if melhor == br else
                        "substit" if melhor == bs else "reestr")
                if melhor < bz:
                    venc[nome] += 1
                print("  %-12s %-12s %8.0f %8.0f %8.0f %8.0f %8.0f  %s"
                      % (a[:12], b[:12], bt, bz, br, bs, bu,
                         "<- %s BATE zero" % nome if melhor < bz else ""),
                      flush=True)
        n = sum(venc.values())
        print("\n    adaptado bate DO ZERO em %d de %d pares  %s\n"
              % (n, pares, dict(venc) if venc else ""))

"""
novaphys.teto_confunde — o teto de 400 avaliacoes confundiu CINCO experimentos
================================================================================

O PADRAO QUE SO APARECEU DEPOIS DE CINCO RODADAS
--------------------------------------------------
Todo acrescimo de vocabulario de hoje perdeu, e o PLACEBO pareado perdeu quase o
mesmo:

    85 propriedades boas        9/80   (-3)      placebo arbitrario  11/80  (-1)
    corte por MDL do mapa      12/80   (+0)      corte ao acaso      12/80  (+0)
    raio (1a versao)           12/80   (+0)      placebo geometrico  12/80  (+0)
    raio (emissor/barreira)    11/80   (-1)      placebo geometrico  10/80  (-2)
    LLM escrevendo codec        8/30   (+0)

Cinco medidas independentes dizendo a mesma coisa: **o dano vem do NUMERO de
candidatos, nao do conteudo deles.** Quando o placebo perde junto, a variavel
causal nao e a qualidade do vocabulario — e o ORCAMENTO.

E o teto de 400 nao e uma propriedade do problema: fui EU que o escolhi, no
comeco da sessao, para a rodada de quatro bracos do proponente cair de ~5 h para
~2 h. Depois medi tudo por cima dele. Pior: mais cedo hoje eu matei justamente o
teste desse teto, chamando de "manutencao de pouco valor". Era o confundidor de
todos os experimentos seguintes.

O QUE ESTE MODULO MEDE
------------------------
A MESMA comparacao (base x base+raio) em varios tetos, na mesma fatia de
tarefas. Se o `raio` ganha com teto alto, a leitura de hoje inverte: o
vocabulario estava certo e a parede era a busca. Se nao ganha em teto nenhum, o
vocabulario realmente nao e o que falta, e cinco nulos ficam de pe.

Fatia pequena de proposito — 30 tarefas, para caber em teto alto — e contem as 12
com assinatura de raio mais as primeiras que a base resolve, para que perda e
ganho apareçam na mesma rodada.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  T1  O TETO LIMITA A BASE: a base resolve MAIS com teto 4000 que com 400
      (se nao, o teto nunca foi a parede e eu estou errado de novo)
  T2  O RAIO PAGA COM FOLGA: em teto 4000, base+raio - base >= +2
  T3  A PERDA E ORCAMENTO: a perda de base+raio cai a zero quando o teto sobe
  T4  MONOTONIA: o ganho de base+raio sobre a base nao DIMINUI ao subir o teto
PREVISAO: T1 passa — o teto de 400 e apertado e `_montar_base` tem teto
adaptativo justamente porque cortar cedo custava tarefas. T2 e a aposta central e
eu digo que passa, +2 a +4. T3 passa. T4 passa. Se T1 falhar, a conclusao e dura
e importante: o teto nunca foi a parede, os cinco nulos valem como estao, e o
problema esta na GERACAO de candidatos e nao no orcamento para avalia-los.
"""

from __future__ import annotations

import argparse
import time

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys.raio import Raio

# as 12 com assinatura de raio, medidas por gap_generativo.py
COM_RAIO = ("05a7bcf2", "0d87d2a6", "12422b43", "13713586", "18419cfa",
            "1c56ad9f", "20981f0e", "2546ccf6", "2685904e", "29700607",
            "2b01abd0", "319f2597")


def roda(tarefas, codecs, teto, etiqueta):
    ok, t0 = set(), time.time()
    for tarefa, tr, te in tarefas:
        try:
            fm, pilha, err = D.montar(tr, codecs=codecs, teto=teto)
            if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y
                                          for x, y in te):
                ok.add(tarefa)
        except Exception:
            pass
    print(f"    [{etiqueta}] {len(ok)}/{len(tarefas)}  {time.time()-t0:.0f}s",
          flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tetos", default="400,1200,4000")
    ap.add_argument("--extra", type=int, default=18)
    a = ap.parse_args()
    todas = carregar(PASTA_AVAL)[:80]
    fatia = [t for t in todas if t[0] in COM_RAIO]
    resto = [t for t in todas if t[0] not in COM_RAIO][: a.extra]
    fatia = fatia + resto
    tetos = [int(x) for x in a.tetos.split(",")]

    print("=" * 78)
    print(f"TETO CONFUNDE — {len(fatia)} tarefas ({len(COM_RAIO)} com assinatura "
          f"de raio), tetos {tetos}")
    print("=" * 78)

    linhas = []
    for teto in tetos:
        print(f"  teto {teto}:", flush=True)
        b = roda(fatia, D.CODECS, teto, f"base {teto}")
        c = roda(fatia, D.CODECS + [Raio()], teto, f"raio {teto}")
        linhas.append((teto, len(b), len(c), len(c - b), len(b - c)))

    print("")
    print("=" * 78)
    print(f"  {'teto':>6s} {'base':>5s} {'+raio':>6s} {'ganhou':>7s} {'perdeu':>7s}")
    for teto, nb, nc, g, p in linhas:
        print(f"  {teto:6d} {nb:5d} {nc:6d} {g:7d} {p:7d}")

    base0 = linhas[0][1]
    baseF = linhas[-1][1]
    gF = linhas[-1][2] - linhas[-1][1]
    pF = linhas[-1][4]
    ganhos = [nc - nb for _t, nb, nc, _g, _p in linhas]
    print("")
    print(f"T1 {'.' if baseF > base0 else 'FALHOU .'} o teto limitava a base: "
          f"{base0} -> {baseF}")
    print(f"T2 {'.' if gF >= 2 else 'FALHOU .'} raio paga em teto alto: "
          f"{gF:+d} (>= +2)")
    print(f"T3 {'.' if pF == 0 else 'FALHOU .'} a perda virou zero: {pF}")
    print(f"T4 {'.' if ganhos == sorted(ganhos) else 'FALHOU .'} ganho nao "
          f"diminui com teto maior: {ganhos}")


if __name__ == "__main__":
    main()

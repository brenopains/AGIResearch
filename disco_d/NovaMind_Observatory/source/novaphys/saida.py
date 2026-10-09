"""
novaphys.saida — as SAIDAS do ARC sao formas canonicas recorrentes?
====================================================================

A PREMISSA QUE PRECISA SER VERIFICADA ANTES DE CONSTRUIR
----------------------------------------------------------
Proposta do usuario: o ARC nao pergunta "como o objeto mudou", pergunta "qual e
a figura certa no final". Entao, em vez de calcular delta, GERAR diretamente a
figura final — a partir de um banco de "formas canonicas": quadriculado
perfeito, forma simetrica, linhas paralelas uniformes.

A parte da direcao esta certa e o censo concorda: 199 das nao resolvidas
produzem formas que nao existiam na entrada.

Mas o BANCO tem uma premissa embutida: que as saidas do ARC sejam formas
tipicas RECORRENTES. Se forem, um banco pequeno cobre muita coisa. Se cada
saida for idiossincratica, "banco de formas canonicas" e uma lista de 400
casos, e construir isso e escrever a resposta a mao.

Esta sessao aprendeu, a duras penas, a medir a premissa antes de construir em
cima dela: `dica.py` custou 154 segundos e matou um sistema de cinco estagios
ao medir que professor perfeito compra +3. Este arquivo faz o mesmo aqui.

O QUE SE MEDE, e so nas SAIDAS
--------------------------------
Ignora-se a entrada por completo. Para cada grade de saida dos pares de treino,
pergunta-se se ela e uma forma canonica, com a assinatura valendo em TODAS as
saidas daquela tarefa:

    uniforme       uma cor so
    simetrica_h    espelho horizontal de si mesma
    simetrica_v    espelho vertical
    simetrica_d    simetrica na diagonal (transposta igual)
    periodica      repeticao de um bloco menor (quadriculado, listras)
    poucas_cores   no maximo 2 cores
    pequena        no maximo 9 celulas
    igual_entrada  identica a entrada (nao muda nada)

E a medida que decide: quantas saidas DISTINTAS existem? Se as tarefas
compartilham saidas, ha banco. Se cada tarefa tem a sua, nao ha.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  S1  >= 40% das saidas sao canonicas por alguma assinatura
      -> o banco existe e vale construir o gerador
  S2  15% a 39%
      -> ha um nucleo canonico, mas ele cobre uma minoria
  S3  < 15%
      -> as saidas sao idiossincraticas e "banco de formas canonicas" seria
         escrever a resposta a mao. A direcao de GERAR continua certa; o banco
         e que nao existe.

  S4  REPETICAO ENTRE TAREFAS: quantas saidas aparecem em mais de uma tarefa?
      Se ~zero, nao ha vocabulario compartilhado de figuras finais, e o
      gerador teria que INVENTAR cada saida em vez de reconhece-la de um banco.

MINHA CALIBRAGEM: vinte e quatro previsoes na sessao, OITO certas, as dezesseis
erradas TODAS otimistas.

PREVISAO: S2 e S4 proximo de zero. Espero um nucleo canonico de 15-30% (muitas
saidas do ARC sao simetricas ou de poucas cores, quase por construcao do
benchmark), e espero repeticao entre tarefas praticamente nula — porque o ARC
foi montado justamente para que decorar saidas nao funcione.

Se S4 der ~zero, a leitura e precisa e util: gerar a figura final e a direcao
certa, mas nao ha banco a reconhecer — ha figura a CONSTRUIR, e cada uma e
nova. O que muda tudo sobre como atacar.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, List, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.eter import PASTA_AVAL


def _periodica(g: Grid) -> bool:
    h, w = dims(g)
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                continue
            if h % ph or w % pw:
                continue
            if all(g[i][j] == g[i % ph][j % pw]
                   for i in range(h) for j in range(w)):
                return True
    return False


def assinaturas(b: Grid) -> Set[str]:
    h, w = dims(b)
    cores = {c for l in b for c in l}
    fora = set()
    if len(cores) == 1:
        fora.add("uniforme")
    if b == tuple(tuple(l[::-1]) for l in b):
        fora.add("simetrica_h")
    if b == b[::-1]:
        fora.add("simetrica_v")
    if h == w and b == tuple(zip(*b)):
        fora.add("simetrica_d")
    if _periodica(b):
        fora.add("periodica")
    if len(cores) <= 2:
        fora.add("poucas_cores")
    if h * w <= 9:
        fora.add("pequena")
    return fora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    print("=" * 74)
    print("SAIDA — as saidas do ARC sao formas canonicas recorrentes?")
    print("=" * 74)
    tarefas = carregar(PASTA_AVAL)

    por_tarefa: Dict[str, Set[str]] = {}
    todas_saidas: List[Grid] = []
    iguais = 0
    for n, tr, _te in tarefas:
        comuns = None
        for a, b in tr:
            todas_saidas.append(b)
            s = assinaturas(b)
            if a == b:
                s.add("igual_entrada")
            comuns = s if comuns is None else (comuns & s)
        por_tarefa[n] = comuns or set()
        if "igual_entrada" in (comuns or set()):
            iguais += 1

    n = len(tarefas)
    c = Counter(a for s in por_tarefa.values() for a in s)
    sem = sum(1 for s in por_tarefa.values() if not s)
    print(f"\n{n} tarefas, {len(todas_saidas)} grades de saida\n")
    print(f"{'assinatura':16s} {'tarefas':>8s} {'%':>7s}")
    print("-" * 34)
    for k, v in c.most_common():
        print(f"{k:16s} {v:>8d} {v/n:>7.1%}")
    print("-" * 34)
    print(f"{'NENHUMA':16s} {sem:>8d} {sem/n:>7.1%}")

    canon = n - sem
    # S4 — a mesma saida aparece em mais de uma tarefa?
    dono: Dict[Grid, Set[str]] = {}
    for nm, tr, _te in tarefas:
        for _a, b in tr:
            dono.setdefault(b, set()).add(nm)
    compart = [g for g, s in dono.items() if len(s) > 1]
    print(f"\nS4 · saidas DISTINTAS: {len(dono)} de {len(todas_saidas)}")
    print(f"     saidas que aparecem em MAIS DE UMA tarefa: {len(compart)}")

    print("\n" + "=" * 74)
    p = canon / n
    if p >= 0.40:
        print(f"S1 · {p:.0%} das tarefas tem saida canonica. O banco existe e")
        print("     vale construir o gerador.")
    elif p >= 0.15:
        print(f"S2 · {p:.0%} tem nucleo canonico, mas e minoria.")
    else:
        print(f"S3 · so {p:.0%}. As saidas sao idiossincraticas.")
    if len(compart) <= 2:
        print("\nS4 · praticamente NENHUMA saida se repete entre tarefas. Nao")
        print("     ha vocabulario compartilhado de figuras finais: o gerador")
        print("     teria que INVENTAR cada saida, nao reconhece-la de um")
        print("     banco. Gerar continua certo; banco nao existe.")
    else:
        print(f"\nS4 · {len(compart)} saidas se repetem entre tarefas. Ha")
        print("     algum vocabulario compartilhado.")


if __name__ == "__main__":
    main()

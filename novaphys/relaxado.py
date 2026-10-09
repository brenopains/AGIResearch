"""
novaphys.relaxado — deixar estruturas INCOMPLETAS viverem e se combinarem
===========================================================================

O QUE ISTO TESTA
------------------
`eter_inferencia.py` deduz o operador resolvendo as restricoes, e acha 3/400
com precisao de 100%. Mas ele e tudo-ou-nada: o par (leitura_v, leitura_f)
precisa cobrir 100% das celulas ou o candidato morre.

Isso e o oposto de imaginacao. Uma peneira que exige perfeicao no primeiro
passo mata todo estado intermediario — nenhuma estrutura meio formada
sobrevive tempo suficiente para se combinar com outra.

O regime relaxado guarda operadores que explicam a MAIOR PARTE das celulas
(>= `limiar`), aplica um deles, e roda a inferencia de novo sobre o que
sobrou. Duas estruturas incompletas compondo uma completa.

A LEITURA MECANICA DO "AFROUXAR PRIORS"
-----------------------------------------
Na literatura de estados alterados, o efeito descrito nao e "gerar formas
aleatorias": e REDUZIR A PRECISAO dos priors de cima para baixo, deixando o
sinal de baixo para cima reorganizar a representacao. Aqui isso tem forma
exata e um botao: `limiar`. Em 1.0 e o regime rigido de hoje; abaixo disso,
hipoteses parciais sobrevivem.

Nao ha nada de mistico nisso e nao ha metafora: e literalmente afrouxar o
criterio de aceitacao de hipotese intermediaria, mantendo o verificador final
intacto. O verificador continua exigindo igualdade exata em todos os pares —
o que se relaxa e o que chega a ser CONSIDERADO, nunca o que e ACEITO.

ORCAMENTO
-----------
Duas rodadas de inferencia custam ~2x uma. Por isso o braco rigido recebe o
DOBRO de tentativas de operador, para que a comparacao nao seja "relaxado fez
mais trabalho". Contado em pares de leitura examinados.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  R0  PRE-CONDICAO. Em limiar 1.0 o relaxado tem que reproduzir exatamente as
      3 tarefas do rigido. Se divergir, os dois nao sao o mesmo metodo com um
      botao, e a comparacao nao mede o botao.

  R1  algum limiar < 1.0 resolve >= 4 tarefas a mais que o rigido
      -> deixar estrutura incompleta viver paga, e ha um valor de limiar
         que da para reportar

  R2  ganho de 1 a 3 tarefas

  R3  nenhum ganho, ou perda
      -> a composicao de parciais nao alcanca; exigir perfeicao no primeiro
         passo nao era o gargalo

MINHA PREVISAO, escrita antes de rodar: R2, ganho de 1 a 3. A razao e que uma
fatia do ARC e "faca X e depois recolora" ou "faca X e depois mova", que e
exatamente uma composicao de dois passos locais. Mas tarefas de dois passos
LOCAIS sao raras — a maioria precisa de objeto, e objeto nao esta nesta
gramatica.

Nao espero milagre. Espero medir quanto vale o botao, e o valor de saber isso
e que ele passa a ser um parametro conhecido em vez de uma intuicao.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo, valida
from novaphys.eter_inferencia import (Leitura, Teste, aplicar, inferir,
                                      todas_leituras, todos_testes)
from novaphys.eter import PASTA_AVAL


def cobertura_bits(treino, leituras, bg) -> Tuple[List, int, List[int]]:
    """Mascara de acerto de cada leitura, como inteiro, e o total de celulas."""
    desloc, base = [], 0
    for a, _b in treino:
        h, w = dims(a)
        desloc.append(base)
        base += h * w
    TODAS = (1 << base) - 1
    fora = []
    for L in leituras:
        bits, n = 0, 0
        for k, (a, b) in enumerate(treino):
            v = L.valores(a, bg)
            h, w = dims(a)
            off = desloc[k]
            for i in range(h):
                vi, bi = v[i], b[i]
                for j in range(w):
                    if vi[j] == bi[j]:
                        bits |= 1 << (off + i * w + j)
                        n += 1
        if n:
            fora.append((L, bits, n))
    return fora, TODAS, desloc


def parciais(treino, limiar: float, teto: int
             ) -> Tuple[List[Tuple[Teste, Leitura, Leitura]], int]:
    """
    Operadores que explicam >= `limiar` das celulas. Em limiar 1.0 isto e
    exatamente o conjunto que o metodo rigido aceitaria.
    """
    if any(dims(a) != dims(b) for a, b in treino):
        return [], 0
    bg = fundo(treino[0][0])
    cores = sorted({c for a, b in treino for l in (a + b) for c in l})
    leituras = todas_leituras(cores)
    cob, TODAS, desloc = cobertura_bits(treino, leituras, bg)
    if not cob:
        return [], 0
    total = TODAS.bit_count()      # bin().count() era o gargalo
    alvo = int(total * limiar)

    testes = todos_testes(cores)
    mt = []
    for T in testes:
        bits = 0
        for k, (a, _b) in enumerate(treino):
            mk = T.mascara(a, bg)
            h, w = dims(a)
            off = desloc[k]
            for i in range(h):
                for j in range(w):
                    if mk[i][j]:
                        bits |= 1 << (off + i * w + j)
        mt.append(bits)

    cob.sort(key=lambda x: -x[2])
    fora: List[Tuple[Teste, Leitura, Leitura]] = []
    gasto = 0
    for Lv, bv, _nv in cob:
        if len(fora) >= teto:
            break
        for Lf, bf, _nf in cob:
            gasto += 1
            if len(fora) >= teto:
                break
            juntos = bv | bf
            if juntos.bit_count() < alvo:
                continue
            # o teste precisa ser verdadeiro onde SO v acerta, falso onde SO f
            ev, ef = bv & ~bf, bf & ~bv
            for T, bt in zip(testes, mt):
                if (ev & ~bt) == 0 and (ef & bt) == 0:
                    fora.append((T, Lv, Lf))
                    break
    return fora, gasto


def resolver(treino, teste, limiar: float, teto: int
             ) -> Tuple[Optional[str], int]:
    """
    Um passo rigido; se falhar, um passo parcial seguido de inferencia sobre
    o residuo. O verificador final continua exigindo igualdade exata.
    """
    op, gasto = inferir(treino)
    if op is not None:
        if all(aplicar(op, x) == y for x, y in teste):
            return f"1 passo: [{op[0]} ? {op[1]} : {op[2]}]", gasto
        return None, gasto

    # CONTROLE. A primeira versao retornava aqui quando limiar==1.0, ou seja
    # o braco "rigido" fazia UM passo e os relaxados faziam DOIS. Isso
    # confunde relaxamento com profundidade: o ganho medido podia ser todo
    # de compor, e nada de afrouxar. Agora o limiar 1.0 tambem compoe — com
    # operadores que cobrem 100% — e a diferenca passa a isolar o botao.

    cand, g2 = parciais(treino, limiar, teto)
    gasto += g2
    for p in cand:
        meio = []
        bom = True
        for a, b in treino:
            r = aplicar(p, a)
            if r is None or dims(r) != dims(b):
                bom = False
                break
            meio.append((r, b))
        if not bom:
            continue
        op2, g3 = inferir(meio)
        gasto += g3
        if op2 is None:
            continue
        ok = True
        for x, y in teste:
            r = aplicar(p, x)
            if r is None:
                ok = False
                break
            r = aplicar(op2, r)
            if r != y:
                ok = False
                break
        if ok:
            return (f"2 passos: [{p[0]} ? {p[1]} : {p[2]}] "
                    f"-> [{op2[0]} ? {op2[1]} : {op2[2]}]"), gasto
    return None, gasto


def uma(args):
    nome, tr, te, limiar, teto = args
    e, g = resolver(tr, te, limiar, teto)
    return nome, e, g


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--teto", type=int, default=25,
                    help="operadores parciais tentados por tarefa")
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    tarefas = carregar(a.pasta)
    print("=" * 76)
    print("RELAXADO — deixar estruturas incompletas viverem e se combinarem")
    print("=" * 76)
    print("limiar 1.00 = regime rigido de hoje (tudo ou nada)")
    print(f"abaixo disso, operadores parciais sobrevivem e sao compostos")
    print(f"teto de {a.teto} parciais por tarefa\n")

    print(f"{'limiar':>7} {'resolve':>9} {'2 passos':>10} "
          f"{'gasto medio':>13}")
    print("-" * 44)
    base: Set[str] = set()
    for limiar in (1.0, 0.95, 0.85, 0.70, 0.50):
        t0 = time.time()
        with Pool(a.nucleos) as pool:
            res = pool.map(uma, [(n, tr, te, limiar, a.teto)
                                 for n, tr, te in tarefas], chunksize=4)
        ok = {n for n, e, _g in res if e}
        dois = sum(1 for _n, e, _g in res if e and e.startswith("2"))
        gasto = sum(g for _n, _e, g in res) / len(res)
        if limiar == 1.0:
            base = ok
        print(f"{limiar:>7.2f} {len(ok):>9} {dois:>10} {gasto:>13,.0f}"
              f"   (+{len(ok - base)} vs rigido, {time.time()-t0:.0f}s)")
        if limiar < 1.0 and ok - base:
            for n, e, _g in res:
                if n in ok - base:
                    print(f"          {n} -> {e[:78]}")

    print("-" * 44)
    print("\nR0 · o limiar 1.00 tem que reproduzir o rigido (3 tarefas)")
    print(f"     deu {len(base)}: {sorted(base)}")


if __name__ == "__main__":
    main()

"""
novaphys.curva — o retorno por codec e linear ou superlinear?
==============================================================

A PERGUNTA QUE DECIDE
-----------------------
`descricao.py` mostrou que credito parcial produz composicao (D1 passou: tres
tarefas por pilha de dois codecs, uma delas inedita). Isso valida o mecanismo.
Nao diz se ele ESCALA.

A diferenca entre "mais um +1" e "salto exponencial" e a forma da curva:

    LINEAR       cada codec traz as tarefas que so ele resolve.
                 N codecs -> c*N tarefas. E a arquitetura de hoje, onde a
                 uniao de 14 motores e a soma de 14 numeros pequenos.

    SUPERLINEAR  cada codec novo entra em pilha com TODOS os anteriores.
                 O numero de pilhas de tamanho <= k cresce como N^k. Se as
                 tarefas moram em pilhas e nao em codecs isolados, a curva
                 dobra para cima.

Isto e mensuravel agora, sem construir nada: rodar o nucleo com subconjuntos
de tamanho k dos codecs que ja existem, e ver a forma.

E o teste e honesto porque e barato de refutar: se a curva for reta, a tese do
credito parcial e verdadeira mas irrelevante, e eu volto a escrever motores um
por um.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C1  CONVEXA. O ganho medio do k-esimo codec CRESCE com k.
      -> pilhas sao o mecanismo; a estrategia passa a ser "mais codecs
         pequenos", nao "motores maiores"
  C2  RETA. ganho por codec constante -> credito parcial ajuda mas nao escala
  C3  CONCAVA. ganho por codec cai -> saturacao, e o gargalo esta noutro lugar

MINHA CALIBRAGEM: trinta e seis previsoes, ONZE certas (D1 acertou).

PREVISAO: C3, concava. Com onze codecs a maioria das pilhas boas usa `copia`
mais um, e `copia` ja esta em quase todas — entao o segundo codec rende muito e
o quinto rende pouco. Espero concavidade E composicao ao mesmo tempo, que e o
resultado chato e provavelmente o verdadeiro. Se der C1 eu estou errado do jeito
que interessa.
"""

from __future__ import annotations

import argparse
import random
import time
from itertools import combinations
from typing import List

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

TAREFAS = None


def _ini():
    global TAREFAS
    TAREFAS = [(n, tr, te) for n, tr, te in carregar(PASTA_AVAL)
               if not any(dims(a) != dims(b) for a, b in tr)]


def _uma(args):
    """(indices do subconjunto, indice da tarefa) -> resolveu?"""
    ids, ti = args
    n, tr, te = TAREFAS[ti]
    cods = [D.CODECS[i] for i in ids]
    try:
        pilha, err = D.montar(tr, codecs=cods)
        if err != 0 or not pilha:
            return ids, n, 0, 0
        ok = all(D.aplicar(pilha, x) == y for x, y in te)
        return ids, n, 1 if ok else 0, len(pilha)
    except Exception:
        return ids, n, 0, 0


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    ap.add_argument("--amostras", type=int, default=6)
    a = ap.parse_args()

    _ini()
    N = len(D.CODECS)
    print("=" * 76)
    print("CURVA — o retorno por codec e linear ou superlinear?")
    print("=" * 76)
    print(f"  {N} codecs, {len(TAREFAS)} tarefas com dims iguais")
    print("")

    rnd = random.Random(11)
    subs = []
    for k in range(1, N + 1):
        todos = list(combinations(range(N), k))
        esc = todos if len(todos) <= a.amostras else rnd.sample(todos, a.amostras)
        subs.extend(esc)

    trab = [(s, ti) for s in subs for ti in range(len(TAREFAS))]
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, trab, chunksize=64)

    por_sub = {}
    tarefas_por_sub = {}
    for ids, n, ok, tam in res:
        por_sub[ids] = por_sub.get(ids, 0) + ok
        if ok:
            tarefas_por_sub.setdefault(ids, []).append((n, tam))

    print(f"  {len(subs)} subconjuntos x {len(TAREFAS)} tarefas "
          f"({time.time()-t0:.0f}s)")
    print("")
    print("   k   subconj   media   melhor   pilhas 2+   ganho do k-esimo")
    print("  ---  -------  -------  ------   ---------   ----------------")
    med = {}
    ant = 0.0
    for k in range(1, N + 1):
        ss = [s for s in subs if len(s) == k]
        if not ss:
            continue
        vals = [por_sub.get(s, 0) for s in ss]
        m = sum(vals) / len(vals)
        med[k] = m
        comp = sum(1 for s in ss for _n, t in tarefas_por_sub.get(s, [])
                   if t >= 2) / len(ss)
        print(f"  {k:3d}  {len(ss):7d}  {m:7.2f}  {max(vals):6d}   "
              f"{comp:9.2f}   {m-ant:+16.2f}")
        ant = m

    ganhos = [med[k] - med[k - 1] for k in range(2, N + 1) if k in med]
    if len(ganhos) >= 4:
        meio = len(ganhos) // 2
        prim = sum(ganhos[:meio]) / meio
        ult = sum(ganhos[meio:]) / (len(ganhos) - meio)
        print("")
        print(f"  ganho medio, primeira metade dos codecs: {prim:.3f}")
        print(f"  ganho medio, segunda  metade dos codecs: {ult:.3f}")
        print("")
        print("=" * 76)
        if ult > prim * 1.15:
            print("C1 . CONVEXA. Pilhas sao o mecanismo: mais codecs pequenos.")
        elif ult > prim * 0.85:
            print("C2 . RETA. Credito parcial ajuda mas nao escala sozinho.")
        else:
            print("C3 . CONCAVA. Satura; o gargalo esta noutro lugar.")


if __name__ == "__main__":
    main()

"""
novaphys.soma — o ganho do mundo sobrevive a busca inteira?
============================================================

POR QUE ESTE ARQUIVO EXISTE
-----------------------------
`fisica.py` mediu o efeito da dobra a PROFUNDIDADE 2, onde a base resolve
9/400. Deu +3, com o controle de permutacao em zero. Mas a base de referencia
da sessao e 14/400, a PROFUNDIDADE 3. E ha um efeito medido em `eter.py` que
pode comer o ganho inteiro:

    DILUICAO. Com orcamento casado, acrescentar primitivas aumenta o fator de
    ramificacao e o mesmo numero de candidatos passa a cobrir uma fracao menor
    do espaco. Em `eter.py`, 48-137 operadores novos renderam +1 liquido e
    PERDERAM duas tarefas que a base resolvia (`0c786b71`, `833dafe3`).

A profundidade 3 e onde isso morde: 24 primitivas dao 24^3 = 13.824 sequencias
e cabem no orcamento de 14.424. Com 38 primitivas sao 54.872 — o orcamento
corta a busca antes do fim. Entao o braco com mais primitivas paga um preco
real, e e assim que tem que ser: quem ganha vocabulario perde alcance.

O TESTE
---------
Mesmo orcamento, mesma profundidade, so muda o vocabulario:

    BASE          24 primitivas escritas a mao
    BASE+RUIDO    24 + 14 permutacoes extraidas do mundo sem estrutura
    BASE+DOBRA    24 + 14 leis extraidas do mundo com fisica e camadas

O braco RUIDO nao e decoracao: ele paga EXATAMENTE o mesmo preco de diluicao
que o braco DOBRA. Se os dois cairem junto, a queda e diluicao. Se so o RUIDO
cair, a diferenca e a estrutura do mundo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  S0  PRE-CONDICAO. BASE tem que dar 14/400 a profundidade 3. Se nao der,
      este arquivo diverge do resto da sessao e a comparacao nao vale.

  S1  DOBRA > BASE  e  DOBRA > RUIDO       o ganho sobrevive a diluicao
  S2  DOBRA = BASE  e  DOBRA > RUIDO       a diluicao empata o ganho; o
                                           efeito e real mas nao paga o custo
  S3  DOBRA < BASE                         a diluicao come o ganho inteiro
  S4  DOBRA = RUIDO                        nao era estrutura, era vocabulario

MINHA PREVISAO, escrita antes de rodar: S2 ou S3. A profundidade 3 e cara e
14 primitivas a mais quase quadruplicam o espaco. As tres tarefas ganhas
(`195ba7dc`, `5d2a5c43`, `d282b262`) sao de 1 passo e devem sobreviver, mas
aposto que a base perde de 2 a 4 tarefas de 3 passos que o orcamento deixa de
alcancar. Espero DOBRA entre 12 e 15, com RUIDO abaixo dos dois.

Se der S1, e a primeira vez na sessao que uma primitiva NAO escrita por mim
move o numero de referencia. Se der S3, a leitura tambem e util e desagradavel:
o ganho de prior extraido do mundo e real e mesmo assim NEGATIVO no total,
porque busca exaustiva cega nao suporta vocabulario grande. Nesse caso o
gargalo nao e o prior — e o mecanismo de busca, e adicionar prior bom a uma
busca burra piora.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .fisica import (PASTA_AVAL, PRIMITIVAS, ajustar_mapa, aplicar_mapa,
                     carregar, extrair, valida)

Grid = Tuple[Tuple[int, ...], ...]


def resolver(treino, extras: Sequence[Tuple[str, Callable]],
             orcamento: int, prof: int
             ) -> Tuple[Optional[List[str]], Optional[Dict], int]:
    d = dict(PRIMITIVAS)
    d.update(dict(extras))
    nomes = list(d)
    gasto = 0
    for p in range(0, prof + 1):
        for seq in itertools.product(nomes, repeat=p):
            if gasto >= orcamento:
                return None, None, gasto
            gasto += 1
            saidas, ok = [], True
            for a, _b in treino:
                r: Optional[Grid] = a
                for n in seq:
                    r = d[n](r)
                    if not valida(r):
                        ok = False
                        break
                if not ok:
                    break
                saidas.append(r)
            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, treino)):
                return list(seq), None, gasto
            m = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
            if m is not None and all(aplicar_mapa(r, m) == b
                                     for r, (_, b) in zip(saidas, treino)):
                return list(seq), m, gasto
    return None, None, gasto


def confere(seq, mapa, extras, teste) -> bool:
    d = dict(PRIMITIVAS)
    d.update(dict(extras))
    for a, b in teste:
        r: Optional[Grid] = a
        for n in seq:
            r = d[n](r)
            if not valida(r):
                return False
        if mapa:
            r = aplicar_mapa(r, mapa)
        if r != b:
            return False
    return True


def avaliar(tarefas, extras, orcamento, prof, rotulo) -> Set[str]:
    ok: Set[str] = set()
    t0 = time.time()
    for nome, tr, te in tarefas:
        seq, mapa, _g = resolver(tr, extras, orcamento, prof)
        if seq is not None and confere(seq, mapa, extras, te):
            ok.add(nome)
    print(f"  {rotulo:14s} {len(ok):3d}/{len(tarefas)}  "
          f"({len(ok)/len(tarefas):5.2%})  {len(PRIMITIVAS) + len(extras):2d} "
          f"primitivas  ({time.time()-t0:.0f}s)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--cenas", type=int, default=40)
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--prof", type=int, default=3)
    ap.add_argument("--semente", type=int, default=20260905)
    a = ap.parse_args()

    print("=" * 74)
    print("SOMA — o ganho do mundo sobrevive a busca inteira?")
    print("=" * 74)

    fis = extrair(True, a.cenas, a.semente)
    rui = extrair(False, a.cenas, a.semente + 1)
    k = min(len(fis), len(rui))
    fis, rui = fis[:k], rui[:k]
    print(f"\nS0 · {k} primitivas extraidas em cada braco")
    print(f"     dobra : {[n for n, _ in fis]}")
    print(f"     ruido : {len(rui)} permutacoes")
    print(f"\norcamento casado {a.orcamento} candidatos/tarefa, "
          f"profundidade {a.prof}")
    print(f"espaco a prof {a.prof}: base {len(PRIMITIVAS) ** a.prof}, "
          f"com extras {(len(PRIMITIVAS) + k) ** a.prof}\n")

    tarefas = carregar(a.pasta)
    base = avaliar(tarefas, [], a.orcamento, a.prof, "BASE")
    ruido = avaliar(tarefas, rui, a.orcamento, a.prof, "BASE+RUIDO")
    dobra = avaliar(tarefas, fis, a.orcamento, a.prof, "BASE+DOBRA")

    print(f"\nDOBRA - BASE  = {len(dobra) - len(base):+d}")
    print(f"DOBRA - RUIDO = {len(dobra) - len(ruido):+d}")
    print(f"RUIDO - BASE  = {len(ruido) - len(base):+d}   "
          f"<- o preco puro da diluicao")
    print(f"\nganhas pela DOBRA : {sorted(dobra - base)}")
    print(f"PERDIDAS pela DOBRA: {sorted(base - dobra)}")
    print(f"perdidas pelo RUIDO: {sorted(base - ruido)}")

    print("\n" + "=" * 74)
    if len(base) != 14:
        print(f"S0 · a BASE deu {len(base)}/400, nao 14/400. Este arquivo")
        print("     diverge do resto da sessao — ler o resto com ressalva.")
    if len(dobra) > len(base) and len(dobra) > len(ruido):
        print("S1 · o ganho SOBREVIVE a diluicao. Primeira vez na sessao que")
        print("     primitiva nao escrita por mim move o numero de referencia.")
    elif len(dobra) == len(ruido):
        print("S4 · dobra empata com permutacao: era vocabulario, nao mundo.")
    elif len(dobra) >= len(base):
        print("S2 · a diluicao empata o ganho. O efeito e real e nao paga o")
        print("     custo de busca que cobra.")
    else:
        print("S3 · a diluicao come o ganho inteiro. E a leitura desagradavel:")
        print("     prior bom somado a busca burra PIORA o resultado. O")
        print("     gargalo nao e o prior, e o mecanismo de busca.")


if __name__ == "__main__":
    main()

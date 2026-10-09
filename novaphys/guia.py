"""
novaphys.guia — a regua estrutural economiza orcamento de verdade?
====================================================================

POR QUE ESTE TESTE E O QUE DECIDE
-----------------------------------
`descricao.py` mediu AUC 0,860 para a regua estrutural contra 0,511 da regua
de pixel, e a ablacao mostrou que o sinal nao vem das dimensoes (0,853 sem
elas, estavel em 6 sementes).

Isso ainda nao vale nada, e a razao e simples: os negativos daquele teste eram
cadeias SORTEADAS. Na busca real o concorrente nao e ruido — sao os outros
14.423 candidatos da enumeracao, muitos deles primos proximos do certo.
Separar o certo do ruido e facil. Separar o certo dos vizinhos e o problema.

Aqui a regua e posta para trabalhar no unico lugar que importa: escolher a
ordem em que a busca gasta o orcamento.

TRES BRACOS, ORCAMENTO IDENTICO
---------------------------------
    LARGURA     o que ja temos. Enumera por profundidade crescente, cega.
    ESTRUTURA   melhor-primeiro. Expande o prefixo cuja saida esta
                estruturalmente mais perto do alvo.
    PIXEL       melhor-primeiro pela fracao de pixels coincidentes.

O braco PIXEL nao e enfeite: sem ele nao da para separar "a regua estrutural
ajuda" de "melhor-primeiro ajuda". Se os dois melhor-primeiro empatarem, o
ganho e do algoritmo e a regua nao acrescentou nada — e o achado da AUC nao
se converte em nada util.

A moeda e a mesma nos tres: numero de programas completos verificados contra
os pares de treino. Pontuar um filho custa uma verificacao e e cobrado.

O RISCO REAL DESTE DESENHO
----------------------------
Melhor-primeiro guloso e vulneravel a paisagem enganosa. Se o caminho certo
passar por um estado estruturalmente feio — e `0c9aba6e` prova que isso
acontece, o `pOU` tinha 0% de pixels e estava certo — o guia enterra o ramo
bom e gasta o orcamento em becos bonitos. Nesse caso ESTRUTURA perde para
LARGURA, e a leitura correta e que a regua mede semelhanca mas nao mede
PROMESSA. Sao coisas diferentes e eu nao sei ainda se a segunda existe.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO. Em tarefas cuja solucao tem 1 passo, os tres bracos tem
      que achar igual: com profundidade 1 nao ha ordem a escolher. Se
      divergirem ai, ha bug na contabilidade e o resto nao vale ler.

  P1  ESTRUTURA resolve >= 3 tarefas a mais que LARGURA, com orcamento
      identico, E supera PIXEL
      -> a regua converte em economia; guiar a busca volta a ser real

  P2  ESTRUTURA ~ PIXEL, ambos > LARGURA
      -> o ganho e do melhor-primeiro, nao da regua. A AUC nao se converteu

  P3  ESTRUTURA <= LARGURA
      -> guloso se perde na paisagem enganosa; semelhanca nao e promessa

MINHA PREVISAO, escrita antes de rodar: espero P2 ou P3, e mais P3 que P2.
O motivo e concreto e vem dos nossos proprios dados: 33 das 51 solucoes tem
UM passo, onde nao ha ordem a escolher; e entre as que tem mais passos,
`0c9aba6e` mostra que o estado intermediario correto pode ser pessimo pela
regua. Guloso morre exatamente nesse caso.

Se P1 aparecer, e o primeiro ganho de mecanismo da sessao inteira, e eu vou
querer ver o controle PIXEL bem longe antes de acreditar.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import time
from typing import Dict, List, Optional, Sequence, Tuple

from novaphys.arc import (Grid, DICIONARIO, PRIMITIVAS, ajustar_mapa,
                          aplicar_mapa, carregar, dims, valida)
from novaphys.descricao import semelhanca_estrutural, semelhanca_pixel
from novaphys.eter import PASTA_AVAL

NOMES = [n for n, _ in PRIMITIVAS]


def _aceita(saidas: Sequence[Grid], treino) -> Optional[Dict[int, int]]:
    """None se nao resolve; {} se resolve direto; mapa se resolve recolorindo."""
    if all(r == b for r, (_, b) in zip(saidas, treino)):
        return {}
    m = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
    if m is not None and all(aplicar_mapa(r, m) == b
                             for r, (_, b) in zip(saidas, treino)):
        return m
    return None


def por_largura(treino, orcamento: int, prof_max: int = 3
                ) -> Tuple[Optional[List[str]], int]:
    gasto = 0
    for prof in range(0, prof_max + 1):
        for seq in itertools.product(NOMES, repeat=prof):
            if gasto >= orcamento:
                return None, gasto
            gasto += 1
            saidas, ok = [], True
            for a, _b in treino:
                r: Optional[Grid] = a
                for n in seq:
                    r = DICIONARIO[n](r)
                    if not valida(r):
                        ok = False
                        break
                if not ok:
                    break
                saidas.append(r)
            if ok and _aceita(saidas, treino) is not None:
                return list(seq), gasto
    return None, gasto


def melhor_primeiro(treino, orcamento: int, regua: str,
                    prof_max: int = 3) -> Tuple[Optional[List[str]], int]:
    """
    Expande sempre o prefixo mais promissor pela `regua`. Cada no expandido
    gera |NOMES| filhos, e CADA filho custa uma verificacao — cobrada.

    O empate e desfeito por ordem de insercao, para que o braco nao dependa
    de ordenacao instavel de tuplas.
    """
    alvo = treino[0][1]
    sim = semelhanca_estrutural if regua == "estrutura" else semelhanca_pixel
    gasto = 0
    contador = itertools.count()

    saidas0 = [a for a, _ in treino]
    gasto += 1
    if _aceita(saidas0, treino) is not None:
        return [], gasto

    fila: List[Tuple[float, int, List[str], List[Grid]]] = [
        (-sim(saidas0[0], alvo), next(contador), [], saidas0)]

    while fila and gasto < orcamento:
        _s, _c, ops, saidas = heapq.heappop(fila)
        if len(ops) >= prof_max:
            continue
        for n in NOMES:
            if gasto >= orcamento:
                break
            gasto += 1
            novas, ok = [], True
            for r in saidas:
                r2 = DICIONARIO[n](r)
                if not valida(r2):
                    ok = False
                    break
                novas.append(r2)
            if not ok:
                continue
            if _aceita(novas, treino) is not None:
                return ops + [n], gasto
            heapq.heappush(fila, (-sim(novas[0], alvo), next(contador),
                                  ops + [n], novas))
    return None, gasto


def confere(ops: Sequence[str], treino, teste) -> bool:
    """Reaplica com o mesmo mapa de cor deduzido do treino."""
    saidas = []
    for a, _b in treino:
        r: Optional[Grid] = a
        for n in ops:
            r = DICIONARIO[n](r)
            if not valida(r):
                return False
        saidas.append(r)
    m = _aceita(saidas, treino)
    if m is None:
        return False
    for a, b in teste:
        r = a
        for n in ops:
            r = DICIONARIO[n](r)
            if not valida(r):
                return False
        if m and aplicar_mapa(r, m) != b:
            return False
        if not m and r != b:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--orcamento", type=int, default=14424)
    a = ap.parse_args()

    print("=" * 76)
    print("GUIA — a regua estrutural economiza orcamento de verdade?")
    print("=" * 76)
    print(f"orcamento IDENTICO nos 3 bracos: {a.orcamento} verificacoes\n")

    tarefas = carregar(a.pasta)
    res: Dict[str, set] = {"LARGURA": set(), "ESTRUTURA": set(),
                           "PIXEL": set()}
    custo: Dict[str, List[int]] = {k: [] for k in res}
    prof1: Dict[str, set] = {k: set() for k in res}

    t0 = time.time()
    for nome, tr, te in tarefas:
        for br in res:
            if br == "LARGURA":
                ops, g = por_largura(tr, a.orcamento)
            else:
                ops, g = melhor_primeiro(tr, a.orcamento, br.lower())
            if ops is not None and confere(ops, tr, te):
                res[br].add(nome)
                custo[br].append(g)
                if len(ops) <= 1:
                    prof1[br].add(nome)

    dt = time.time() - t0
    print(f"{'braco':12s} {'resolve':>8} {'%':>7} {'custo mediano':>15}")
    print("-" * 46)
    for br in ("LARGURA", "ESTRUTURA", "PIXEL"):
        c = sorted(custo[br])
        med = c[len(c) // 2] if c else 0
        print(f"{br:12s} {len(res[br]):>8} {len(res[br])/len(tarefas):>7.2%} "
              f"{med:>15,}")
    print("-" * 46)
    print(f"({dt:.0f}s)\n")

    # P0
    p1 = {br: prof1[br] for br in res}
    iguais = p1["LARGURA"] == p1["ESTRUTURA"] == p1["PIXEL"]
    print(f"P0 · solucoes de <=1 passo por braco: "
          f"{len(p1['LARGURA'])}, {len(p1['ESTRUTURA'])}, {len(p1['PIXEL'])}")
    if iguais:
        print("     P0 · passa — sem ordem a escolher, os tres coincidem")
    else:
        print("     P0 · ATENCAO — divergem onde nao deveriam. Ha diferenca")
        print("     de contabilidade entre os bracos; ler o resto com reserva.")
        for br in res:
            print(f"       so em {br}: "
                  f"{sorted(p1[br] - set().union(*(p1[x] for x in res if x != br)))}")

    L, E, P = (len(res["LARGURA"]), len(res["ESTRUTURA"]), len(res["PIXEL"]))
    print(f"\nESTRUTURA - LARGURA = {E - L:+d}")
    print(f"ESTRUTURA - PIXEL   = {E - P:+d}")
    print(f"\nso ESTRUTURA resolve: {sorted(res['ESTRUTURA'] - res['LARGURA'])}")
    print(f"so LARGURA resolve  : {sorted(res['LARGURA'] - res['ESTRUTURA'])}")

    print("\n" + "=" * 76)
    if E - L >= 3 and E > P:
        print("P1 · a regua converteu em economia. Primeiro ganho de mecanismo.")
        print("     Conferir o controle PIXEL antes de acreditar.")
    elif abs(E - P) < 3 and E > L:
        print("P2 · o ganho e do melhor-primeiro, nao da regua estrutural.")
    else:
        print("P3 · guloso se perde. Semelhanca nao e promessa — a regua diz")
        print("     quao parecido o estado esta, nao se ele leva a algum lugar.")


if __name__ == "__main__":
    main()

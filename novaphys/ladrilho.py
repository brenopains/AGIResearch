"""
novaphys.ladrilho — a lente de LADRILHO, tirada da introspeccao humana
========================================================================

DE ONDE VEM, E A PROVENIENCIA IMPORTA
---------------------------------------
Nao saiu de mim. Saiu de 26 minutos de gravacao do usuario resolvendo tarefas
do ARC em voz alta, no trecho em que ele resolveu em dois minutos uma tarefa
que NENHUM motor nosso toca (17:00-17:58):

    "eu vou dividir ele em 3 blocos... eu vou colocar os 3 em caixas, 3
     quadrados, SO PRA EU CONSEGUIR ORGANIZAR EFICIENTEMENTE A MINHA CABECA...
     depois eu vou tirar os quadrados e vou juntar os 3... e ai eu vou
     empilhar eles."

Tres coisas ai nao existem em motor nenhum nosso: agrupamento temporario que
nao aparece na saida, decomposicao em blocos periodicos, e empilhamento. As
oito lentes de `segmenta.py` enxergam componente conexa, cor, linha, coluna e
separador EXPLICITO. Nenhuma enxerga LADRILHO — repeticao periodica de um
bloco sem nada marcando a divisao.

E `00576224`, a tarefa em questao, e da familia: entrada 2x2, saida 6x6, que e
a entrada ladrilhada 3x3 com espelhamento em algumas posicoes.

O MECANISMO
-------------
Mesmo principio que ja rendeu tudo no projeto: LER, nao procurar.

  1. As dimensoes da saida sao multiplo inteiro das da entrada? Se sim, a saida
     e uma grade de a x b ladrilhos.
  2. Para cada posicao de ladrilho, LE-SE qual das oito simetrias do quadrado
     (D4) leva a entrada naquele ladrilho. Se nenhuma leva, a tarefa nao e
     desta familia e o motor devolve None — nao chuta.
  3. GENERALIZA: essa simetria e funcao de que? Da linha do ladrilho, da
     coluna, da paridade da soma, ou e constante? Testa-se a lista fixa de
     preditores e fica o primeiro consistente.

Nenhuma enumeracao de operadores. A transformacao de cada ladrilho e lida da
diferenca, e a unica busca e sobre QUAL PREDITOR — e sao seis.

O SISTEMA IMUNOLOGICO, E UMA TENSAO QUE PRECISA SER MEDIDA
------------------------------------------------------------
`ockham.py` mediu que validacao cruzada deixe-um-de-fora leva a precisao de
4,7% para 100%. Mas ela exige >= 3 pares de treino, e `00576224` tem DOIS.

Ou seja: o portao que salvou o motor de 80 familias REJEITA, por construcao, a
tarefa que motivou este arquivo.

Isso nao e contradicao — e calibragem. O portao foi ajustado para um espaco de
80 familias contra 2-4 exemplos. Aqui o espaco e 8 simetrias x 6 preditores =
48, e boa parte e mutuamente exclusiva. O risco de ajuste ao acaso e menor.

Entao mede-se DOS DOIS JEITOS e reporta-se a precisao de cada um, em vez de eu
escolher qual conta. Se sem portao a precisao continuar alta, o portao estava
severo demais para este espaco. Se despencar, o portao estava certo e a tarefa
de 2 pares e inalcancavel por honestidade.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  L0  PRE-CONDICAO. Sem o portao, `00576224` tem que ser resolvida. Se a lente
      nao pega a tarefa que a motivou, ela esta errada.

  L1  >= 4 tarefas novas, com precisao >= 70%
  L2  1 a 3 novas
  L3  zero

  L4  O QUE MAIS INTERESSA: a precisao SEM portao. Se ficar >= 70%, o
      resultado e que validacao cruzada deve ser calibrada pelo TAMANHO DO
      ESPACO, e nao aplicada uniformemente — o que muda como usar o
      imunologico em todo o resto do projeto.

MINHA CALIBRAGEM: quatorze previsoes na sessao, CINCO certas, e as nove
erradas todas otimistas.

PREVISAO: L2, 1 a 3 tarefas, e L4 positivo. Aposto que a precisao sem portao
fica alta (>= 80%) porque o espaco e pequeno e as simetrias sao mutuamente
exclusivas — duas simetrias diferentes quase nunca produzem o mesmo ladrilho.
E aposto que o portao rejeita `00576224` e mais uma ou duas por falta de
pares, o que tornaria o numero COM portao menor que o SEM.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.eter import PASTA_AVAL


# ══════════════════════════════════════════════ as oito simetrias do quadrado
def _rot(g: Grid) -> Grid:
    return tuple(zip(*g[::-1]))                # type: ignore[return-value]


def _esp(g: Grid) -> Grid:
    return tuple(tuple(l[::-1]) for l in g)


def d4(g: Grid) -> List[Tuple[str, Grid]]:
    fora, v = [], g
    for k in range(4):
        fora.append((f"r{k}", v))
        fora.append((f"r{k}m", _esp(v)))
        v = _rot(v)
    return fora


def realizar(g: Grid, tag: str) -> Optional[Grid]:
    for nome, v in d4(g):
        if nome == tag:
            return v
    return None


# ══════════════════════════════════════════════ os preditores do ladrilho
PREDITORES: List[Tuple[str, Callable]] = [
    ("const", lambda i, j, a, b: 0),
    ("linha", lambda i, j, a, b: i),
    ("coluna", lambda i, j, a, b: j),
    ("par_linha", lambda i, j, a, b: i % 2),
    ("par_coluna", lambda i, j, a, b: j % 2),
    ("par_soma", lambda i, j, a, b: (i + j) % 2),
]


def ladrilhos(a: Grid, b: Grid):
    """
    A saida e uma grade de ladrilhos da entrada? Se sim, devolve (na, nb) e a
    lista (i, j, tag) com a simetria que leva a entrada naquele ladrilho.
    Devolve None se as dimensoes nao baterem ou se algum ladrilho nao for
    nenhuma das oito simetrias — nada de chute.
    """
    h, w = dims(a)
    H, W = dims(b)
    if H % h or W % w:
        return None
    na, nb = H // h, W // w
    if na * nb < 2:
        return None
    variantes = d4(a)
    fora = []
    for i in range(na):
        for j in range(nb):
            bloco = tuple(tuple(b[i * h + y][j * w + x] for x in range(w))
                          for y in range(h))
            tag = None
            for nome, v in variantes:
                if v == bloco:
                    tag = nome
                    break
            if tag is None:
                return None
            fora.append((i, j, tag))
    return na, nb, fora


def deduzir(treino):
    """Le a simetria de cada ladrilho e resolve QUAL preditor a determina."""
    forma: Set[Tuple[int, int]] = set()
    obs: Dict[str, Dict[Any, Set[str]]] = {p: {} for p, _f in PREDITORES}
    for a, b in treino:
        r = ladrilhos(a, b)
        if r is None:
            return None
        na, nb, lst = r
        forma.add((na, nb))
        for (i, j, tag) in lst:
            for p, f in PREDITORES:
                obs[p].setdefault(f(i, j, na, nb), set()).add(tag)
    if len(forma) != 1:
        return None                       # a forma do ladrilho tem que ser fixa
    na, nb = next(iter(forma))
    for p, _f in PREDITORES:
        if any(len(v) > 1 for v in obs[p].values()):
            continue
        return (na, nb, p, {k: next(iter(v)) for k, v in obs[p].items()})
    return None


def aplicar(regra, g: Grid) -> Optional[Grid]:
    na, nb, p, mapa = regra
    f = dict(PREDITORES)[p]
    h, w = dims(g)
    if h * na > 30 or w * nb > 30:
        return None
    fora = [[0] * (w * nb) for _ in range(h * na)]
    for i in range(na):
        for j in range(nb):
            k = f(i, j, na, nb)
            if k not in mapa:
                return None
            bloco = realizar(g, mapa[k])
            if bloco is None or dims(bloco) != (h, w):
                return None
            for y in range(h):
                for x in range(w):
                    fora[i * h + y][j * w + x] = bloco[y][x]
    return tuple(tuple(l) for l in fora)


def imune(treino) -> bool:
    """Deixe-um-de-fora nos pares de TREINO. Exige >= 3 pares."""
    if len(treino) < 3:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir(resto)
        if r is None:
            return False
        try:
            if aplicar(r, treino[i][0]) != treino[i][1]:
                return False
        except Exception:
            return False
    return True


def _uma(args):
    nome, tr, te = args
    try:
        r = deduzir(tr)
        if r is None:
            return nome, False, False, False, len(tr)
        if any(aplicar(r, a) != b for a, b in tr):
            return nome, False, False, False, len(tr)
        ok = all(aplicar(r, x) == y for x, y in te)
        return nome, True, imune(tr), ok, len(tr)
    except Exception:
        return nome, False, False, False, len(tr)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 78)
    print("LADRILHO — a lente vinda da introspeccao humana")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=8)
    ajust = [n for n, aj, _i, _o, _k in res if aj]
    sem = {n for n, _aj, _i, o, _k in res if o}
    com = {n for n, _aj, i, o, _k in res if i and o}
    imunes = [n for n, _aj, i, _o, _k in res if i]
    curtos = [n for n, aj, _i, _o, k in res if aj and k < 3]

    print(f"\n  ajustou o TREINO       : {len(ajust):3d}/400")
    print(f"  sobreviveu ao IMUNE    : {len(imunes):3d}/400")
    print(f"\n  SEM portao  passou no teste: {len(sem):3d}/400   "
          f"precisao {len(sem)/max(len(ajust),1):.1%}")
    print(f"  COM portao  passou no teste: {len(com):3d}/400   "
          f"precisao {len(com)/max(len(imunes),1):.1%}")
    print(f"\n  ({time.time()-t0:.0f}s)  tarefas com < 3 pares que ajustaram: "
          f"{len(curtos)}")
    print(f"\nL0 · `00576224` resolvida sem portao: {'00576224' in sem}")
    print(f"     com portao: {'00576224' in com}")
    print(f"\n  SEM portao: {sorted(sem)}")
    print(f"  COM portao: {sorted(com)}")

    p_sem = len(sem) / max(len(ajust), 1)
    print("\n" + "=" * 78)
    if "00576224" not in sem:
        print("L0 FALHOU: a lente nao pega a tarefa que a motivou.")
    if len(sem) >= 4 and p_sem >= 0.70:
        print("L1 · a lente de ladrilho entrega, e com precisao.")
    elif len(sem) >= 1:
        print("L2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("L3 · a familia de ladrilho nao esta no ARC de avaliacao.")
    if p_sem >= 0.70:
        print(f"\nL4 · precisao SEM portao = {p_sem:.1%}. A validacao cruzada")
        print("     deve ser calibrada pelo TAMANHO DO ESPACO, nao aplicada")
        print("     uniformemente: aqui sao 48 hipoteses e o ajuste ao acaso")
        print("     e raro, ao contrario das 80 familias de `segmenta`.")
    else:
        print(f"\nL4 · precisao SEM portao = {p_sem:.1%}. O portao estava certo")
        print("     tambem aqui; espaco pequeno nao dispensa validacao.")


if __name__ == "__main__":
    main()

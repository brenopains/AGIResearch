"""
novaphys.emissao — DESENHAR, em vez de transformar
====================================================

O QUE O CENSO MEDIU
---------------------
`inventario.py` classificou as 373 tarefas nao resolvidas por assinatura
estrutural, exigindo que a assinatura valesse em TODOS os pares de treino:

    formas_novas    199 nao resolvidas   14 resolvidas
    mais_objetos    149                  15
    formas_iguais    40                   7

**53% das nao resolvidas produzem formas que nao existiam na entrada.**

E onde as formas sao PRESERVADAS resolvemos 15% (7 de 47); onde ha formas
NOVAS resolvemos 6,6% (14 de 213).

POR QUE ISSO CONDENA TODO O MOTOR ATUAL
-----------------------------------------
Nosso motor de objetos e TRANSFORMACIONAL: pareia objeto de entrada com objeto
de saida e le o delta. A funcao `deltas` devolve None quando um objeto de saida
nao tem origem plausivel — esta escrito assim de proposito, e eu registrei como
"criacao nao e tratada aqui".

Criacao e a maioria. Tudo que construimos hoje move, pinta, apaga, espelha,
escala, recorta ou ladrilha: operacoes sobre o que JA ESTA LA. Nada desenha.

O MECANISMO: LER O QUE MUDOU, NAO PROCURAR O OPERADOR
-------------------------------------------------------
Mesmo principio que rendeu 3.000x no eter: nao se enumera desenho. Olha-se as
CELULAS QUE MUDARAM entre entrada e saida e pergunta-se se elas sao exatamente
o que uma familia de emissao produziria, com os parametros LIDOS dos dados.

Quatro familias, cada uma com parametros deduzidos:

    RAIO       de cada celula de cor C, emitir na direcao D ate bater em algo
               ou na borda. Parametros lidos: C, D, cor da pintura.
    PREENCHER  inundar as regioes de fundo CERCADAS (nao alcancaveis a partir
               da borda). Parametro lido: cor da pintura.
    CONECTAR   entre pares de celulas de mesma cor na mesma linha ou coluna,
               com so fundo no meio, preencher o segmento. Parametro: cor.
    CONTORNO   as celulas de fundo adjacentes a cada objeto. Parametro: cor.

A cor da pintura e lida do proprio diff: ou e uma cor fixa, ou e a cor da
fonte. Nada e sorteado.

VERIFICACAO
-------------
Igualdade EXATA na grade de teste, como sempre. Mais o portao imunologico
(validacao cruzada deixe-um-de-fora) onde houver >= 3 pares — com o relatorio
mostrando os dois numeros, porque `ladrilho.py` mediu que em espaco pequeno o
portao custa verdadeiro positivo.

O espaco aqui e pequeno: 4 familias x ~10 cores x 8 direcoes ~ 300 hipoteses,
e elas sao mutuamente exclusivas (o conjunto de celulas que cada uma produz e
muito diferente). Pela calibragem de `ladrilho.py`, ajustar deve ser evidencia
forte.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  E0  PRE-CONDICAO. Alguma familia tem que disparar em alguma tarefa. Se
      nenhuma dispara, a implementacao esta quebrada e nao ha o que medir.

  E1  >= 8 tarefas novas
      -> emissao era a representacao que faltava, e o censo apontou certo
  E2  3 a 7
      -> ajuda, na ordem das outras mudancas de representacao (objetos: +3,
         ladrilho: +3)
  E3  <= 2
      -> as 199 com formas novas precisam de emissao mais rica que estas
         quatro familias, ou de outra coisa

MINHA CALIBRAGEM: vinte previsoes na sessao, SETE certas, as treze erradas
TODAS otimistas. Aplico a correcao.

PREVISAO: E2, 3 a 7. O argumento a favor e forte — e mudanca de
REPRESENTACAO, a unica categoria que moveu o numero hoje (duas vezes em duas,
+3 cada). O argumento contra e que quatro familias sao poucas para 199 tarefas,
e que as versoes reais no ARC costumam ter condicao extra (raio so de certas
celulas, preenchimento so de certas regioes) que estas versoes puras nao
capturam.

Aposto no mesmo patamar das outras duas mudancas de representacao: +3.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.eter import PASTA_AVAL

DIRS = {"baixo": (1, 0), "cima": (-1, 0), "dir": (0, 1), "esq": (0, -1),
        "se": (1, 1), "sd": (1, -1), "ne": (-1, 1), "nd": (-1, -1)}


def fundo(g: Grid) -> int:
    return Counter(v for l in g for v in l).most_common(1)[0][0]


# ══════════════════════════════════════════════ as quatro familias
def emitir_raio(g: Grid, bg: int, fonte: int, d: str,
                cor: Optional[int]) -> Grid:
    """De cada celula de `fonte`, anda em `d` pintando fundo ate bater."""
    h, w = dims(g)
    dy, dx = DIRS[d]
    fora = [list(l) for l in g]
    for i in range(h):
        for j in range(w):
            if g[i][j] != fonte:
                continue
            y, x = i + dy, j + dx
            while 0 <= y < h and 0 <= x < w and g[y][x] == bg:
                fora[y][x] = fonte if cor is None else cor
                y, x = y + dy, x + dx
    return tuple(tuple(l) for l in fora)


def emitir_preencher(g: Grid, bg: int, _f, _d, cor: Optional[int]) -> Grid:
    """Inunda o fundo NAO alcancavel a partir da borda: o interior cercado."""
    h, w = dims(g)
    vis = [[False] * w for _ in range(h)]
    fila = deque()
    for i in range(h):
        for j in (0, w - 1):
            if g[i][j] == bg and not vis[i][j]:
                vis[i][j] = True
                fila.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if g[i][j] == bg and not vis[i][j]:
                vis[i][j] = True
                fila.append((i, j))
    while fila:
        y, x = fila.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                    and g[ny][nx] == bg):
                vis[ny][nx] = True
                fila.append((ny, nx))
    fora = [list(l) for l in g]
    for i in range(h):
        for j in range(w):
            if g[i][j] == bg and not vis[i][j]:
                fora[i][j] = 1 if cor is None else cor
    return tuple(tuple(l) for l in fora)


def emitir_conectar(g: Grid, bg: int, _f, _d, cor: Optional[int]) -> Grid:
    """Liga pares de mesma cor na mesma linha/coluna com so fundo no meio."""
    h, w = dims(g)
    fora = [list(l) for l in g]
    for i in range(h):
        ocup = [j for j in range(w) if g[i][j] != bg]
        for a, b in zip(ocup, ocup[1:]):
            if g[i][a] == g[i][b] and b - a > 1:
                if all(g[i][k] == bg for k in range(a + 1, b)):
                    for k in range(a + 1, b):
                        fora[i][k] = g[i][a] if cor is None else cor
    for j in range(w):
        ocup = [i for i in range(h) if g[i][j] != bg]
        for a, b in zip(ocup, ocup[1:]):
            if g[a][j] == g[b][j] and b - a > 1:
                if all(g[k][j] == bg for k in range(a + 1, b)):
                    for k in range(a + 1, b):
                        fora[k][j] = g[a][j] if cor is None else cor
    return tuple(tuple(l) for l in fora)


def emitir_contorno(g: Grid, bg: int, _f, _d, cor: Optional[int]) -> Grid:
    """Pinta as celulas de fundo 4-adjacentes a qualquer objeto."""
    h, w = dims(g)
    fora = [list(l) for l in g]
    for i in range(h):
        for j in range(w):
            if g[i][j] != bg:
                continue
            viz = [g[i + dy][j + dx]
                   for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                   if 0 <= i + dy < h and 0 <= j + dx < w]
            n = [v for v in viz if v != bg]
            if n:
                fora[i][j] = n[0] if cor is None else cor
    return tuple(tuple(l) for l in fora)


FAMILIAS = [
    ("raio", emitir_raio, True),
    ("preencher", emitir_preencher, False),
    ("conectar", emitir_conectar, False),
    ("contorno", emitir_contorno, False),
]


def deduzir(treino):
    """
    Le as celulas que mudaram e testa se alguma familia as reproduz EXATAMENTE,
    com parametros lidos dos dados. Nada e sorteado.
    """
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    bg = fundo(treino[0][0])
    # cores candidatas para a pintura: as que aparecem no diff, mais "a propria
    # cor da fonte" (representada por None)
    mudou: Set[int] = set()
    for a, b in treino:
        for la, lb in zip(a, b):
            for x, y in zip(la, lb):
                if x != y:
                    mudou.add(y)
    cores: List[Optional[int]] = [None] + sorted(mudou)
    fontes = sorted({c for a, _b in treino for l in a for c in l if c != bg})

    for nome, fn, usa_fonte in FAMILIAS:
        combos = ([(f, d) for f in fontes for d in DIRS] if usa_fonte
                  else [(None, None)])
        for f, d in combos:
            for cor in cores:
                try:
                    if all(fn(a, bg, f, d, cor) == b for a, b in treino):
                        return (nome, f, d, cor, bg)
                except Exception:
                    continue
    return None


def aplicar(regra, g: Grid) -> Optional[Grid]:
    nome, f, d, cor, _bg = regra
    fn = dict((n, x) for n, x, _u in FAMILIAS)[nome]
    try:
        return fn(g, fundo(g), f, d, cor)
    except Exception:
        return None


def imune(treino) -> bool:
    if len(treino) < 3:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir(resto)
        if r is None:
            return False
        if aplicar(r, treino[i][0]) != treino[i][1]:
            return False
    return True


def _uma(args):
    nome, tr, te = args
    try:
        r = deduzir(tr)
        if r is None:
            return nome, False, False, False, None
        ok = all(aplicar(r, x) == y for x, y in te)
        return nome, True, imune(tr), ok, r[0]
    except Exception:
        return nome, False, False, False, None


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 78)
    print("EMISSAO — desenhar, em vez de transformar")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [(n, f) for n, aj, _i, _o, f in res if aj]
    imunes = [n for n, _aj, i, _o, _f in res if i]
    sem = {n: f for n, _aj, _i, o, f in res if o}
    com = {n: f for n, _aj, i, o, f in res if i and o}

    print(f"\n  ajustou o TREINO       : {len(ajust):3d}/400  "
          f"({time.time()-t0:.0f}s)")
    print(f"  sobreviveu ao IMUNE    : {len(imunes):3d}/400")
    print(f"\n  SEM portao passou teste: {len(sem):3d}/400   "
          f"precisao {len(sem)/max(len(ajust),1):.1%}")
    print(f"  COM portao passou teste: {len(com):3d}/400   "
          f"precisao {len(com)/max(len(imunes),1):.1%}")
    print(f"\n  familias que resolveram: "
          f"{dict(Counter(sem.values()).most_common())}")
    print(f"  familias que ajustaram : "
          f"{dict(Counter(f for _n, f in ajust).most_common())}")
    print(f"\n  tarefas SEM portao: {sorted(sem)}")

    print("\n" + "=" * 78)
    if not ajust:
        print("E0 FALHOU: nenhuma familia disparou. Implementacao quebrada.")
        return
    if len(sem) >= 8:
        print("E1 · emissao era a representacao que faltava, e o censo")
        print("     apontou certo.")
    elif len(sem) >= 3:
        print("E2 · ajuda, no mesmo patamar das outras mudancas de")
        print("     representacao desta sessao (objetos +3, ladrilho +3).")
    else:
        print("E3 · as 199 com formas novas precisam de emissao mais rica que")
        print("     estas quatro familias puras.")


if __name__ == "__main__":
    main()

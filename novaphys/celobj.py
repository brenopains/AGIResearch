"""
novaphys.celobj — testes RELACIONAIS entre celula e objeto
============================================================

O ALVO, MEDIDO E NOMEADO
--------------------------
`censofino.py` mediu 146 tarefas nao resolvidas de mesma dimensao que produzem
formas novas. O perfil mais comum, com 37 tarefas:

    novas_uma_cor + preserva

Ou seja: acrescente celulas de UMA cor e nao toque em mais nada. Isso reduz o
problema a uma pergunta binaria:

    QUAIS celulas de fundo viram a cor C?

E essa e exatamente a forma do motor mais antigo do projeto:

    celula(i,j) = TESTE ? cor : entrada[i][j]

O que falta nao e a maquina — e o VOCABULARIO DE TESTES.

O QUE JA FOI REFUTADO, PARA NAO REPETIR
-----------------------------------------
    testes por CELULA ISOLADA (cor, borda, paridade, vizinho igual)
        -> 3/400 (`eter_inferencia`)
    propriedades que comparam OBJETO com OBJETO (forma unica, dentro_de,
    toca_outro, n_vizinhos...)
        -> ZERO, e nem falso positivo novo (`relacional.py`): induzem as
           mesmas particoes que as intrinsecas

Isto e uma terceira coisa: relacao CELULA <-> OBJETO. Nao "este objeto e
diferente dos outros", e sim "esta celula esta entre dois objetos", "esta
celula esta na linha de um objeto", "esta celula esta dentro da caixa de
alguem". E o que decide ONDE PINTAR.

OS TESTES
-----------
    entre_dois       na reta entre dois objetos, so com fundo no meio
    linha_de_obj     a linha da celula contem algum objeto
    coluna_de_obj    a coluna contem
    cruz             linha E coluna contem
    dentro_caixa     dentro da caixa envolvente de algum objeto
    dentro_maior     dentro da caixa do maior objeto
    adjacente4/8     encosta em objeto
    dist_k           a distancia exata k (1,2,3) de algum objeto
    cercado          fundo nao alcancavel a partir da borda
    linha_cor_c      a linha contem a cor c
    coluna_cor_c     a coluna contem a cor c

A COR e lida do diff: constante, ou "a cor do objeto mais proximo", ou "a cor
do objeto na mesma linha/coluna". Nada sorteado.

O METODO E O DE SEMPRE: LER, NAO PROCURAR
-------------------------------------------
Le-se do diff o CONJUNTO EXATO de celulas que viraram a cor C. Depois procura-se
qual teste produz exatamente esse conjunto. Nao se enumera programa: resolve-se
qual predicado tem aquele conjunto como extensao.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO. Algum teste tem que disparar em alguma tarefa.

  P1  >= 8 inedidas -> o vocabulario relacional celula-objeto era o que
                       faltava, e o perfil de 37 tarefas era o alvo certo
  P2  3 a 7         -> ajuda, no patamar das cinco mudancas de representacao
  P3  <= 2          -> a colocacao nas 37 e idiossincratica mesmo, e nem
                       relacao celula-objeto a captura

MINHA CALIBRAGEM: vinte e sete previsoes, DEZ certas, as dezessete erradas
todas otimistas. E o padrao de vinte tentativas: o que vem de MEDIDA entrega,
o que vem de palpite meu da zero. Este veio de medida — o perfil de 37 foi
contado, nao imaginado — mas os TESTES especificos sao escolha minha, e essa
metade e palpite.

PREVISAO: P2 na faixa baixa, +2 a +4. O alvo esta medido, mas os doze testes
sao meus, e o censo fino ja mostrou que os mecanismos nomeados que eu conhecia
(encher, simetrizar, colar, linha, carimbar) somam so 16 das 146. Se a
colocacao nas 37 fosse um dos meus testes conhecidos, o censo teria visto.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Callable, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo, objetos
from novaphys.eter import PASTA_AVAL


def _cercado(g: Grid, bg: int) -> Set[Tuple[int, int]]:
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
    return {(i, j) for i in range(h) for j in range(w)
            if g[i][j] == bg and not vis[i][j]}


def conjuntos(g: Grid) -> Dict[str, Set[Tuple[int, int]]]:
    """
    Para cada teste, o CONJUNTO de celulas de fundo que ele marca. Calculado
    uma vez por grade; a deducao so compara conjuntos.
    """
    bg = fundo(g)
    h, w = dims(g)
    livre = {(i, j) for i in range(h) for j in range(w) if g[i][j] == bg}
    ocup = {(i, j) for i in range(h) for j in range(w) if g[i][j] != bg}
    linhas = {i for i, _j in ocup}
    cols = {j for _i, j in ocup}
    objs = objetos(g, bg)
    fora: Dict[str, Set[Tuple[int, int]]] = {}

    fora["linha_de_obj"] = {p for p in livre if p[0] in linhas}
    fora["coluna_de_obj"] = {p for p in livre if p[1] in cols}
    fora["cruz"] = {p for p in livre if p[0] in linhas and p[1] in cols}
    fora["cercado"] = _cercado(g, bg)

    # entre dois ocupados na mesma linha/coluna, so com fundo no meio
    entre = set()
    for i in range(h):
        oc = sorted(j for j in range(w) if (i, j) in ocup)
        for a, b in zip(oc, oc[1:]):
            if b - a > 1:
                entre |= {(i, k) for k in range(a + 1, b)}
    for j in range(w):
        oc = sorted(i for i in range(h) if (i, j) in ocup)
        for a, b in zip(oc, oc[1:]):
            if b - a > 1:
                entre |= {(k, j) for k in range(a + 1, b)}
    fora["entre_dois"] = entre & livre

    # entre dois da MESMA COR
    entre_c = set()
    for i in range(h):
        oc = sorted(j for j in range(w) if (i, j) in ocup)
        for a, b in zip(oc, oc[1:]):
            if b - a > 1 and g[i][a] == g[i][b]:
                entre_c |= {(i, k) for k in range(a + 1, b)}
    for j in range(w):
        oc = sorted(i for i in range(h) if (i, j) in ocup)
        for a, b in zip(oc, oc[1:]):
            if b - a > 1 and g[a][j] == g[b][j]:
                entre_c |= {(k, j) for k in range(a + 1, b)}
    fora["entre_mesma_cor"] = entre_c & livre

    if objs:
        caixas = []
        for cel, _k in objs:
            ys = [y for y, _x in cel]
            xs = [x for _y, x in cel]
            caixas.append((min(ys), min(xs), max(ys), max(xs)))
        fora["dentro_caixa"] = {p for p in livre if any(
            y0 <= p[0] <= y1 and x0 <= p[1] <= x1
            for y0, x0, y1, x1 in caixas)}
        m = max(len(c) for c, _k in objs)
        gy0, gx0, gy1, gx1 = next(
            cx for cx, (c, _k) in zip(caixas, objs) if len(c) == m)
        fora["dentro_maior"] = {p for p in livre
                                if gy0 <= p[0] <= gy1 and gx0 <= p[1] <= gx1}

    for k in (1, 2, 3):
        fora[f"dist_{k}"] = {
            p for p in livre
            if min((abs(p[0] - y) + abs(p[1] - x) for y, x in ocup),
                   default=99) == k}
    fora["adjacente8"] = {
        p for p in livre
        if any((p[0] + dy, p[1] + dx) in ocup
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy, dx) != (0, 0))}

    for c in {v for l in g for v in l if v != bg}:
        lc = {i for i in range(h) if c in g[i]}
        cc = {j for j in range(w) if any(g[i][j] == c for i in range(h))}
        fora[f"linha_cor{c}"] = {p for p in livre if p[0] in lc}
        fora[f"coluna_cor{c}"] = {p for p in livre if p[1] in cc}
    return {k: v for k, v in fora.items() if v}


def deduzir(treino):
    """Le o conjunto pintado do diff e resolve QUAL teste tem essa extensao."""
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    alvos, cores, cs = [], set(), []
    for a, b in treino:
        bg = fundo(a)
        h, w = dims(a)
        pint = {(i, j) for i in range(h) for j in range(w)
                if a[i][j] == bg and b[i][j] != bg}
        # PRESERVA: nada mais pode mudar. Se mudar, nao e desta familia.
        if any(a[i][j] != bg and a[i][j] != b[i][j]
               for i in range(h) for j in range(w)):
            return None
        if not pint:
            return None
        cores |= {b[i][j] for i, j in pint}
        alvos.append(pint)
        cs.append(conjuntos(a))
    if len(cores) != 1:
        return None
    cor = next(iter(cores))
    comuns = set(cs[0])
    for c in cs[1:]:
        comuns &= set(c)
    for t in sorted(comuns):
        if all(c[t] == alvo for c, alvo in zip(cs, alvos)):
            return (t, cor)
    return None


def aplicar(regra, g: Grid) -> Optional[Grid]:
    t, cor = regra
    try:
        cs = conjuntos(g)
    except Exception:
        return None
    if t not in cs:
        return None
    fora = [list(l) for l in g]
    for (i, j) in cs[t]:
        fora[i][j] = cor
    return tuple(tuple(l) for l in fora)


def imune(treino) -> bool:
    if len(treino) < 3:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir(resto)
        if r is None or aplicar(r, treino[i][0]) != treino[i][1]:
            return False
    return True


def _uma(args):
    nome, tr, te = args
    try:
        r = deduzir(tr)
        if r is None:
            return nome, False, False, False, None
        if any(aplicar(r, a) != b for a, b in tr):
            return nome, False, False, False, None
        ok = all(aplicar(r, x) == y for x, y in te)
        return nome, True, imune(tr), ok, r[0]
    except Exception:
        return nome, False, False, False, None


JA = {"195ba7dc", "5d2a5c43", "d282b262", "37d3e8b2", "66e6c45b", "8ee62060",
      "903d1b4a", "981571dc", "9ddd00f0", "ae58858e", "af22c60d", "c7d4e6ad",
      "ca8f78db", "fc754716", "32e9702f", "332efdb3", "47996f11", "00dbd492",
      "72a961c9", "b1fc8b8e", "f45f5ca7", "358ba94e", "5ffb2104", "e0fb7511",
      "00576224", "59341089", "bc4146bd", "070dd51e", "aa18de87", "3194b014",
      "ccd554ac", "cd3c21df"}


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("CELOBJ — testes relacionais entre celula e objeto")
    print("=" * 76)
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
    print(f"  SEM portao passou teste: {len(sem):3d}/400   "
          f"precisao {len(sem)/max(len(ajust),1):.1%}")
    print(f"  COM portao passou teste: {len(com):3d}/400")
    print(f"\n  testes que resolveram: "
          f"{dict(Counter(sem.values()).most_common())}")
    print(f"  testes que ajustaram : "
          f"{dict(Counter(f for _n, f in ajust).most_common(8))}")
    novas = sorted(set(sem) - JA)
    print(f"\n  tarefas: {sorted(sem)}")
    print(f"  INEDITAS: {len(novas)}  {novas}")

    print("\n" + "=" * 76)
    if not ajust:
        print("P0 FALHOU: nenhum teste disparou.")
        return
    if len(novas) >= 8:
        print("P1 · o vocabulario relacional celula-objeto era o que faltava.")
    elif len(novas) >= 3:
        print("P2 · ajuda, no patamar das mudancas de representacao.")
    else:
        print("P3 · a colocacao nas 37 e idiossincratica; nem relacao")
        print("     celula-objeto a captura.")


if __name__ == "__main__":
    main()

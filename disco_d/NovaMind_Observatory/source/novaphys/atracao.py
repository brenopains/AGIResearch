"""
novaphys.atracao — a peca anda ate ENCOSTAR, e a direcao e a que bate em algo
==============================================================================

DE ONDE VEM
-------------
De duas falhas seguidas, medidas ate o fim.

`fluido.py` deu F0. O traco mostrou que a `184a9768` nao enche copo: a peca
SOME de um lugar e APARECE em outro. Isso abriu um censo de conservacao de
forma, 14 tarefas. `encaixe.py` leu essas 14 como quebra-cabeca e deu E0: a
`f3e62deb` tem UMA peca e buracos [1, 91] — o destino e fundo aberto, nao
buraco. A assinatura "conserva forma" nao e uma familia, e um saco de gatos:
quebra-cabeca, escorregar, linha que varre, espelhamento.

Entao a pergunta virou: PARA ONDE a peca vai? Medida, 265 pecas movidas:

    escorrega ate travar ....... 80/265 = 30,2%
    tarefas com 100% ........... e1d2900e (21/21), f3e62deb (6/6)
    parciais ................... 1c56ad9f 32/44, ecaa0ec1 12/16, 9b4c17c4 7/9

Nenhuma direcao global funciona: em `e1d2900e` as direcoes saem 5/5/5/6 nas
quatro ortogonais. A direcao e lida POR PECA. Olhando a tarefa, a leitura e
obvia e deduzivel:

    cada `1` solto escorrega ate ENCOSTAR no bloco `22` mais proximo

    ..1..22....1..1...22....1.....  ->  ....1221.........1221.........

O `1` da coluna 2 anda para a direita ate a 4; o da 11 anda para a esquerda ate
a 7; os dois encostam no mesmo bloco por lados opostos.

A DIRECAO NAO PRECISA DE DICA EXTERNA. E a unica em que o movimento termina
BATENDO NUM OBJETO em vez de na parede. Isso separa `e1d2900e` (tem alvo) de
`f3e62deb` (nao tem alvo nenhum, escorrega ate a borda — essa continua sem
regra de direcao, e fica de fora de proposito).

AS VARIANTES
--------------
    moveis     as cores que se movem, lidas do diff do treino
    dirs       4 ortogonais, ou 8 com as diagonais
    desempate  exigir direcao UNICA que bata em objeto, ou a de menor percurso
    parada     encosta (para antes), ou anda ate o limite

Todos os destinos sao calculados sobre a grade ORIGINAL, nunca em cascata: duas
pecas convergindo para o mesmo bloco por lados opostos nao podem depender da
ordem em que eu as processo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. `e1d2900e` tem que ajustar o treino. Eu li a regra dela a
      mao no diff acima e a medida deu 21/21. Se nao passar, e erro de
      implementacao, nao resultado.

  A1  >= 4 inedidas -> atracao era um mecanismo largo
  A2  2 a 3         -> ajuda, no patamar das mudancas de representacao
  A3  <= 1          -> cauda longa outra vez, agora na terceira tentativa
                       seguida sobre o mesmo censo

MINHA CALIBRAGEM: trinta e duas previsoes, DEZ certas, as vinte e duas erradas
todas otimistas. Nas duas ultimas eu previ E2 e F2 e as duas deram ZERO — e as
duas foram derrotadas nao pela busca, mas por eu ter lido mal o mecanismo.

PREVISAO: A3, exatamente 1. A medida sustenta UMA tarefa limpa (`e1d2900e`).
As parciais (32/44, 12/16, 7/9) nao fecham por reproducao exata, e parcial
nunca virou tarefa nesta arquitetura. Estou prevendo o meu proprio piso pela
primeira vez na sessao, e e o que o dado diz.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL

ORTO = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))
Cel = Tuple[int, int]


def objetos(g: Grid, bg: int):
    """Componentes 4-conexos monocromaticos: (celulas, cor)."""
    h, w = dims(g)
    por_cor: Dict[int, Set[Cel]] = {}
    for i in range(h):
        for j in range(w):
            if g[i][j] != bg:
                por_cor.setdefault(g[i][j], set()).add((i, j))
    fora = []
    for k, s in por_cor.items():
        s = set(s)
        while s:
            p = s.pop()
            c = [p]
            f = deque([p])
            while f:
                y, x = f.popleft()
                for dy, dx in ORTO:
                    q = (y + dy, x + dx)
                    if q in s:
                        s.discard(q)
                        c.append(q)
                        f.append(q)
            fora.append((frozenset(c), k))
    return fora


def deslizar(pc, parede: Set[Cel], h: int, w: int, d):
    """
    Anda ate travar. Devolve (passos, bateu_em_objeto). `bateu_em_objeto` e
    False quando quem barrou foi a borda — e essa e a diferenca que escolhe a
    direcao: a peca vai para onde tem ALGO, nao para onde tem parede.
    """
    dy, dx = d
    k = 0
    while True:
        nv = [(y + dy * (k + 1), x + dx * (k + 1)) for y, x in pc]
        if any(not (0 <= y < h and 0 <= x < w) for y, x in nv):
            return k, False
        if any(q in parede for q in nv):
            return k, True
        k += 1


def aplicar(regra, g: Grid) -> Optional[Grid]:
    try:
        moveis, ndirs, desempate, alvo = regra
        bg = fundo(g)
        h, w = dims(g)
        objs = objetos(g, bg)
        dirs = ORTO if ndirs == 4 else ORTO + DIAG
        fixos = {q for c, k in objs if k not in moveis for q in c}
        if not fixos:
            return None
        andou = False
        fora = [list(l) for l in g]
        for c, k in objs:
            if k not in moveis:
                continue
            # `alvo` decide se uma peca movel pode barrar outra. No `e1d2900e`
            # nao pode: na linha 4 o `1` da coluna 11 bateria no `1` da 14 a 3
            # passos e pararia ali, quando a verdade e ele atravessar e encostar
            # no bloco a 4 passos do outro lado. Peca e atraida por ANCORA, nao
            # por peca. Mas ha tarefa em que empilham; entao e variante, medida.
            outros = fixos if alvo == "fixos" else (
                fixos | {q for c2, k2 in objs
                         if k2 in moveis and c2 != c for q in c2})
            cand = []
            for d in dirs:
                passos, obj = deslizar(c, outros, h, w, d)
                if obj and passos > 0:
                    cand.append((passos, d))
            if not cand:
                continue
            if len(cand) > 1:
                if desempate == "unico":
                    return None
                cand.sort()
            passos, (dy, dx) = cand[0]
            for y, x in c:
                fora[y][x] = bg
            for y, x in c:
                fora[y + dy * passos][x + dx * passos] = k
            andou = True
        if not andou:
            return None
        return tuple(tuple(l) for l in fora)
    except Exception:
        return None


def deduzir(treino):
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    moveis: Optional[FrozenSet[int]] = None
    for a, b in treino:
        bg = fundo(a)
        h, w = dims(a)
        cs = frozenset(a[i][j] for i in range(h) for j in range(w)
                       if a[i][j] != bg and b[i][j] != a[i][j])
        if not cs:
            return None
        if moveis is None:
            moveis = cs
        elif moveis != cs:
            return None
    for ndirs in (4, 8):
        for desempate in ("unico", "perto"):
            for alvo in ("fixos", "todos"):
                r = (moveis, ndirs, desempate, alvo)
                if all(aplicar(r, a) == b for a, b in treino):
                    return r
    return None


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
        ok = all(aplicar(r, x) == y for x, y in te)
        return nome, True, imune(tr), ok, f"{r[1]}dir/{r[2]}/{r[3]}"
    except Exception:
        return nome, False, False, False, None


JA = {"195ba7dc", "5d2a5c43", "d282b262", "37d3e8b2", "66e6c45b", "8ee62060",
      "903d1b4a", "981571dc", "9ddd00f0", "ae58858e", "af22c60d", "c7d4e6ad",
      "ca8f78db", "fc754716", "32e9702f", "332efdb3", "47996f11", "00dbd492",
      "72a961c9", "b1fc8b8e", "f45f5ca7", "358ba94e", "5ffb2104", "e0fb7511",
      "00576224", "59341089", "bc4146bd", "070dd51e", "aa18de87", "3194b014",
      "ccd554ac", "cd3c21df", "55059096"}


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("ATRACAO — a peca anda ate encostar, e a direcao e a que bate em algo")
    print("=" * 76)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [n for n, aj, _i, _o, _f in res if aj]
    imunes = [n for n, _aj, i, _o, _f in res if i]
    sem = {n: f for n, _aj, _i, o, f in res if o}
    com = {n: f for n, _aj, i, o, f in res if i and o}

    print("")
    print(f"  ajustou o TREINO       : {len(ajust):3d}/400  "
          f"({time.time()-t0:.0f}s)")
    print(f"  sobreviveu ao IMUNE    : {len(imunes):3d}/400")
    print(f"  SEM portao passou teste: {len(sem):3d}/400   "
          f"precisao {len(sem)/max(len(ajust),1):.1%}")
    print(f"  COM portao passou teste: {len(com):3d}/400")
    print("")
    print(f"  A0 . e1d2900e ajustou o treino: {'e1d2900e' in ajust}")
    print(f"  variantes: {dict(Counter(sem.values()).most_common())}")
    print(f"  tarefas: {sorted(sem)}")
    novas = sorted(set(sem) - JA)
    print(f"  INEDITAS: {len(novas)}  {novas}")

    print("")
    print("=" * 76)
    if "e1d2900e" not in ajust:
        print("A0 FALHOU: a tarefa lida a mao nao ajustou. Erro de")
        print("     implementacao, nao resultado.")
    if len(novas) >= 4:
        print("A1 . atracao era um mecanismo largo.")
    elif len(novas) >= 2:
        print("A2 . ajuda, no patamar das mudancas de representacao.")
    else:
        print("A3 . cauda longa outra vez, na terceira tentativa sobre o mesmo")
        print("     censo. Previsto.")


if __name__ == "__main__":
    main()

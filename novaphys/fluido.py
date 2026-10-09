"""
novaphys.fluido — recipiente enchendo e liquido escoando
=========================================================

DE ONDE VEM
-------------
A maior familia da amostra do usuario: 5 dos 14 relatos.

    #4   "fluidos e liquidos, como eu coloco agua num copo e como ele preenche"
    #8   "agua entrando no copo mas sempre em FORMATO DE GANCHO, por isso sei a
          direcao com que deve entrar"
    #9   "copos ou recipientes sendo preenchidos"
    #11  "o liquido sai ou entra de ambos recipientes sempre ligando ao cinza
          mas NUNCA PERDE A SIMETRIA E O TAMANHO"
    #13  "liquidos caindo de cima pra baixo... o cinza COLIDE com alguns deles
          ... tipo lava caindo no Minecraft, TRACANDO CAMINHOS POSSIVEIS"

`trajetoria.py` ja tentou duas versoes grosseiras — `escoa` (avanca numa
direcao, desvia pelas diagonais) e `inunda` (alaga a partir de celulas de uma
cor) — e as duas deram zero. Os relatos apontam para algo mais especifico.

AS TRES IDEIAS QUE OS RELATOS CONTEM E AS VERSOES ANTERIORES NAO TINHAM
-------------------------------------------------------------------------
1. A FONTE NAO E UMA COR — e a BORDA. Agua entra de fora, nao de uma celula
   pintada. `inunda` exigia uma cor-fonte e por isso nunca disparava em tarefa
   onde o liquido vem de cima.

2. RECIPIENTE E DIFERENTE DE REGIAO CERCADA. Um copo e aberto em cima. O fundo
   dele E alcancavel a partir da borda, entao `preencher` (fundo nao
   alcancavel) nunca o pega. O que define o copo e nao poder ESCOAR: a agua
   fica onde nao consegue sair na direcao da gravidade.

3. A DIRECAO E LIDA DA FORMA. "Formato de gancho, por isso sei a direcao com
   que deve entrar."

AS QUATRO VARIANTES
---------------------
    retem(d)      celulas de fundo de onde NAO existe raio livre ate a borda
                  na direcao d. E a agua que fica no copo — inclusive em copo
                  aberto em cima, que `preencher` nao pega.
    molha(d)      celulas de fundo alcancaveis a partir da borda entrando na
                  direcao d e espalhando lateralmente. E o liquido que ENTRA.
    interior      fundo dentro da caixa envolvente de algum objeto. E o copo
                  visto como recipiente, sem depender de fechamento.
    cercado       fundo nao alcancavel da borda (a versao antiga, mantida como
                  base de comparacao).

A cor e lida do diff: constante, ou a cor do objeto que contem a celula.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  F0  PRE-CONDICAO. Alguma variante tem que disparar em alguma tarefa.

  F1  >= 4 inedidas -> fluido era a familia grande que o relato indicou, e 5/14
                       da amostra se traduz em cobertura
  F2  2 a 3         -> ajuda, no patamar das mudancas de representacao
  F3  <= 1          -> nem fluido; e a cauda longa confirmada tambem na maior
                       familia que a amostra humana revelou

MINHA CALIBRAGEM: trinta previsoes, DEZ certas, as vinte erradas todas
otimistas. E o dado mais recente e severo: `sovaco.py` implementou uma regra
que o usuario verbalizou com precisao cirurgica, funcionou de primeira com
precisao 100%, e cobriu UMA tarefa.

PREVISAO: F2, 2 a 3. Se a razao regra/tarefa do sovaco (1 para 1) valer aqui,
cinco relatos de fluido dariam ~5 tarefas — mas as minhas quatro variantes sao
aproximacoes das cinco descricoes dele, nao as cinco regras exatas. Aposto na
metade.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo, objetos
from novaphys.eter import PASTA_AVAL

DIRS = {"baixo": (1, 0), "cima": (-1, 0), "dir": (0, 1), "esq": (0, -1)}
ORTO = ((-1, 0), (1, 0), (0, -1), (0, 1))


def retem(g: Grid, d: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """
    Fundo de onde NAO ha raio livre ate a borda na direcao d: a agua que fica.
    Pega copo ABERTO em cima, que `cercado` nunca pega.
    """
    bg = fundo(g)
    h, w = dims(g)
    dy, dx = d
    fora = set()
    for i in range(h):
        for j in range(w):
            if g[i][j] != bg:
                continue
            y, x = i + dy, j + dx
            livre = True
            while 0 <= y < h and 0 <= x < w:
                if g[y][x] != bg:
                    livre = False
                    break
                y, x = y + dy, x + dx
            if not livre:
                fora.add((i, j))
    return fora


def molha(g: Grid, d: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """
    Fundo alcancavel entrando pela borda oposta a d e espalhando lateralmente.
    E o liquido que ENTRA — a fonte e a borda, nao uma cor.
    """
    bg = fundo(g)
    h, w = dims(g)
    dy, dx = d
    lados = ((0, -1), (0, 1)) if dy else ((-1, 0), (1, 0))
    fila = deque()
    vis: Set[Tuple[int, int]] = set()
    if dy > 0:
        ini = [(0, j) for j in range(w)]
    elif dy < 0:
        ini = [(h - 1, j) for j in range(w)]
    elif dx > 0:
        ini = [(i, 0) for i in range(h)]
    else:
        ini = [(i, w - 1) for i in range(h)]
    for p in ini:
        if g[p[0]][p[1]] == bg:
            vis.add(p)
            fila.append(p)
    while fila:
        y, x = fila.popleft()
        for ey, ex in (d,) + lados:
            ny, nx = y + ey, x + ex
            if (0 <= ny < h and 0 <= nx < w and g[ny][nx] == bg
                    and (ny, nx) not in vis):
                vis.add((ny, nx))
                fila.append((ny, nx))
    return vis


def interior(g: Grid) -> Set[Tuple[int, int]]:
    """Fundo dentro da caixa envolvente de algum objeto: o copo como recipiente."""
    bg = fundo(g)
    h, w = dims(g)
    fora = set()
    for cel, _k in objetos(g, bg):
        ys = [y for y, _x in cel]
        xs = [x for _y, x in cel]
        for i in range(min(ys), max(ys) + 1):
            for j in range(min(xs), max(xs) + 1):
                if g[i][j] == bg:
                    fora.add((i, j))
    return fora


def cercado(g: Grid) -> Set[Tuple[int, int]]:
    bg = fundo(g)
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
        for dy, dx in ORTO:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                    and g[ny][nx] == bg):
                vis[ny][nx] = True
                fila.append((ny, nx))
    return {(i, j) for i in range(h) for j in range(w)
            if g[i][j] == bg and not vis[i][j]}


def variantes(g: Grid) -> Dict[str, Set[Tuple[int, int]]]:
    fora: Dict[str, Set[Tuple[int, int]]] = {"cercado": cercado(g),
                                             "interior": interior(g)}
    for n, d in DIRS.items():
        fora[f"retem_{n}"] = retem(g, d)
        fora[f"molha_{n}"] = molha(g, d)
    # interseccoes uteis: agua que fica DENTRO do recipiente
    fora["retem_baixo_interior"] = fora["retem_baixo"] & fora["interior"]
    return {k: v for k, v in fora.items() if v}


def cor_do_dono(g: Grid, p: Tuple[int, int]) -> Optional[int]:
    """A cor do objeto cuja caixa contem a celula — o copo que segura a agua."""
    bg = fundo(g)
    for cel, k in objetos(g, bg):
        ys = [y for y, _x in cel]
        xs = [x for _y, x in cel]
        if min(ys) <= p[0] <= max(ys) and min(xs) <= p[1] <= max(xs):
            return k
    return None


def deduzir(treino):
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    alvos, cores, vs = [], set(), []
    for a, b in treino:
        bg = fundo(a)
        h, w = dims(a)
        # ERRO DE ESCOPO REMOVIDO: eu exigia `preserva` aqui, copiado do
        # perfil das 38 tarefas. Mas as de preenchimento MUDAM celulas
        # existentes tambem — a 184a9768 altera 51 delas. A guarda rejeitava
        # antes da verificacao de verdade, e era redundante: a comparacao
        # exata `aplicar(r,a)==b` ja rejeita quem muda o que nao devia.
        p = {(i, j): b[i][j] for i in range(h) for j in range(w)
             if a[i][j] == bg and b[i][j] != bg}
        if not p:
            return None
        alvos.append(p)
        cores |= set(p.values())
        vs.append(variantes(a))
    comuns = set(vs[0])
    for v in vs[1:]:
        comuns &= set(v)
    for nome in sorted(comuns):
        if any(v[nome] != set(al) for v, al in zip(vs, alvos)):
            continue
        if len(cores) == 1:
            return (nome, next(iter(cores)))
        if all(al[p] == cor_do_dono(a, p) for (a, _b), al in zip(treino, alvos)
               for p in al):
            return (nome, None)
    return None


def aplicar(regra, g: Grid) -> Optional[Grid]:
    try:
        nome, cor = regra
        v = variantes(g)
        if nome not in v:
            return None
        fora = [list(l) for l in g]
        for (i, j) in v[nome]:
            c = cor if cor is not None else cor_do_dono(g, (i, j))
            if c is None:
                return None
            fora[i][j] = c
        return tuple(tuple(l) for l in fora)
    except Exception:
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
      "ccd554ac", "cd3c21df", "55059096"}


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("FLUIDO — recipiente enchendo e liquido escoando")
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
    print(f"\n  variantes: {dict(Counter(sem.values()).most_common())}")
    novas = sorted(set(sem) - JA)
    print(f"  tarefas: {sorted(sem)}")
    print(f"  INEDITAS: {len(novas)}  {novas}")

    print("\n" + "=" * 76)
    if not ajust:
        print("F0 FALHOU: nenhuma variante disparou.")
        return
    if len(novas) >= 4:
        print("F1 · fluido era a familia grande que o relato indicou.")
    elif len(novas) >= 2:
        print("F2 · ajuda, no patamar das mudancas de representacao.")
    else:
        print("F3 · nem fluido. Cauda longa confirmada tambem na maior")
        print("     familia que a amostra humana revelou.")


if __name__ == "__main__":
    main()

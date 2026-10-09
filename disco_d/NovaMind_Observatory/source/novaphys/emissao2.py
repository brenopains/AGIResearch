"""
novaphys.emissao2 — emissao CONDICIONADA
==========================================

O DIAGNOSTICO QUE ORIGINA ESTE ARQUIVO
----------------------------------------
`emissao.py` implementou quatro familias de desenho puras — raio, preencher,
conectar, contorno — e deu 2/400, ambas de `conectar`, precisao 100%.

O numero que diagnostica nao e o 2: e que **so 2 tarefas sequer ajustaram o
treino**. As familias puras quase nunca explicam os dados exatamente.

E isso confirma a ressalva registrada ANTES de rodar: as versoes reais no ARC
vem COM CONDICAO. Nao e "emita raio de toda celula vermelha" — e "emita raio
das vermelhas QUE ESTAO NA BORDA", "preencha as regioes COM MAIS DE N
CELULAS", "conecte os pares DA MESMA COR". Familias incondicionais disparam ou
tudo ou nada.

O CONSERTO, E ELE COMPOE PECAS QUE JA EXISTEM
-----------------------------------------------
    celula(i,j) = TESTE ? emitir : nada

A emissao ganha um PREDICADO DE FONTE: quais celulas emitem. E um modo: pintar
so sobre fundo, ou sobrescrever.

Predicados de fonte, todos computaveis sem saber a tarefa:

    cor==k          celulas de uma cor (o comportamento antigo)
    borda           celulas na borda da grade
    maior_obj       celulas do maior objeto
    menor_obj       celulas do menor objeto
    unico_tam       celulas de objetos cujo tamanho e unico
    isolado         celulas sem vizinho nao-fundo

CUIDADO COM O ESPACO, e ele e o risco medido
----------------------------------------------
A sessao mediu cinco vezes que ampliar o espaco custa mais do que rende (21x
de sobreajuste com 8 lentes; 53 falsos positivos gerados por modificadores e
gaveta; recombinacao 7 -> 2). Entao a expansao aqui e deliberadamente
CONTIDA: o predicado SUBSTITUI a selecao por cor, nao multiplica com ela.

E o portao imunologico continua, com os dois numeros reportados — porque
`ladrilho.py` mediu que em espaco pequeno e exclusivo o portao custa
verdadeiro positivo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C0  PRE-CONDICAO. As 2 de `emissao.py` (070dd51e, aa18de87) tem que
      sobreviver. Condicionar so pode ADICIONAR: `cor==k` reproduz o
      comportamento antigo.

  C1  >= 6 novas   -> a condicao era o que faltava; emissao e a familia grande
                      que o censo apontou (199 tarefas com formas novas)
  C2  3 a 5        -> ajuda, no patamar das outras mudancas de representacao
  C3  <= 2         -> nem condicionada; as 199 precisam de outra coisa

MINHA CALIBRAGEM: vinte e uma previsoes na sessao, SETE certas, as quatorze
erradas TODAS otimistas. Isso ja e um vies medido e eu aplico a correcao.

PREVISAO: C2 na faixa baixa, +2 a +4 sobre as 2 atuais. O argumento a favor e
que o diagnostico foi especifico e o conserto e direcionado. O argumento contra
e que "condicao" no ARC raramente e um predicado simples de celula — costuma
ser relacional ("o objeto que e diferente dos outros", "o que esta dentro do
retangulo"), e predicado relacional nao esta nesta lista.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Callable, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.emissao import (DIRS, emitir_conectar, emitir_contorno,
                              emitir_preencher, fundo)
from novaphys.eter import PASTA_AVAL


def _objetos(g: Grid, bg: int) -> List[Set[Tuple[int, int]]]:
    h, w = dims(g)
    vis = [[False] * w for _ in range(h)]
    fora = []
    for i in range(h):
        for j in range(w):
            if vis[i][j] or g[i][j] == bg:
                continue
            cor = g[i][j]
            cel, fila = set(), deque([(i, j)])
            vis[i][j] = True
            while fila:
                y, x = fila.popleft()
                cel.add((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                            and g[ny][nx] == cor):
                        vis[ny][nx] = True
                        fila.append((ny, nx))
            fora.append(cel)
    return fora


def fontes(g: Grid, bg: int) -> Dict[str, Set[Tuple[int, int]]]:
    """
    Conjunto de celulas que emitem, por predicado. Calculado uma vez por grade.
    O predicado SUBSTITUI a selecao por cor em vez de multiplicar com ela —
    conter o espaco e deliberado: ampliar custou 21x de sobreajuste antes.
    """
    h, w = dims(g)
    fora: Dict[str, Set[Tuple[int, int]]] = {}
    for c in {v for l in g for v in l if v != bg}:
        fora[f"cor=={c}"] = {(i, j) for i in range(h) for j in range(w)
                             if g[i][j] == c}
    fora["borda"] = {(i, j) for i in range(h) for j in range(w)
                     if g[i][j] != bg and (i in (0, h - 1) or
                                           j in (0, w - 1))}
    objs = _objetos(g, bg)
    if objs:
        m = max(len(o) for o in objs)
        n = min(len(o) for o in objs)
        fora["maior_obj"] = {p for o in objs if len(o) == m for p in o}
        fora["menor_obj"] = {p for o in objs if len(o) == n for p in o}
        ct = Counter(len(o) for o in objs)
        fora["unico_tam"] = {p for o in objs if ct[len(o)] == 1 for p in o}
    fora["isolado"] = {
        (i, j) for i in range(h) for j in range(w) if g[i][j] != bg
        and not any(0 <= i + dy < h and 0 <= j + dx < w
                    and g[i + dy][j + dx] != bg
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))}
    return {k: v for k, v in fora.items() if v}


def raio_cond(g: Grid, bg: int, src: Set[Tuple[int, int]], d: str,
              cor: Optional[int], sobrescreve: bool) -> Grid:
    h, w = dims(g)
    dy, dx = DIRS[d]
    fora = [list(l) for l in g]
    for (i, j) in src:
        y, x = i + dy, j + dx
        while 0 <= y < h and 0 <= x < w:
            if g[y][x] != bg and not sobrescreve:
                break
            fora[y][x] = g[i][j] if cor is None else cor
            y, x = y + dy, x + dx
    return tuple(tuple(l) for l in fora)


SEM_FONTE = [("preencher", emitir_preencher), ("conectar", emitir_conectar),
             ("contorno", emitir_contorno)]


def deduzir(treino):
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    bg = fundo(treino[0][0])
    mudou: Set[int] = set()
    for a, b in treino:
        for la, lb in zip(a, b):
            for x, y in zip(la, lb):
                if x != y:
                    mudou.add(y)
    if not mudou:
        return None
    cores: List[Optional[int]] = [None] + sorted(mudou)

    # familias sem fonte, comportamento identico ao de `emissao.py`
    for nome, fn in SEM_FONTE:
        for cor in cores:
            try:
                if all(fn(a, bg, None, None, cor) == b for a, b in treino):
                    return ("simples", nome, None, None, cor, False)
            except Exception:
                continue

    # raio CONDICIONADO
    fs = [fontes(a, bg) for a, _b in treino]
    comuns = set(fs[0])
    for f in fs[1:]:
        comuns &= set(f)
    for pred in sorted(comuns):
        for d in DIRS:
            for cor in cores:
                for sob in (False, True):
                    try:
                        if all(raio_cond(a, bg, f[pred], d, cor, sob) == b
                               for (a, b), f in zip(treino, fs)):
                            return ("raio", pred, d, None, cor, sob)
                    except Exception:
                        continue
    return None


def aplicar(regra, g: Grid) -> Optional[Grid]:
    tipo, a1, d, _x, cor, sob = regra
    bg = fundo(g)
    try:
        if tipo == "simples":
            fn = dict(SEM_FONTE)[a1]
            return fn(g, bg, None, None, cor)
        f = fontes(g, bg)
        if a1 not in f:
            return None
        return raio_cond(g, bg, f[a1], d, cor, sob)
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
        rot = f"{r[0]}:{r[1]}" + (f":{r[2]}" if r[2] else "")
        return nome, True, imune(tr), ok, rot
    except Exception:
        return nome, False, False, False, None


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 78)
    print("EMISSAO2 — emissao condicionada por predicado de fonte")
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

    print(f"\n  ajustou o TREINO       : {len(ajust):3d}/400  (`emissao`: 2)"
          f"  ({time.time()-t0:.0f}s)")
    print(f"  sobreviveu ao IMUNE    : {len(imunes):3d}/400")
    print(f"\n  SEM portao passou teste: {len(sem):3d}/400   "
          f"precisao {len(sem)/max(len(ajust),1):.1%}")
    print(f"  COM portao passou teste: {len(com):3d}/400   "
          f"precisao {len(com)/max(len(imunes),1):.1%}")
    antes = {"070dd51e", "aa18de87"}
    print(f"\nC0 · das 2 de `emissao`, mantidas: {sorted(antes & set(sem))}")
    print(f"\n  regras: {dict(Counter(sem.values()).most_common())}")
    print(f"  tarefas: {sorted(sem)}")
    print(f"  novas  : {sorted(set(sem) - antes)}")

    print("\n" + "=" * 78)
    if not antes <= set(sem):
        print(f"C0 FALHOU: perdi {sorted(antes - set(sem))}.")
    nov = len(set(sem) - antes)
    if nov >= 6:
        print("C1 · a condicao era o que faltava. Emissao e a familia grande")
        print("     que o censo apontou.")
    elif nov >= 3:
        print("C2 · ajuda, no patamar das outras mudancas de representacao.")
    else:
        print("C3 · nem condicionada. As 199 com formas novas precisam de")
        print("     predicado RELACIONAL, nao de predicado de celula.")


if __name__ == "__main__":
    main()

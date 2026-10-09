"""
novaphys.compositor — MONTAR a saida, em vez de derivar da entrada
====================================================================

O CAMINHO ATE AQUI, TODO MEDIDO
---------------------------------
    `inventario.py`   199 das nao resolvidas produzem formas que NAO existiam
                      na entrada. Criacao e a maioria.
    `emissao.py`      quatro familias de desenho puras: 2/400.
    `emissao2.py`     condicionadas por predicado de celula: ZERO novas.
    `relacional.py`   8 propriedades que comparam objetos: ZERO, e nem falso
                      positivo novo — sao redundantes.
    `condicao.py`     par de propriedades construido por tarefa: ZERO.
    `saida.py`        73% das saidas nao tem NENHUMA assinatura canonica, e
                      1347 das 1363 grades de saida sao UNICAS. So 4 se
                      repetem entre tarefas.

A ultima medida corrige a proposta do usuario de um jeito preciso: gerar a
figura final e a direcao certa, mas NAO HA BANCO A RECONHECER. Cada saida e
nova. E a diferenca entre reconhecer uma palavra e formar uma frase.

Entao o que falta nao e um classificador sobre formas tipicas — e um
COMPOSITOR: a saida e MONTADA a partir de leituras da entrada, e a montagem
especifica nunca foi vista.

O QUE MUDA EM RELACAO A TUDO ANTERIOR
---------------------------------------
    motor de delta   saida = objetos da entrada transformados
    emissao          saida = entrada + celulas desenhadas
    COMPOSITOR       saida = tela nova, cujo TAMANHO e CONTEUDO sao lidos da
                     entrada. A entrada vira FONTE DE PARAMETROS, nao
                     substrato a modificar.

Quatro familias, todas com parametros deduzidos e nada sorteado:

  A  CONSTANTE      a mesma saida para toda entrada. Trivial, mas e uma
                    familia real e o custo de testar e zero.
  B  TELA UNIFORME  saida = grade h x w de uma cor, onde h, w e a cor sao
                    LIDOS de quantidades da entrada (n de objetos, n de cores,
                    tamanho do maior, dimensoes, contagem da cor dominante).
  C  RECORTE POR SELECAO  saida = o objeto escolhido por uma propriedade,
                    recortado sozinho na sua caixa. E a familia `dim_subgrade`,
                    17 tarefas pelo censo.
  D  REPETICAO CONTADA    saida = um bloco repetido N vezes, com N lido de uma
                    contagem da entrada.

As LEITURAS sao funcoes inteiras da entrada, computaveis sem saber a tarefa. A
deducao resolve QUAL leitura, comparando com as dimensoes e conteudos
observados — nao enumera saidas.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C0  PRE-CONDICAO. Alguma familia tem que disparar. Se nenhuma dispara em
      nenhuma tarefa, a implementacao esta quebrada.

  C1  >= 6 tarefas novas  -> compor era o degrau
  C2  3 a 5               -> ajuda, no patamar das mudancas de representacao
                             desta sessao (+3, +3, +3, +2)
  C3  <= 2                -> nem compondo; e ai a familia de 199 exige
                             composicao com estrutura interna, nao tela
                             parametrizada

MINHA CALIBRAGEM: vinte e cinco previsoes, NOVE certas, as dezesseis erradas
TODAS otimistas. As certas foram as que fiz depois de comecar a corrigir para
o proprio vies.

PREVISAO: C2 na faixa baixa, +2 a +4. A familia C tem 17 candidatas pelo
censo e e a mais promissora; B e D sao apostas menores. Nao aposto alto porque
as quatro ultimas construcoes deram 0, 0, 0 e 2, e todas tinham argumento
estrutural tao bom quanto este.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.eter import PASTA_AVAL


def fundo(g: Grid) -> int:
    return Counter(v for l in g for v in l).most_common(1)[0][0]


def objetos(g: Grid, bg: int, diag: bool = False):
    h, w = dims(g)
    viz = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        viz += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
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
                for dy, dx in viz:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                            and g[ny][nx] == cor):
                        vis[ny][nx] = True
                        fila.append((ny, nx))
            fora.append((frozenset(cel), cor))
    return fora


# ══════════════════════════════════════════════ leituras inteiras da entrada
def leituras(g: Grid) -> Dict[str, int]:
    """Quantidades computaveis sem saber a tarefa. Sao os PARAMETROS."""
    bg = fundo(g)
    h, w = dims(g)
    objs = objetos(g, bg)
    cont = Counter(v for l in g for v in l if v != bg)
    tams = [len(c) for c, _k in objs] or [0]
    ct = Counter(tams)
    return {
        "n_objs": len(objs),
        "n_cores": len(cont),
        "tam_maior": max(tams),
        "tam_menor": min(tams),
        "altura": h,
        "largura": w,
        "n_dominante": cont.most_common(1)[0][1] if cont else 0,
        "n_tam_unico": sum(1 for t in tams if ct[t] == 1),
        "n_nao_fundo": sum(cont.values()),
    }


def cores_lidas(g: Grid) -> Dict[str, int]:
    bg = fundo(g)
    objs = objetos(g, bg)
    cont = Counter(v for l in g for v in l if v != bg)
    fora = {"fundo": bg}
    if cont:
        fora["dominante"] = cont.most_common(1)[0][0]
        fora["rara"] = cont.most_common()[-1][0]
    if objs:
        m = max(len(c) for c, _k in objs)
        n = min(len(c) for c, _k in objs)
        fora["cor_maior"] = next(k for c, k in objs if len(c) == m)
        fora["cor_menor"] = next(k for c, k in objs if len(c) == n)
    return fora


SELETORES: List[Tuple[str, Callable]] = [
    ("maior", lambda objs: max(objs, key=lambda o: len(o[0]))),
    ("menor", lambda objs: min(objs, key=lambda o: len(o[0]))),
    ("tam_unico", lambda objs: next(
        o for o in objs
        if Counter(len(c) for c, _k in objs)[len(o[0])] == 1)),
    ("cor_unica", lambda objs: next(
        o for o in objs
        if Counter(k for _c, k in objs)[o[1]] == 1)),
]


def recortar(g: Grid, sel: str, diag: bool) -> Optional[Grid]:
    bg = fundo(g)
    objs = objetos(g, bg, diag)
    if not objs:
        return None
    try:
        cel, _k = dict(SELETORES)[sel](objs)
    except (StopIteration, ValueError):
        return None
    ys = [y for y, _x in cel]
    xs = [x for _y, x in cel]
    y0, x0, y1, x1 = min(ys), min(xs), max(ys), max(xs)
    return tuple(tuple(g[y][x] for x in range(x0, x1 + 1))
                 for y in range(y0, y1 + 1))


# ══════════════════════════════════════════════ deducao das quatro familias
def deduzir(treino):
    # A — saida constante
    saidas = {b for _a, b in treino}
    if len(saidas) == 1 and len(treino) > 1:
        return ("const", next(iter(saidas)))

    # C — recorte por selecao
    for sel, _f in SELETORES:
        for diag in (False, True):
            try:
                if all(recortar(a, sel, diag) == b for a, b in treino):
                    return ("recorte", sel, diag)
            except Exception:
                continue

    # B — tela uniforme com h, w e cor LIDOS da entrada
    ls = [leituras(a) for a, _b in treino]
    cs = [cores_lidas(a) for a, _b in treino]
    hs = [dims(b)[0] for _a, b in treino]
    ws = [dims(b)[1] for _a, b in treino]
    uni = [len({c for l in b for c in l}) == 1 for _a, b in treino]
    if all(uni):
        cor_b = [next(iter({c for l in b for c in l})) for _a, b in treino]
        for eh in _exprs(ls, hs):
            for ew in _exprs(ls, ws):
                for ec in _exprs_cor(cs, cor_b):
                    return ("tela", eh, ew, ec)

    # D — repeticao contada: saida = bloco (a entrada) repetida N x M
    for eh in _exprs(ls, [h // dims(a)[0] if dims(a)[0] and
                          h % dims(a)[0] == 0 else -1
                          for (a, _b), h in zip(treino, hs)]):
        for ew in _exprs(ls, [w // dims(a)[1] if dims(a)[1] and
                              w % dims(a)[1] == 0 else -1
                              for (a, _b), w in zip(treino, ws)]):
            if all(_repetir(a, _ler(eh, l), _ler(ew, l)) == b
                   for (a, b), l in zip(treino, ls)):
                return ("repetir", eh, ew)
    return None


def _exprs(ls: List[Dict[str, int]], alvos: List[int]):
    """Que leitura (ou constante) reproduz os alvos em TODOS os pares?"""
    if any(v < 0 for v in alvos):
        return []
    fora = []
    if len(set(alvos)) == 1:
        fora.append(("const", alvos[0]))
    for k in ls[0]:
        if all(l.get(k) == v for l, v in zip(ls, alvos)):
            fora.append(("ler", k))
        for m in (2, 3):
            if all(l.get(k, 0) * m == v for l, v in zip(ls, alvos)):
                fora.append(("mul", k, m))
    return fora


def _exprs_cor(cs, alvos):
    fora = []
    if len(set(alvos)) == 1:
        fora.append(("const", alvos[0]))
    for k in cs[0]:
        if all(c.get(k) == v for c, v in zip(cs, alvos)):
            fora.append(("ler", k))
    return fora


def _ler(e, l: Dict[str, int]) -> int:
    if e[0] == "const":
        return e[1]
    if e[0] == "ler":
        return l.get(e[1], -1)
    return l.get(e[1], 0) * e[2]


def _repetir(g: Grid, na: int, nb: int) -> Optional[Grid]:
    if na < 1 or nb < 1 or na > 10 or nb > 10:
        return None
    h, w = dims(g)
    if h * na > 30 or w * nb > 30:
        return None
    return tuple(tuple(g[i % h][j % w] for j in range(w * nb))
                 for i in range(h * na))


def aplicar(regra, g: Grid) -> Optional[Grid]:
    try:
        t = regra[0]
        if t == "const":
            return regra[1]
        if t == "recorte":
            return recortar(g, regra[1], regra[2])
        if t == "tela":
            l, c = leituras(g), cores_lidas(g)
            h, w = _ler(regra[1], l), _ler(regra[2], l)
            cor = (regra[3][1] if regra[3][0] == "const"
                   else c.get(regra[3][1], -1))
            if not (0 < h <= 30 and 0 < w <= 30) or cor < 0:
                return None
            return tuple(tuple(cor for _ in range(w)) for _ in range(h))
        if t == "repetir":
            l = leituras(g)
            return _repetir(g, _ler(regra[1], l), _ler(regra[2], l))
    except Exception:
        return None
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


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("COMPOSITOR — montar a saida a partir de leituras da entrada")
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
    print(f"\n  SEM portao passou teste: {len(sem):3d}/400   "
          f"precisao {len(sem)/max(len(ajust),1):.1%}")
    print(f"  COM portao passou teste: {len(com):3d}/400")
    print(f"\n  familias que resolveram: "
          f"{dict(Counter(sem.values()).most_common())}")
    print(f"  familias que ajustaram : "
          f"{dict(Counter(f for _n, f in ajust).most_common())}")
    ja = {"195ba7dc", "5d2a5c43", "d282b262", "37d3e8b2", "66e6c45b",
          "8ee62060", "903d1b4a", "981571dc", "9ddd00f0", "ae58858e",
          "af22c60d", "c7d4e6ad", "ca8f78db", "fc754716", "32e9702f",
          "332efdb3", "47996f11", "00dbd492", "72a961c9", "b1fc8b8e",
          "f45f5ca7", "358ba94e", "5ffb2104", "e0fb7511", "00576224",
          "59341089", "bc4146bd", "070dd51e", "aa18de87"}
    novas = sorted(set(sem) - ja)
    print(f"\n  tarefas: {sorted(sem)}")
    print(f"  INEDITAS para a uniao: {len(novas)}  {novas}")

    print("\n" + "=" * 76)
    if not ajust:
        print("C0 FALHOU: nenhuma familia disparou.")
        return
    if len(novas) >= 6:
        print("C1 · compor era o degrau.")
    elif len(novas) >= 3:
        print("C2 · ajuda, no patamar das mudancas de representacao da sessao.")
    else:
        print("C3 · nem compondo. A familia de 199 exige composicao com")
        print("     ESTRUTURA INTERNA, nao tela parametrizada.")


if __name__ == "__main__":
    main()

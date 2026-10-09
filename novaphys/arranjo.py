"""
novaphys.arranjo — composicao com ESTRUTURA INTERNA
====================================================

O QUE O COMPOSITOR FEZ E O QUE FALTOU
---------------------------------------
`compositor.py` montou a saida a partir de leituras da entrada e deu +3
inedidas com precisao 100%: recortar o objeto selecionado, tela uniforme com
dimensoes lidas, e repeticao por contagem.

Mas em todas elas o CONTEUDO e homogeneo: uma copia inteira, ou uma cor so. A
saida nao tem PARTES que se relacionam.

O censo diz que faltam 199 tarefas produzindo formas novas, e cobrimos 5. O que
falta e composicao onde a saida e um ARRANJO: varias partes distintas,
posicionadas por uma regra.

O QUE ENTRA
-------------
    E  ARRANJO DE OBJETOS  a saida e composta pelos objetos da entrada, cada um
                           recortado na sua caixa, dispostos em linha ou coluna,
                           ORDENADOS por uma propriedade (tamanho, cor, posicao
                           original). A ordem e deduzida, nao escolhida.

    F  PARTE REPETIDA      a saida e UM objeto selecionado, repetido N vezes,
                           com N lido de uma contagem da entrada.

A diferenca em relacao a tudo anterior: a saida tem MAIS DE UMA PARTE, e a
posicao de cada uma depende de uma relacao (a ordem) e nao de uma propriedade
individual.

LIMITE DECLARADO, e ele e severo
----------------------------------
Para o arranjo em linha ser bem definido, as partes precisam ter caixas do
MESMO tamanho — senao o layout e ambiguo e eu estaria escolhendo por elas. As
tarefas com partes de tamanhos diferentes ficam fora por construcao, e isso
provavelmente e a maioria.

Prefiro a familia estreita e honesta a uma larga com regra de layout inventada
por mim.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  E0  PRE-CONDICAO. Alguma familia tem que disparar.

  E1  >= 5 inedidas  -> arranjo era o degrau
  E2  2 a 4          -> ajuda, no patamar das cinco mudancas de representacao
                        desta sessao (+3, +3, +3, +2, +3)
  E3  <= 1           -> arranjo linear nao e a forma da composicao no ARC

MINHA CALIBRAGEM: vinte e seis previsoes, DEZ certas, as dezesseis erradas
todas otimistas — e concentradas na primeira metade, antes de eu passar a medir
a premissa antes de construir.

PREVISAO: E2 na faixa baixa, +1 a +3. A restricao de caixas iguais e severa e
deve cortar a maior parte da familia. Aposto que `F` (parte repetida) rende
mais que `E` (arranjo ordenado), porque exige menos das partes.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo, leituras, objetos
from novaphys.eter import PASTA_AVAL


def _caixa(cel) -> Tuple[int, int, int, int]:
    ys = [y for y, _x in cel]
    xs = [x for _y, x in cel]
    return min(ys), min(xs), max(ys), max(xs)


def _recorte(g: Grid, cel) -> Grid:
    y0, x0, y1, x1 = _caixa(cel)
    return tuple(tuple(g[y][x] for x in range(x0, x1 + 1))
                 for y in range(y0, y1 + 1))


ORDENS: List[Tuple[str, Callable]] = [
    ("tam_cresc", lambda o: len(o[0])),
    ("tam_decr", lambda o: -len(o[0])),
    ("cor", lambda o: o[1]),
    ("pos_linha", lambda o: (_caixa(o[0])[0], _caixa(o[0])[1])),
    ("pos_coluna", lambda o: (_caixa(o[0])[1], _caixa(o[0])[0])),
]


def arranjar(g: Grid, ordem: str, eixo: str, diag: bool) -> Optional[Grid]:
    """
    Os objetos da entrada, recortados e dispostos em linha ou coluna, na ordem
    dada. Exige caixas do MESMO tamanho — senao o layout seria escolha minha.
    """
    bg = fundo(g)
    objs = objetos(g, bg, diag)
    if len(objs) < 2:
        return None
    partes = [_recorte(g, c) for c, _k in objs]
    if len({dims(p) for p in partes}) != 1:
        return None
    idx = sorted(range(len(objs)), key=lambda i: dict(ORDENS)[ordem](objs[i]))
    ps = [partes[i] for i in idx]
    h, w = dims(ps[0])
    if eixo == "coluna":
        if h * len(ps) > 30:
            return None
        return tuple(l for p in ps for l in p)
    if w * len(ps) > 30:
        return None
    return tuple(tuple(v for p in ps for v in p[i]) for i in range(h))


SELS: List[Tuple[str, Callable]] = [
    ("maior", lambda objs: max(objs, key=lambda o: len(o[0]))),
    ("menor", lambda objs: min(objs, key=lambda o: len(o[0]))),
    ("tam_unico", lambda objs: next(
        o for o in objs
        if Counter(len(c) for c, _k in objs)[len(o[0])] == 1)),
]


def repetir_parte(g: Grid, sel: str, eixo: str, n: int,
                  diag: bool) -> Optional[Grid]:
    """Um objeto selecionado, recortado, repetido N vezes."""
    if n < 1 or n > 12:
        return None
    bg = fundo(g)
    objs = objetos(g, bg, diag)
    if not objs:
        return None
    try:
        cel, _k = dict(SELS)[sel](objs)
    except (StopIteration, ValueError):
        return None
    p = _recorte(g, cel)
    h, w = dims(p)
    if eixo == "coluna":
        if h * n > 30:
            return None
        return tuple(l for _ in range(n) for l in p)
    if w * n > 30:
        return None
    return tuple(tuple(v for _ in range(n) for v in p[i]) for i in range(h))


def _exprs(ls: List[Dict[str, int]], alvos: List[int]):
    if any(v is None or v < 0 for v in alvos):
        return []
    fora = []
    if len(set(alvos)) == 1:
        fora.append(("const", alvos[0]))
    for k in ls[0]:
        if all(l.get(k) == v for l, v in zip(ls, alvos)):
            fora.append(("ler", k))
    return fora


def _ler(e, l) -> int:
    return e[1] if e[0] == "const" else l.get(e[1], -1)


def deduzir(treino):
    # E — arranjo ordenado
    for ordem, _f in ORDENS:
        for eixo in ("coluna", "linha"):
            for diag in (False, True):
                try:
                    if all(arranjar(a, ordem, eixo, diag) == b
                           for a, b in treino):
                        return ("arranjo", ordem, eixo, diag)
                except Exception:
                    continue
    # F — parte repetida N vezes, N lido de uma contagem
    ls = [leituras(a) for a, _b in treino]
    for sel, _f in SELS:
        for eixo in ("coluna", "linha"):
            for diag in (False, True):
                alvos = []
                for (a, b) in treino:
                    p = repetir_parte(a, sel, eixo, 1, diag)
                    if p is None:
                        alvos = None
                        break
                    ph, pw = dims(p)
                    bh, bw = dims(b)
                    d = bh // ph if eixo == "coluna" else bw // pw
                    r = bh % ph if eixo == "coluna" else bw % pw
                    alvos.append(-1 if r else d)
                if not alvos:
                    continue
                for e in _exprs(ls, alvos):
                    if all(repetir_parte(a, sel, eixo, _ler(e, l), diag) == b
                           for (a, b), l in zip(treino, ls)):
                        return ("repete_parte", sel, eixo, diag, e)
    return None


def aplicar(regra, g: Grid) -> Optional[Grid]:
    try:
        if regra[0] == "arranjo":
            return arranjar(g, regra[1], regra[2], regra[3])
        _t, sel, eixo, diag, e = regra
        return repetir_parte(g, sel, eixo, _ler(e, leituras(g)), diag)
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
      "ccd554ac", "cd3c21df"}


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("ARRANJO — composicao com estrutura interna")
    print("=" * 76)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [(n, f) for n, aj, _i, _o, f in res if aj]
    imunes = [n for n, _aj, i, _o, _f in res if i]
    sem = {n: f for n, _aj, _i, o, f in res if o}

    print(f"\n  ajustou o TREINO       : {len(ajust):3d}/400  "
          f"({time.time()-t0:.0f}s)")
    print(f"  sobreviveu ao IMUNE    : {len(imunes):3d}/400")
    print(f"  passou no TESTE        : {len(sem):3d}/400   "
          f"precisao {len(sem)/max(len(ajust),1):.1%}")
    print(f"\n  familias: {dict(Counter(sem.values()).most_common())}")
    novas = sorted(set(sem) - JA)
    print(f"  tarefas: {sorted(sem)}")
    print(f"  INEDITAS: {len(novas)}  {novas}")

    print("\n" + "=" * 76)
    if not ajust:
        print("E0 FALHOU: nenhuma familia disparou.")
        return
    if len(novas) >= 5:
        print("E1 · arranjo era o degrau.")
    elif len(novas) >= 2:
        print("E2 · ajuda, no patamar das cinco mudancas de representacao.")
    else:
        print("E3 · arranjo linear nao e a forma da composicao no ARC.")


if __name__ == "__main__":
    main()

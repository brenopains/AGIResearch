"""
novaphys.trajetoria — caminho viavel e escoamento
===================================================

DE ONDE VEM, E A PROVENIENCIA E A EVIDENCIA MAIS FORTE DA SESSAO
------------------------------------------------------------------
O usuario resolveu 14 das 38 tarefas do perfil "acrescenta uma cor, preserva o
resto" e relatou o criterio de colocacao de cada uma. Classificados:

    caminho viavel sem curva     tarefas 1, 2, 6, 10     4
    fluido / recipiente enchendo tarefas 4, 8, 9, 11, 13 5
    simetria / espelho           tarefas 3, 7, 14        3
    encaixe de formas            tarefa 5                1
    crescimento com colisao      tarefa 12               1

**NOVE DE CATORZE sao trajetoria** — caminho ou fluxo. Algo percorrendo o
espaco por uma rota viavel.

DUAS CORRECOES QUE ISSO IMPOS
-------------------------------
1. Depois da PRIMEIRA gravacao (tres tarefas numericas) eu registrei como dado
   negativo que "o modo de trajetoria nao e o padrao", porque ele nao apareceu
   em 26 minutos. Errado: aquelas tres eram abstratas, estas sao espaciais. O
   negativo era artefato da selecao que eu mesmo fiz.

2. `celobj.py` tinha um teste `entre_dois` — celulas entre dois objetos — e ele
   disparou UMA vez. O desenho do usuario mostra o motivo: as linhas dele sao
   DIAGONAIS, e meu `entre_dois` so checava mesma linha e mesma coluna.

POR QUE O CENSO NAO VIU
-------------------------
As assinaturas eram `novas_colam`, `novas_linha`, `novas_enchem`. Um caminho
diagonal entre dois objetos nao cola (so toca nas pontas), nao e linha inteira,
nao e regiao cercada. Passa reto pelas tres. O vocabulario do censo era CEGO
para trajetoria, e por isso o maior mecanismo que ele achou tinha 5 tarefas.

O QUE ENTRA
-------------
    CAMINHO   liga pares de celulas ocupadas alinhadas numa das OITO direcoes,
              com so fundo no meio. Variantes deduzidas: quais pares (qualquer
              um / so mesma cor), quais direcoes (orto / diagonal / todas), e a
              cor da pintura (constante ou a das pontas).

    ESCOA     de cada celula fonte, avanca numa direcao pelo fundo; quando
              bloqueada de frente, tenta as duas diagonais a frente. E a lava
              caindo do relato: "tracando caminhos possiveis".

    INUNDA    a partir das celulas fonte, alaga o fundo alcancavel — o copo
              enchendo.

LIMITE DECLARADO
------------------
O relato descreve algo mais fino que isto: "o sovaco da cruz precisa estar na
mesma direcao" — uma AFORDANCIA DIRECIONAL da forma, nao uma relacao
posicional. A cruz tem quatro cantos concavos e o caminho so sai pelo que
aponta para o alvo. Isso nao esta implementado aqui: implemento a versao
posicional (alinhamento em 8 direcoes) e a afordancia fica de fora.

Se a versao posicional render pouco, a afordancia e o proximo alvo nomeado —
e isso ja e informacao.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  T0  PRE-CONDICAO. Alguma familia tem que disparar.

  T1  >= 8 inedidas -> trajetoria era o degrau, e 64% do subconjunto amostrado
                       pelo usuario se traduz em cobertura
  T2  3 a 7         -> ajuda, no patamar das mudancas de representacao
  T3  <= 2          -> a versao POSICIONAL nao basta; e a afordancia
                       direcional que carrega a familia

MINHA CALIBRAGEM: vinte e oito previsoes, DEZ certas, as dezoito erradas todas
otimistas. Mas o padrao de origem e mais informativo que o de numero: o que
veio de MEDIDA entregou (+2, +3, +3), o que veio de palpite meu deu zero.

Esta vem da evidencia mais direta que ja tive — nao um censo estrutural, e sim
o criterio verbalizado por quem resolve. PREVISAO: T2, 3 a 7. Nao aposto em T1
porque a afordancia esta de fora, e ela aparece explicitamente em 2 dos 4
relatos de caminho.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL

OITO = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTO = OITO[:4]
DIAG = OITO[4:]


def caminhos(g: Grid, dirs, so_mesma_cor: bool) -> Dict[Tuple[int, int], int]:
    """
    Celulas de fundo que ficam ENTRE duas ocupadas alinhadas numa das
    direcoes dadas, com so fundo no meio. Devolve celula -> cor da ponta.

    A versao anterior (`celobj.entre_dois`) so olhava linha e coluna. As
    diagonais eram o que faltava, e sao o que o desenho do usuario mostra.
    """
    bg = fundo(g)
    h, w = dims(g)
    fora: Dict[Tuple[int, int], int] = {}
    for i in range(h):
        for j in range(w):
            if g[i][j] == bg:
                continue
            for dy, dx in dirs:
                y, x = i + dy, j + dx
                meio = []
                while 0 <= y < h and 0 <= x < w and g[y][x] == bg:
                    meio.append((y, x))
                    y, x = y + dy, x + dx
                if not meio or not (0 <= y < h and 0 <= x < w):
                    continue
                if so_mesma_cor and g[y][x] != g[i][j]:
                    continue
                for p in meio:
                    fora[p] = g[i][j]
    return fora


def escoa(g: Grid, fonte: int, d: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """
    De cada celula da cor fonte, avanca na direcao; bloqueada de frente, tenta
    as duas diagonais a frente. E a lava caindo do relato.
    """
    bg = fundo(g)
    h, w = dims(g)
    dy, dx = d
    lados = ((dy, dx - 1), (dy, dx + 1)) if dy else ((dy - 1, dx), (dy + 1, dx))
    vis: Set[Tuple[int, int]] = set()
    fila = deque((i, j) for i in range(h) for j in range(w)
                 if g[i][j] == fonte)
    while fila:
        y, x = fila.popleft()
        for ey, ex in (d,) + lados:
            ny, nx = y + ey, x + ex
            if (0 <= ny < h and 0 <= nx < w and g[ny][nx] == bg
                    and (ny, nx) not in vis):
                vis.add((ny, nx))
                fila.append((ny, nx))
    return vis


def inunda(g: Grid, fonte: int) -> Set[Tuple[int, int]]:
    """Alaga o fundo alcancavel a partir das celulas da cor fonte."""
    bg = fundo(g)
    h, w = dims(g)
    vis: Set[Tuple[int, int]] = set()
    fila = deque((i, j) for i in range(h) for j in range(w)
                 if g[i][j] == fonte)
    while fila:
        y, x = fila.popleft()
        for dy, dx in ORTO:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and g[ny][nx] == bg
                    and (ny, nx) not in vis):
                vis.add((ny, nx))
                fila.append((ny, nx))
    return vis


DIRS_NOME = {"orto": ORTO, "diag": DIAG, "oito": OITO}


def deduzir(treino):
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    alvos, cores = [], set()
    for a, b in treino:
        bg = fundo(a)
        h, w = dims(a)
        if any(a[i][j] != bg and a[i][j] != b[i][j]
               for i in range(h) for j in range(w)):
            return None                        # PRESERVA: nada pode mudar
        p = {(i, j): b[i][j] for i in range(h) for j in range(w)
             if a[i][j] == bg and b[i][j] != bg}
        if not p:
            return None
        alvos.append(p)
        cores |= set(p.values())

    # CAMINHO
    for nome, dirs in DIRS_NOME.items():
        for mc in (True, False):
            cs = [caminhos(a, dirs, mc) for a, _b in treino]
            if any(set(c) != set(al) for c, al in zip(cs, alvos)):
                continue
            if all(all(al[p] == c[p] for p in al) for c, al in zip(cs, alvos)):
                return ("caminho", nome, mc, None)
            if len(cores) == 1:
                return ("caminho", nome, mc, next(iter(cores)))

    if len(cores) != 1:
        return None
    cor = next(iter(cores))
    fontes = sorted({v for a, _b in treino for l in a for v in l
                     if v != fundo(a)})

    # ESCOA
    for f in fontes:
        for dn, d in (("baixo", (1, 0)), ("cima", (-1, 0)),
                      ("dir", (0, 1)), ("esq", (0, -1))):
            if all(escoa(a, f, d) == set(al) for (a, _b), al in
                   zip(treino, alvos)):
                return ("escoa", f, dn, cor)

    # INUNDA
    for f in fontes:
        if all(inunda(a, f) == set(al) for (a, _b), al in zip(treino, alvos)):
            return ("inunda", f, None, cor)
    return None


def aplicar(regra, g: Grid) -> Optional[Grid]:
    try:
        t = regra[0]
        fora = [list(l) for l in g]
        if t == "caminho":
            c = caminhos(g, DIRS_NOME[regra[1]], regra[2])
            for (i, j), cor in c.items():
                fora[i][j] = regra[3] if regra[3] is not None else cor
        elif t == "escoa":
            d = {"baixo": (1, 0), "cima": (-1, 0),
                 "dir": (0, 1), "esq": (0, -1)}[regra[2]]
            for (i, j) in escoa(g, regra[1], d):
                fora[i][j] = regra[3]
        else:
            for (i, j) in inunda(g, regra[1]):
                fora[i][j] = regra[3]
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
      "ccd554ac", "cd3c21df"}


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("TRAJETORIA — caminho viavel e escoamento")
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
    print(f"\n  familias: {dict(Counter(sem.values()).most_common())}")
    novas = sorted(set(sem) - JA)
    print(f"  tarefas: {sorted(sem)}")
    print(f"  INEDITAS: {len(novas)}  {novas}")
    print(f"\n  55059096 (a que ele desenhou) resolvida: "
          f"{'55059096' in sem}")

    print("\n" + "=" * 76)
    if not ajust:
        print("T0 FALHOU: nenhuma familia disparou.")
        return
    if len(novas) >= 8:
        print("T1 · trajetoria era o degrau.")
    elif len(novas) >= 3:
        print("T2 · ajuda, no patamar das mudancas de representacao.")
    else:
        print("T3 · a versao POSICIONAL nao basta. E a AFORDANCIA DIRECIONAL")
        print("     ('o sovaco da cruz na mesma direcao') que carrega a")
        print("     familia, e ela nao esta implementada aqui.")


if __name__ == "__main__":
    main()

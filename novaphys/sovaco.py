"""
novaphys.sovaco — afordancia direcional: o caminho sai pelo canto concavo
==========================================================================

DE ONDE VEM
-------------
Relato do usuario resolvendo `55059096`, primeira das 38:

    "sao os que se apontam e tem um caminho possivel, o caminho possivel e
     aquele que nao faz curva, por isso as vezes um ou mais nao se conecta,
     porque ele nao esta na direcao do SOVACO ou do canto da cruz. Ambos
     sovacos das cruzes precisam estar na mesma direcao."

`trajetoria.py` implementou a versao POSICIONAL — ligar quaisquer duas celulas
ocupadas alinhadas em uma das oito direcoes — e deu T3, zero inedidas, sem
resolver justamente a tarefa que ele desenhou.

O motivo, escrito antes de rodar e confirmado: a versao posicional GERA DEMAIS.
Numa cruz, dezenas de celulas estao alinhadas com dezenas de outras.

A regra dele e muito mais restrita, e e sobre a FORMA e nao sobre a posicao: o
caminho so sai pelo canto concavo, e so quando os dois cantos se apontam.

A FORMALIZACAO
----------------
    SOVACO = celula de FUNDO com dois ou mais vizinhos ortogonais OCUPADOS.
    DIRECAO DE SAIDA = a diagonal oposta a esses vizinhos.

Numa cruz, os quatro cantos diagonais adjacentes ao centro sao sovacos, e cada
um aponta para fora em uma diagonal. O caminho anda nessa direcao pelo fundo
ate encontrar algo.

Testado a mao antes de escrever o arquivo, no exemplo 1 de `55059096`:
7 celulas pintadas, 0 faltando, 0 sobrando. Exato de primeira.

VARIANTES DEDUZIDAS
---------------------
    terminacao   o caminho vale se termina em qualquer celula ocupada, ou so
                 se termina em outro SOVACO, ou so se o sovaco de destino
                 aponta de VOLTA (mutuo — "ambos na mesma direcao")
    cor          constante lida do diff, ou a cor da ponta de origem

Nada e sorteado: as variantes sao poucas e a escolha e por reproducao exata do
treino.

O QUE ISTO CONFIRMA SOBRE O METODO
------------------------------------
O censo estrutural nao viu esta familia: `novas_colam` exige adjacencia em toda
celula (um caminho so toca nas pontas), `novas_linha` exige linha inteira,
`novas_enchem` exige regiao cercada. Passa reto pelas tres.

O relato verbal de quem resolve viu em uma frase. E o padrao da sessao inteira:
o que veio de MEDIDA entregou (+2, +3, +3); o que veio de palpite meu deu zero;
e a fonte mais fina de todas foi a voz do usuario descrevendo o criterio.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  V0  PRE-CONDICAO. `55059096` tem que ser resolvida. Ela foi verificada a mao
      no exemplo 1; se o motor completo nao pegar, ha erro de implementacao e
      nao resultado.

  V1  >= 5 inedidas -> afordancia direcional era o degrau
  V2  2 a 4         -> ajuda, no patamar das mudancas de representacao
  V3  <= 1          -> a regra vale para a tarefa dela e nao generaliza

MINHA CALIBRAGEM: vinte e nove previsoes, DEZ certas, as dezenove erradas todas
otimistas.

PREVISAO: V2, 2 a 4. O relato do usuario tem 4 tarefas de caminho em 14 (29%),
mas duas delas mencionam a afordancia e duas falam so de "nao fazer curva" —
entao nem todas as de caminho sao de sovaco. Sobre 38 tarefas, 29% seriam ~11,
mas so parte e afordancia e o motor exige reproducao exata.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL

ORTO = ((-1, 0), (1, 0), (0, -1), (0, 1))


def sovacos(g: Grid, bg: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Canto concavo: celula de fundo com 2+ vizinhos ortogonais ocupados.
    A saida e a diagonal OPOSTA a soma dos vizinhos ocupados — o sentido para
    onde a "virilha" abre.
    """
    h, w = dims(g)
    fora: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i in range(h):
        for j in range(w):
            if g[i][j] != bg:
                continue
            oc = [(dy, dx) for dy, dx in ORTO
                  if 0 <= i + dy < h and 0 <= j + dx < w
                  and g[i + dy][j + dx] != bg]
            if len(oc) < 2:
                continue
            dy = -sum(d[0] for d in oc)
            dx = -sum(d[1] for d in oc)
            if dy and dx:
                fora[(i, j)] = (dy // abs(dy), dx // abs(dx))
    return fora


def tracar(g: Grid, term: str) -> Dict[Tuple[int, int], int]:
    """
    De cada sovaco, anda na direcao dele pelo fundo. O caminho so e pintado se
    a terminacao satisfaz o criterio. Devolve celula -> cor da origem.
    """
    bg = fundo(g)
    h, w = dims(g)
    sv = sovacos(g, bg)
    fora: Dict[Tuple[int, int], int] = {}
    for (i, j), (dy, dx) in sv.items():
        y, x, cam = i, j, []
        while 0 <= y < h and 0 <= x < w and g[y][x] == bg:
            cam.append((y, x))
            y, x = y + dy, x + dx
        if not cam:
            continue
        fim = cam[-1]
        ok = False
        if 0 <= y < h and 0 <= x < w and g[y][x] != bg:
            ok = term == "ocupada"                 # bateu num objeto
        if not ok and term in ("sovaco", "mutuo"):
            # a ultima celula do caminho e ela mesma um sovaco?
            alvo = sv.get(fim)
            if alvo is not None and fim != (i, j):
                ok = (term == "sovaco"
                      or alvo == (-dy, -dx))       # aponta de volta
        if not ok:
            continue
        cor = None
        for ddy, ddx in ORTO:
            ny, nx = i + ddy, j + ddx
            if 0 <= ny < h and 0 <= nx < w and g[ny][nx] != bg:
                cor = g[ny][nx]
                break
        for p in cam:
            fora[p] = cor if cor is not None else 1
    return fora


def deduzir(treino):
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    alvos, cores = [], set()
    for a, b in treino:
        bg = fundo(a)
        h, w = dims(a)
        if any(a[i][j] != bg and a[i][j] != b[i][j]
               for i in range(h) for j in range(w)):
            return None
        p = {(i, j): b[i][j] for i in range(h) for j in range(w)
             if a[i][j] == bg and b[i][j] != bg}
        if not p:
            return None
        alvos.append(p)
        cores |= set(p.values())
    for term in ("mutuo", "sovaco", "ocupada"):
        cs = [tracar(a, term) for a, _b in treino]
        if any(set(c) != set(al) for c, al in zip(cs, alvos)):
            continue
        if all(all(al[p] == c[p] for p in al) for c, al in zip(cs, alvos)):
            return ("sovaco", term, None)
        if len(cores) == 1:
            return ("sovaco", term, next(iter(cores)))
    return None


def aplicar(regra, g: Grid) -> Optional[Grid]:
    try:
        _t, term, cor = regra
        fora = [list(l) for l in g]
        for (i, j), c in tracar(g, term).items():
            fora[i][j] = cor if cor is not None else c
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
        return nome, True, imune(tr), ok, r[1]
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
    print("SOVACO — o caminho sai pelo canto concavo")
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
    print(f"\nV0 · `55059096` resolvida: {'55059096' in sem}")
    print(f"\n  terminacoes: {dict(Counter(sem.values()).most_common())}")
    novas = sorted(set(sem) - JA)
    print(f"  tarefas: {sorted(sem)}")
    print(f"  INEDITAS: {len(novas)}  {novas}")

    print("\n" + "=" * 76)
    if "55059096" not in sem:
        print("V0 FALHOU: a tarefa verificada a mao nao passou. E erro de")
        print("     implementacao, nao resultado.")
    if len(novas) >= 5:
        print("V1 · afordancia direcional era o degrau.")
    elif len(novas) >= 2:
        print("V2 · ajuda, no patamar das mudancas de representacao.")
    else:
        print("V3 · a regra vale para a tarefa dela e nao generaliza.")


if __name__ == "__main__":
    main()

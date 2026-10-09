"""
novaphys.decomposicao — os custos somam em vez de multiplicar?
================================================================

A UNICA DIRECAO QUE ATACA O EXPOENTE
--------------------------------------
`guia.py` fechou a questao da ordem: os tres bracos (largura cega,
melhor-primeiro por estrutura, melhor-primeiro por pixel) resolveram o MESMO
conjunto de 14 tarefas, com custo mediano identico de 21 candidatos. As
tarefas que resolvemos sao achadas de imediato; nao ha orcamento a economizar.

E o orcamento de 14.424 e exatamente 24 + 24² + 24³ — a profundidade 3 e
enumerada por completo. Logo as outras 386 tarefas nao tem solucao NENHUMA
ate 3 passos. Nao e busca ruim: e alcance.

Ampliar vocabulario ja foi medido (+1 tarefa, `eter.py`). Aprofundar e
inviavel: o passo 4 custa 24× o orcamento inteiro, o passo 5 custa 576×.

Sobra uma coisa: fazer os custos SOMAREM.

    tarefa de 6 passos, monolitica     24^6  = 191.102.976
    duas partes de 3 passos cada    2 × 24^3 =      27.648

Isso e 7000× mais barato, e nao depende de buscar melhor — depende de a tarefa
SE PARTIR. A pergunta empirica, que nunca foi feita neste projeto:

    quantas das 386 tarefas nao resolvidas admitem uma particao
    em que cada parte e resolvivel em <= 3 passos?

DECOMPOSITORES, TODOS POSICIONAIS DE PROPOSITO
------------------------------------------------
    paineis   corta por linhas/colunas inteiras de cor unica (separadores)
    linhas    cada linha da grade e uma parte
    colunas   cada coluna e uma parte
    blocos    corta em blocos iguais quando as dimensoes permitem
    objetos   componentes 4-conexos, casados por caixa delimitadora

Os quatro primeiros sao POSICIONAIS: a parte i da entrada corresponde sem
ambiguidade a parte i da saida. Isso importa porque o casamento entre partes e
onde um decompositor esperto poderia trapacear — escolhendo a correspondencia
que faz a conta fechar. Sem ambiguidade, nao ha essa porta.

`objetos` tem casamento por posicao e por isso e o mais fragil dos cinco; esta
incluido porque e o mais natural no ARC, mas seu resultado deve ser lido com
mais reserva que os outros.

DOIS REGIMES, E SO UM DELES E A HIPOTESE
------------------------------------------
    COMPARTILHADO  UM programa aplicado a TODAS as partes
                   "a mesma regra vale para cada objeto"
    POR PARTE      cada parte ganha o seu programa

`compartilhado` e a hipotese de verdade. `por_parte` e o teto de sobreajuste:
com programa livre por parte, uma grade de 30 linhas vira 30 mini-tarefas
independentes e quase tudo "resolve" no treino sem generalizar nada. Ele esta
aqui para MEDIR essa inflacao, nao para ser comemorado.

Por isso tudo e verificado nos pares de TESTE, nunca so no treino.

ORCAMENTO
-----------
Resolver k partes a profundidade d custa k vezes o custo de d. A soma de todas
as mini-buscas e cobrada do MESMO K = 14.424 dos outros experimentos. Se a
decomposicao vencer, tera vencido gastando o mesmo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  D0  PRE-CONDICAO. Os decompositores tem que efetivamente partir as tarefas:
      reportar quantas admitem cada particao (entrada e saida partindo no
      mesmo numero de partes) ANTES de olhar solubilidade. Se quase nenhuma
      partir, nao ha o que medir e o resto nao deve ser lido.

  D1  COMPARTILHADO resolve >= 10 tarefas que a busca monolitica nao resolve,
      com o mesmo K
      -> os custos somam; e o caminho

  D2  ganho entre 3 e 9 tarefas
      -> decomposicao ajuda mas nao e a virada

  D3  ganho < 3
      -> as tarefas do ARC nao se partem desse jeito, e o expoente continua
         intacto

  D4  POR PARTE resolve muito mais que COMPARTILHADO no treino mas nao no
      teste
      -> confirma a inflacao esperada e mede o tamanho dela

MINHA PREVISAO, escrita antes de rodar: espero D2, ganho entre 3 e 8 tarefas.
A razao concreta e que os decompositores posicionais cobrem os casos de painel
e de simetria por linha, que sao uma fatia real mas pequena do ARC; a fatia
grande e "cada objeto se transforma segundo uma regra", e ali o casamento por
posicao vai falhar sempre que objetos se movem — que e justamente quando a
tarefa e interessante.

Se D1 aparecer eu vou conferir o braco POR PARTE antes de acreditar, porque
inflacao por sobreajuste e o modo de falha obvio deste desenho.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import (Grid, DICIONARIO, PRIMITIVAS, ajustar_mapa,
                          aplicar_mapa, carregar, dims, fundo, resolver,
                          valida)
from novaphys.eter import PASTA_AVAL

K = 14424
NOMES = [n for n, _ in PRIMITIVAS]


# ══════════════════════════════════════════════ decompositores
def _sep_cols(g: Grid) -> List[int]:
    h, w = dims(g)
    return [j for j in range(w) if len({g[i][j] for i in range(h)}) == 1]


def _sep_rows(g: Grid) -> List[int]:
    return [i for i, l in enumerate(g) if len(set(l)) == 1]


def d_paineis(g: Grid) -> Optional[List[Grid]]:
    """Corta por separadores inteiros de cor unica. Vertical, depois horizontal."""
    h, w = dims(g)
    cols = _sep_cols(g)
    if cols and len(cols) < w:
        partes, ini = [], 0
        for j in cols + [w]:
            if j > ini:
                p = tuple(l[ini:j] for l in g)
                if p and p[0]:
                    partes.append(p)
            ini = j + 1
        if len(partes) >= 2:
            return partes
    rows = _sep_rows(g)
    if rows and len(rows) < h:
        partes, ini = [], 0
        for i in rows + [h]:
            if i > ini:
                p = tuple(g[ini:i])
                if p and p[0]:
                    partes.append(p)
            ini = i + 1
        if len(partes) >= 2:
            return partes
    return None


def d_linhas(g: Grid) -> Optional[List[Grid]]:
    return [(l,) for l in g] if len(g) >= 2 else None


def d_colunas(g: Grid) -> Optional[List[Grid]]:
    h, w = dims(g)
    return [tuple((g[i][j],) for i in range(h)) for j in range(w)] \
        if w >= 2 else None


def d_blocos(g: Grid) -> Optional[List[Grid]]:
    """Corta em 2x2 blocos iguais quando as dimensoes sao pares."""
    h, w = dims(g)
    if h % 2 or w % 2 or h < 2 or w < 2:
        return None
    ph, pw = h // 2, w // 2
    return [tuple(l[b * pw:(b + 1) * pw] for l in g[a * ph:(a + 1) * ph])
            for a in range(2) for b in range(2)]


def d_objetos(g: Grid) -> Optional[List[Grid]]:
    """Componentes 4-conexos != fundo, em ordem de posicao, recortados."""
    h, w = dims(g)
    bg = fundo(g)
    visto = [[False] * w for _ in range(h)]
    comps = []
    for i in range(h):
        for j in range(w):
            if visto[i][j] or g[i][j] == bg:
                continue
            pilha, cel = [(i, j)], []
            visto[i][j] = True
            while pilha:
                y, x = pilha.pop()
                cel.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not visto[ny][nx]
                            and g[ny][nx] != bg):
                        visto[ny][nx] = True
                        pilha.append((ny, nx))
            comps.append(cel)
    if len(comps) < 2:
        return None
    comps.sort(key=lambda c: (min(y for y, _ in c), min(x for _, x in c)))
    fora = []
    for c in comps:
        ys = [y for y, _ in c]
        xs = [x for _, x in c]
        fora.append(tuple(tuple(g[y][x] for x in range(min(xs), max(xs) + 1))
                          for y in range(min(ys), max(ys) + 1)))
    return fora


DECOMP = [("paineis", d_paineis), ("linhas", d_linhas),
          ("colunas", d_colunas), ("blocos", d_blocos),
          ("objetos", d_objetos)]


# ══════════════════════════════════════════════ mini-busca
def _roda(ops: Sequence[str], g: Grid) -> Optional[Grid]:
    r: Optional[Grid] = g
    for n in ops:
        r = DICIONARIO[n](r)
        if not valida(r):
            return None
    return r


def resolver_partes(mini: Sequence[Tuple[Grid, Grid]], orc: int
                    ) -> Tuple[Optional[List[str]], Optional[Dict], int]:
    """Mesma busca do `arc.py`, com orcamento proprio. Devolve gasto."""
    import itertools
    gasto = 0
    for prof in range(0, 4):
        for seq in itertools.product(NOMES, repeat=prof):
            if gasto >= orc:
                return None, None, gasto
            gasto += 1
            saidas, ok = [], True
            for a, _b in mini:
                r = _roda(seq, a)
                if r is None:
                    ok = False
                    break
                saidas.append(r)
            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, mini)):
                return list(seq), {}, gasto
            m = ajustar_mapa(list(zip(saidas, [b for _, b in mini])))
            if m is not None and all(aplicar_mapa(r, m) == b
                                     for r, (_, b) in zip(saidas, mini)):
                return list(seq), m, gasto
    return None, None, gasto


def partir(fn, treino) -> Optional[List[List[Tuple[Grid, Grid]]]]:
    """
    Parte cada par. Exige o MESMO numero de partes na entrada e na saida, e o
    mesmo numero em todos os pares — senao a correspondencia seria escolha
    minha, e escolher correspondencia e onde este desenho poderia trapacear.
    """
    k = None
    por_indice: List[List[Tuple[Grid, Grid]]] = []
    for a, b in treino:
        pa, pb = fn(a), fn(b)
        if pa is None or pb is None or len(pa) != len(pb):
            return None
        if k is None:
            k = len(pa)
            por_indice = [[] for _ in range(k)]
        elif len(pa) != k:
            return None
        for i in range(k):
            por_indice[i].append((pa[i], pb[i]))
    return por_indice if k and k >= 2 else None


def tentar(fn, treino, teste, compartilhado: bool, orc: int
           ) -> Tuple[bool, int]:
    """True se resolve o treino E acerta o teste. Devolve o gasto."""
    partes = partir(fn, treino)
    if partes is None:
        return False, 0
    k = len(partes)

    if compartilhado:
        todos = [p for parte in partes for p in parte]
        ops, m, g = resolver_partes(todos, orc)
        if ops is None:
            return False, g
        progs = [(ops, m)] * k
        gasto = g
    else:
        progs, gasto = [], 0
        cada = max(1, orc // k)
        for parte in partes:
            ops, m, g = resolver_partes(parte, cada)
            gasto += g
            if ops is None:
                return False, gasto
            progs.append((ops, m))

    for a, b in teste:
        pa, pb = fn(a), fn(b)
        if pa is None or pb is None or len(pa) != k or len(pb) != k:
            return False, gasto
        for i in range(k):
            r = _roda(progs[i][0], pa[i])
            if r is None:
                return False, gasto
            if progs[i][1]:
                r = aplicar_mapa(r, progs[i][1])
            if r != pb[i]:
                return False, gasto
    return True, gasto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()

    print("=" * 76)
    print("DECOMPOSICAO — os custos somam em vez de multiplicar?")
    print("=" * 76)
    print(f"orcamento total por tarefa: {a.k}, o MESMO dos outros experimentos\n")

    tarefas = carregar(a.pasta)

    # ── linha de base monolitica
    base: Set[str] = set()
    for nome, tr, te in tarefas:
        p = resolver(tr, prof=3, orcamento=a.k)
        if p is not None and all(p.aplicar(x) == y for x, y in te):
            base.add(nome)
    print(f"linha de base monolitica: {len(base)}/{len(tarefas)} "
          f"({len(base)/len(tarefas):.2%})\n")

    # ── D0
    print("D0 · PRE-CONDICAO — quantas tarefas cada decompositor PARTE?")
    print("     (entrada e saida partindo no mesmo numero de partes)\n")
    parte_ok: Dict[str, Set[str]] = {}
    for rot, fn in DECOMP:
        s = {nome for nome, tr, _te in tarefas if partir(fn, tr) is not None}
        parte_ok[rot] = s
        print(f"     {rot:9s} {len(s):3d}/{len(tarefas)}  "
              f"({len(s)/len(tarefas):5.1%})")
    uniao_p = set().union(*parte_ok.values())
    print(f"     {'uniao':9s} {len(uniao_p):3d}/{len(tarefas)}  "
          f"({len(uniao_p)/len(tarefas):5.1%})")
    if len(uniao_p) < 40:
        print("\n     D0 · FALHA — quase nada se parte. Nao ha o que medir.")
        return
    print("\n     D0 · passa\n")

    # ── D1 / D4
    print(f"{'decompositor':14s} {'COMPARTILHADO':>15} {'POR PARTE':>12}"
          f"   novas vs base")
    print("-" * 66)
    t0 = time.time()
    res: Dict[str, Dict[str, Set[str]]] = {}
    for rot, fn in DECOMP:
        res[rot] = {"comp": set(), "parte": set()}
        for nome, tr, te in tarefas:
            if nome not in parte_ok[rot]:
                continue
            ok, _g = tentar(fn, tr, te, True, a.k)
            if ok:
                res[rot]["comp"].add(nome)
            ok, _g = tentar(fn, tr, te, False, a.k)
            if ok:
                res[rot]["parte"].add(nome)
        nv = res[rot]["comp"] - base
        print(f"{rot:14s} {len(res[rot]['comp']):>15} "
              f"{len(res[rot]['parte']):>12}   {len(nv):>3}  {sorted(nv)[:4]}")

    comp_u = set().union(*(r["comp"] for r in res.values()))
    parte_u = set().union(*(r["parte"] for r in res.values()))
    print("-" * 66)
    print(f"{'UNIAO':14s} {len(comp_u):>15} {len(parte_u):>12}")
    print(f"({time.time()-t0:.0f}s)\n")

    novas = comp_u - base
    total = base | comp_u
    print(f"base monolitica          : {len(base):3d}  "
          f"{len(base)/len(tarefas):.2%}")
    print(f"decomposicao COMPARTILHADA: {len(comp_u):3d}  "
          f"{len(comp_u)/len(tarefas):.2%}")
    print(f"tarefas NOVAS            : {len(novas):3d}  {sorted(novas)}")
    print(f"TOTAL base + decomposicao: {len(total):3d}  "
          f"{len(total)/len(tarefas):.2%}")

    print("\n" + "=" * 76)
    n = len(novas)
    if n >= 10:
        print("D1 · os custos somam. Conferir POR PARTE antes de comemorar.")
    elif n >= 3:
        print("D2 · decomposicao ajuda, mas nao e a virada.")
    else:
        print("D3 · as tarefas do ARC nao se partem assim. O expoente")
        print("     continua intacto.")
    infl = len(parte_u) - len(comp_u)
    print(f"\nD4 · inflacao do braco POR PARTE: {infl:+d} tarefas no teste.")
    print("     Programa livre por parte deveria inflar muito no treino; o")
    print("     que sobra no TESTE e o que realmente generaliza.")


if __name__ == "__main__":
    main()

"""
novaphys.abducao — ler a transformacao na diferenca, em vez de procurar
=========================================================================

O QUE NOSSO SISTEMA NUNCA FEZ
-------------------------------
Em nenhum experimento desta sessao o sistema COMPAROU a entrada com a saida
para deduzir algo. Ele sempre gerou um programa as cegas, aplicou, e perguntou
ao verificador "deu certo?". Nem `arc.py`, nem `eter.py`, nem `guia.py`, nem
`tipada.py` olham a diferenca.

Um humano faz o contrario, e faz primeiro. Olha os dois lados, ve o que mudou,
e a mudanca ja entrega o programa: "so as celulas vermelhas mudaram, e todas
viraram azuis" — nesse instante o programa esta determinado, sem busca
nenhuma.

Isso e ABDUCAO: resolver para os parametros a partir do dado, em vez de
enumerar candidatos. Custa O(celulas), nao O(24^d).

AS FAMILIAS ABDUZIDAS AQUI
----------------------------
Cada uma olha a diferenca e RESOLVE para o parametro. Nenhuma enumera.

  mapa_global      quais cores mudaram, e para quais. Se o mapa e consistente
                   em todos os pares, o programa esta pronto.
  recolor_objeto   se so celulas de UM objeto mudaram, qual propriedade o
                   distingue (maior, menor, cor rara, forma unica) e para que
                   cor foi. A propriedade e deduzida testando quais candidatas
                   selecionam exatamente o objeto que mudou.
  remocao          se a saida e a entrada menos alguns objetos, que propriedade
                   os removidos compartilham.
  recorte          se a saida aparece como sub-retangulo da entrada, ONDE — e
                   se essa posicao e descritivel (caixa de um objeto, de uma
                   cor, o maior).
  por_objeto       se cada objeto sofreu a MESMA transformacao no lugar.
  preenchimento    se as celulas que mudaram sao exatamente as regioes de fundo
                   fechadas, e para que cor foram.

O QUE E HONESTO E O QUE NAO E
-------------------------------
As familias sao escritas por mim — mesma critica de sempre, e ela procede. O
que muda de verdade nao e "menos trabalho manual", e o MECANISMO: aqui os
parametros sao RESOLVIDOS a partir dos dados, nao enumerados. Uma familia com
tres parametros de dez valores cada custa 1 avaliacao aqui e 1000 na busca.

E esse mecanismo e domain-general: vale para qualquer problema onde a entrada
e a saida podem ser comparadas termo a termo.

ORCAMENTO
-----------
A abducao gasta uma fracao minuscula do teto. Isso e o ponto, e por isso o
numero de candidatos avaliados e reportado junto: se ela resolver 8 tarefas
gastando 40 candidatos onde a busca gasta 14.424 e resolve 14, a comparacao
por tarefa/candidato e brutal e vale mais que o total absoluto.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. O custo medio tem que ficar abaixo de 200 candidatos por
      tarefa. Se passar disso a abducao virou busca disfarcada e a comparacao
      perde o sentido.

  A1  abducao resolve >= 8 tarefas que a monolitica nao resolve
      -> ler a diferenca alcanca o que buscar as cegas nao alcanca

  A2  entre 3 e 7 novas

  A3  menos de 3 novas
      -> as familias que escolhi nao cobrem o que falta

MINHA PREVISAO, escrita antes de rodar: A2, entre 3 e 7. `mapa_global` ja e
coberto pelo `ajustar_mapa` que roda no fim de cada candidato da busca atual,
entao o ganho tem que vir de `recolor_objeto`, `remocao` e `recorte`, que sao
genuinamente novas. Sete parece o teto realista com seis familias.

O numero que eu espero que impressione nao e o total: e o CUSTO. Se sair
abaixo de 100 candidatos por tarefa, a abducao e ~150x mais barata que a busca
por tarefa resolvida, e isso e o argumento de verdade.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import (Grid, carregar, dims, fundo,
                          resolver as resolver_mono, valida)
from novaphys.eter import PASTA_AVAL
from novaphys.tipada import _objs, _objs8, _por_cor, _caixa

Obj = Tuple[Tuple[int, int], ...]
Prog = Callable[[Grid], Optional[Grid]]


# ══════════════════════════════════════════════ propriedades de selecao
def _forma(o: Obj) -> tuple:
    ys = [y for y, _ in o]
    xs = [x for _, x in o]
    return tuple(sorted((y - min(ys), x - min(xs)) for y, x in o))


def seletores() -> List[Tuple[str, Callable]]:
    """
    Maneiras de apontar UM objeto sem dizer qual. A abducao testa qual delas
    seleciona exatamente o objeto que de fato mudou — e essa e a deducao.
    """
    def _cor_do(g, o):
        return g[o[0][0]][o[0][1]]

    return [
        ("maior", lambda g, L: max(L, key=len)),
        ("menor", lambda g, L: min(L, key=len)),
        ("formaUnica", lambda g, L: next(
            (o for o in L if sum(1 for p in L if _forma(p) == _forma(o)) == 1),
            None)),
        ("corUnica", lambda g, L: next(
            (o for o in L
             if sum(1 for p in L if _cor_do(g, p) == _cor_do(g, o)) == 1),
            None)),
        ("maisAlto", lambda g, L: min(L, key=lambda o: min(y for y, _ in o))),
        ("maisEsq", lambda g, L: min(L, key=lambda o: min(x for _, x in o))),
    ]


PARTICOES = [("obj4", _objs), ("obj8", _objs8), ("cor", _por_cor)]


# ══════════════════════════════════════════════ familias abduzidas
def ab_mapa_global(treino) -> Optional[Prog]:
    """Quais cores viraram quais. Resolve o mapa; nao o procura."""
    mapa: Dict[int, int] = {}
    for a, b in treino:
        if dims(a) != dims(b):
            return None
        for la, lb in zip(a, b):
            for ca, cb in zip(la, lb):
                if ca in mapa and mapa[ca] != cb:
                    return None
                mapa[ca] = cb
    if all(k == v for k, v in mapa.items()):
        return None
    return lambda g: tuple(tuple(mapa.get(c, c) for c in l) for l in g)


def ab_recolor_objeto(treino) -> Optional[Prog]:
    """
    Se so celulas de UM objeto mudaram, deduz QUAL propriedade o seleciona e
    para que cor foi. Ambos resolvidos a partir do dado.
    """
    for pn, pf in PARTICOES:
        for sn, sf in seletores():
            cor_alvo: Optional[int] = None
            ok = True
            for a, b in treino:
                if dims(a) != dims(b):
                    ok = False
                    break
                mud = {(i, j) for i, (la, lb) in enumerate(zip(a, b))
                       for j, (ca, cb) in enumerate(zip(la, lb)) if ca != cb}
                if not mud:
                    ok = False
                    break
                L = pf(a)
                if not L:
                    ok = False
                    break
                try:
                    o = sf(a, list(L))
                except Exception:
                    ok = False
                    break
                if o is None or set(o) != mud:
                    ok = False
                    break
                cores = {b[i][j] for i, j in mud}
                if len(cores) != 1:
                    ok = False
                    break
                c = cores.pop()
                if cor_alvo is not None and cor_alvo != c:
                    ok = False
                    break
                cor_alvo = c
            if ok and cor_alvo is not None:
                def prog(g, pf=pf, sf=sf, c=cor_alvo):
                    L = pf(g)
                    if not L:
                        return None
                    try:
                        o = sf(g, list(L))
                    except Exception:
                        return None
                    if o is None:
                        return None
                    s = set(o)
                    return tuple(tuple(c if (i, j) in s else v
                                       for j, v in enumerate(l))
                                 for i, l in enumerate(g))
                return prog
    return None


def ab_remocao(treino) -> Optional[Prog]:
    """Se a saida e a entrada menos UM objeto, qual propriedade o aponta."""
    for pn, pf in PARTICOES:
        for sn, sf in seletores():
            ok = True
            for a, b in treino:
                if dims(a) != dims(b):
                    ok = False
                    break
                bg = fundo(a)
                mud = {(i, j) for i, (la, lb) in enumerate(zip(a, b))
                       for j, (ca, cb) in enumerate(zip(la, lb)) if ca != cb}
                if not mud or any(b[i][j] != bg for i, j in mud):
                    ok = False
                    break
                L = pf(a)
                if not L:
                    ok = False
                    break
                try:
                    o = sf(a, list(L))
                except Exception:
                    ok = False
                    break
                if o is None or set(o) != mud:
                    ok = False
                    break
            if ok:
                def prog(g, pf=pf, sf=sf):
                    bg = fundo(g)
                    L = pf(g)
                    if not L:
                        return None
                    try:
                        o = sf(g, list(L))
                    except Exception:
                        return None
                    if o is None:
                        return None
                    s = set(o)
                    return tuple(tuple(bg if (i, j) in s else v
                                       for j, v in enumerate(l))
                                 for i, l in enumerate(g))
                return prog
    return None


def ab_recorte(treino) -> Optional[Prog]:
    """Se a saida e a caixa de um objeto selecionavel, qual seletor."""
    for pn, pf in PARTICOES:
        for sn, sf in seletores():
            for mascarar in (False, True):
                ok = True
                for a, b in treino:
                    L = pf(a)
                    if not L:
                        ok = False
                        break
                    try:
                        o = sf(a, list(L))
                    except Exception:
                        ok = False
                        break
                    if o is None:
                        ok = False
                        break
                    if mascarar:
                        bg = fundo(a)
                        ys = [y for y, _ in o]
                        xs = [x for _, x in o]
                        s = set(o)
                        r = tuple(tuple(a[y][x] if (y, x) in s else bg
                                        for x in range(min(xs), max(xs) + 1))
                                  for y in range(min(ys), max(ys) + 1))
                    else:
                        r = _caixa(a, o)
                    if r != b:
                        ok = False
                        break
                if ok:
                    def prog(g, pf=pf, sf=sf, msk=mascarar):
                        L = pf(g)
                        if not L:
                            return None
                        try:
                            o = sf(g, list(L))
                        except Exception:
                            return None
                        if o is None:
                            return None
                        if not msk:
                            return _caixa(g, o)
                        bg = fundo(g)
                        ys = [y for y, _ in o]
                        xs = [x for _, x in o]
                        s = set(o)
                        return tuple(
                            tuple(g[y][x] if (y, x) in s else bg
                                  for x in range(min(xs), max(xs) + 1))
                            for y in range(min(ys), max(ys) + 1))
                    return prog
    return None


def ab_preenchimento(treino) -> Optional[Prog]:
    """Se o que mudou sao exatamente as regioes de fundo fechadas."""
    def fechadas(g: Grid) -> Set[Tuple[int, int]]:
        h, w = dims(g)
        bg = fundo(g)
        livre = [[g[i][j] == bg for j in range(w)] for i in range(h)]
        pilha = [(i, j) for i in range(h) for j in range(w)
                 if livre[i][j] and (i in (0, h - 1) or j in (0, w - 1))]
        for i, j in pilha:
            livre[i][j] = False
        while pilha:
            y, x = pilha.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and livre[ny][nx]:
                    livre[ny][nx] = False
                    pilha.append((ny, nx))
        return {(i, j) for i in range(h) for j in range(w) if livre[i][j]}

    cor: Optional[int] = None
    for a, b in treino:
        if dims(a) != dims(b):
            return None
        mud = {(i, j) for i, (la, lb) in enumerate(zip(a, b))
               for j, (ca, cb) in enumerate(zip(la, lb)) if ca != cb}
        if not mud or mud != fechadas(a):
            return None
        cores = {b[i][j] for i, j in mud}
        if len(cores) != 1:
            return None
        c = cores.pop()
        if cor is not None and cor != c:
            return None
        cor = c
    if cor is None:
        return None

    def prog(g, c=cor):
        s = fechadas(g)
        return tuple(tuple(c if (i, j) in s else v for j, v in enumerate(l))
                     for i, l in enumerate(g))
    return prog


def ab_por_objeto(treino) -> Optional[Prog]:
    """
    Se CADA objeto virou a mesma cor, deduzida de uma propriedade dele
    (tamanho -> cor). O mapa tamanho->cor e resolvido, nao procurado.
    """
    mapa: Dict[int, int] = {}
    for a, b in treino:
        if dims(a) != dims(b):
            return None
        L = _objs(a)
        if not L or len(L) < 2:
            return None
        for o in L:
            cores = {b[i][j] for i, j in o}
            if len(cores) != 1:
                return None
            c = cores.pop()
            t = len(o)
            if t in mapa and mapa[t] != c:
                return None
            mapa[t] = c
    if len(set(mapa.values())) < 2:
        return None

    def prog(g, mapa=mapa):
        L = _objs(g)
        if not L:
            return None
        fora = [list(l) for l in g]
        for o in L:
            if len(o) not in mapa:
                return None
            c = mapa[len(o)]
            for i, j in o:
                fora[i][j] = c
        return tuple(tuple(l) for l in fora)
    return prog


FAMILIAS: List[Tuple[str, Callable]] = [
    ("mapa_global", ab_mapa_global),
    ("recolor_objeto", ab_recolor_objeto),
    ("remocao", ab_remocao),
    ("recorte", ab_recorte),
    ("preenchimento", ab_preenchimento),
    ("por_objeto", ab_por_objeto),
]


def abduzir(treino) -> Tuple[Optional[Prog], Optional[str], int]:
    """
    Roda as familias. O custo contado e o numero de PROGRAMAS COMPLETOS
    verificados contra os pares — a mesma moeda dos outros experimentos.
    Cada familia devolve no maximo um programa, entao o custo e minusculo.
    """
    custo = 0
    for nome, fn in FAMILIAS:
        try:
            p = fn(treino)
        except Exception:
            p = None
        if p is None:
            continue
        custo += 1
        if all(p(a) == b for a, b in treino):
            return p, nome, custo
    return None, None, custo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    a = ap.parse_args()

    print("=" * 76)
    print("ABDUCAO — ler a transformacao na diferenca, em vez de procurar")
    print("=" * 76)
    print(f"{len(FAMILIAS)} familias, {len(PARTICOES)} particoes, "
          f"{len(seletores())} seletores\n")

    tarefas = carregar(a.pasta)

    mono: Set[str] = set()
    for nome, tr, te in tarefas:
        p = resolver_mono(tr, prof=3, orcamento=14424)
        if p is not None and all(p.aplicar(x) == y for x, y in te):
            mono.add(nome)
    print(f"monolitica (14.424 candidatos/tarefa): {len(mono):3d}/400  "
          f"{len(mono)/len(tarefas):.2%}")

    t0 = time.time()
    ab: Set[str] = set()
    por_fam: Dict[str, int] = {n: 0 for n, _ in FAMILIAS}
    custo_total = 0
    quais: Dict[str, str] = {}
    for nome, tr, te in tarefas:
        p, fam, c = abduzir(tr)
        custo_total += c
        if p is None:
            continue
        try:
            if all(p(x) == y for x, y in te):
                ab.add(nome)
                por_fam[fam] += 1
                quais[nome] = fam
        except Exception:
            pass
    dt = time.time() - t0
    medio = custo_total / len(tarefas)
    print(f"abducao ({medio:.1f} candidatos/tarefa)        : {len(ab):3d}/400  "
          f"{len(ab)/len(tarefas):.2%}   ({dt:.0f}s)")

    print(f"\nA0 · custo medio {medio:.1f} candidatos/tarefa "
          f"{'· passa' if medio < 200 else '· FALHA, virou busca'}")
    print(f"     {14424/max(medio,0.01):.0f}x mais barato que a busca")

    print("\npor familia:")
    for n, _f in FAMILIAS:
        print(f"   {n:16s} {por_fam[n]:3d}")

    novas = ab - mono
    print(f"\nso a ABDUCAO resolve ({len(novas)}):")
    for n in sorted(novas):
        print(f"   {n}  [{quais[n]}]")
    print(f"so a monolitica ({len(mono - ab)}): {sorted(mono - ab)}")

    total = mono | ab
    print(f"\n{'monolitica':28s} {len(mono):3d}  {len(mono)/len(tarefas):.2%}")
    print(f"{'abducao':28s} {len(ab):3d}  {len(ab)/len(tarefas):.2%}")
    print(f"{'UNIAO':28s} {len(total):3d}  {len(total)/len(tarefas):.2%}")

    print("\n" + "=" * 76)
    if len(novas) >= 8:
        print("A1 · ler a diferenca alcanca o que buscar as cegas nao alcanca.")
    elif len(novas) >= 3:
        print("A2 · ganho real, dentro do esperado.")
    else:
        print("A3 · as familias que escolhi nao cobrem o que falta.")


if __name__ == "__main__":
    main()

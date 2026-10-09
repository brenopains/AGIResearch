"""
novaphys.mundo — primitivas nascidas de um mundo, testadas fora dele
======================================================================

A HIPOTESE, DELE E MINHA
--------------------------
O regresso de Solomonoff nao fecha por matematica: alguem tem que fornecer as
primeiras primitivas. A proposta e que ele feche por ANCORAGEM — um mundo
manipulavel de onde as primitivas saem, em vez de eu escreve-las.

Isso e testavel de forma barata e brutal. Nao precisa de um planeta:

    mundo minimo com objetos
        -> o agente age e observa (antes, depois)
        -> induz a regra a partir do delta
        -> a regra vira primitiva
        -> testa no ARC, que o mundo nunca viu

METRICA: uma primitiva nascida no mundo levanta o ARC acima de 16/400?

O QUE E INDUZIDO E O QUE E DADO — leia antes de acreditar no resultado
-----------------------------------------------------------------------
O extrator NAO ve o codigo da dinamica. Ele ve so pares (antes, depois) e
tenta reconstruir a operacao. Mas as FAMILIAS de hipotese sao minhas:

    translacao de objetos ate bloquear    (direcao e induzida)
    preenchimento de regiao fechada       (cor e induzida)
    remocao de objetos por propriedade    (a propriedade e induzida)

Entao o regresso NAO esta fechado — esta empurrado um nivel. O que o mundo
decide e QUAL operacao especifica existe; o que eu decido e o espaco de
operacoes concebiveis. Dizer outra coisa seria mentir.

O teste continua valendo porque a operacao especifica vem da experiencia, e
o ARC e um benchmark externo que nao compartilha nada com o mundo.

CONTRA APARELHAMENTO
----------------------
O mundo tem SEIS dinamicas, e algumas eu espero que sejam inuteis para o ARC
(recolorir tudo, embaralhar cores). Nao seleciono quais extrair. Se so as
uteis aparecerem, o resultado esta viciado por mim; se as inuteis tambem
forem extraidas e simplesmente nao ajudarem, o teste esta limpo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  W0  PRE-CONDICAO. O extrator tem que recuperar a dinamica correta a partir
      das transicoes, para as seis. Se nao recupera, nada abaixo se le.

  W1  ARC > 16/400 com as primitivas do mundo, ORCAMENTO CASADO
      -> primitiva ancorada em mundo transferiu para benchmark externo.
         Seria o resultado mais forte da sessao.

  W2  ARC = 16/400
      -> experiencia de mundo nao produz primitiva relevante ao ARC na escala
         mais barata; mundo maior nao esta indicado

  W3  ARC < 16/400
      -> as primitivas do mundo DILUEM a busca a orcamento fixo, exatamente
         como `gargalo.py` mediu (12pp de penalidade por molde ruim). Custo
         de vocabulario, e informativo.

MINHA PREVISAO: W2. Espero que `queda` e `preenche_fechado` sejam extraidas
corretamente e que nao resolvam nenhuma tarefa nova, porque as tarefas de
gravidade do ARC costumam precisar de direcao dependente do contexto, nao
de queda global. Aposto contra. Se der W1 com mais de 3 tarefas, procuro
vazamento antes de comemorar.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .arc import (Grid, PRIMITIVAS, DICIONARIO, Programa, ajustar_mapa,
                  aplicar_mapa, carregar, dims, fundo, valida)

AVALIACAO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/evaluation').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
SEMENTE = 20260814
LADO = 12
N_TRANSICOES = 30
DIRS = {"baixo": (1, 0), "cima": (-1, 0), "dir": (0, 1), "esq": (0, -1)}


# ══════════════════════════════════════════════ objetos como conjuntos
def componentes(g: Grid) -> List[Set[Tuple[int, int]]]:
    f = fundo(g)
    h, w = dims(g)
    visto = [[False] * w for _ in range(h)]
    fora = []
    for i in range(h):
        for j in range(w):
            if g[i][j] == f or visto[i][j]:
                continue
            pilha, cel = [(i, j)], set()
            visto[i][j] = True
            while pilha:
                a, b = pilha.pop()
                cel.add((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and not visto[x][y] \
                            and g[x][y] != f:
                        visto[x][y] = True
                        pilha.append((x, y))
            fora.append(cel)
    return fora


def regioes_fechadas(g: Grid) -> Set[Tuple[int, int]]:
    """Celulas de fundo que NAO alcancam a borda — contencao."""
    f = fundo(g)
    h, w = dims(g)
    alcanca = [[False] * w for _ in range(h)]
    pilha = []
    for i in range(h):
        for j in (0, w - 1):
            if g[i][j] == f and not alcanca[i][j]:
                alcanca[i][j] = True
                pilha.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if g[i][j] == f and not alcanca[i][j]:
                alcanca[i][j] = True
                pilha.append((i, j))
    while pilha:
        a, b = pilha.pop()
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = a + da, b + db
            if 0 <= x < h and 0 <= y < w and g[x][y] == f and not alcanca[x][y]:
                alcanca[x][y] = True
                pilha.append((x, y))
    return {(i, j) for i in range(h) for j in range(w)
            if g[i][j] == f and not alcanca[i][j]}


# ══════════════════════════════════════════════ dinamicas do mundo
def _mover_ate_bloquear(g: Grid, di: int, dj: int) -> Grid:
    f = fundo(g)
    h, w = dims(g)
    comps = componentes(g)
    # ordem: quem esta mais na direcao do movimento se move primeiro
    comps.sort(key=lambda c: -max(i * di + j * dj for i, j in c))
    ocupado = {}
    for c in comps:
        for (i, j) in c:
            ocupado[(i, j)] = g[i][j]
    novo = {}
    for c in comps:
        for (i, j) in c:
            del ocupado[(i, j)]
        k = 0
        while True:
            k += 1
            ok = True
            for (i, j) in c:
                a, b = i + di * k, j + dj * k
                if not (0 <= a < h and 0 <= b < w) or (a, b) in ocupado:
                    ok = False
                    break
            if not ok:
                k -= 1
                break
        for (i, j) in c:
            a, b = i + di * k, j + dj * k
            ocupado[(a, b)] = g[i][j]
            novo[(a, b)] = g[i][j]
    return tuple(tuple(novo.get((i, j), f) for j in range(w)) for i in range(h))


def d_queda(g: Grid) -> Grid:
    return _mover_ate_bloquear(g, 1, 0)


def d_empurra_dir(g: Grid) -> Grid:
    return _mover_ate_bloquear(g, 0, 1)


def d_preenche(g: Grid, cor: int = 4) -> Grid:
    r = regioes_fechadas(g)
    if not r:
        return g
    h, w = dims(g)
    return tuple(tuple(cor if (i, j) in r else g[i][j] for j in range(w))
                 for i in range(h))


def d_remove_solto(g: Grid) -> Grid:
    """Some com objetos que nao encostam na borda."""
    f = fundo(g)
    h, w = dims(g)
    fica = set()
    for c in componentes(g):
        if any(i in (0, h - 1) or j in (0, w - 1) for i, j in c):
            fica |= c
    return tuple(tuple(g[i][j] if (i, j) in fica else f for j in range(w))
                 for i in range(h))


def d_recolore_tudo(g: Grid) -> Grid:
    """Esperada INUTIL para o ARC. Esta aqui para o teste nao ficar viciado."""
    f = fundo(g)
    return tuple(tuple(7 if c != f else c for c in l) for l in g)


def d_embaralha_cor(g: Grid) -> Grid:
    """Tambem esperada inutil."""
    f = fundo(g)
    m = {c: ((c * 3 + 1) % 9) + 1 for c in range(10) if c != f}
    return tuple(tuple(m.get(c, c) for c in l) for l in g)


DINAMICAS: Dict[str, Callable[[Grid], Grid]] = {
    "queda": d_queda, "empurra_dir": d_empurra_dir,
    "preenche": d_preenche, "remove_solto": d_remove_solto,
    "recolore_tudo": d_recolore_tudo, "embaralha_cor": d_embaralha_cor,
}


def cenario(rng: random.Random) -> Grid:
    h = w = LADO
    m = [[0] * w for _ in range(h)]
    for _ in range(rng.randint(2, 5)):
        cor = rng.randint(1, 9)
        i0, j0 = rng.randrange(h - 3), rng.randrange(w - 3)
        for a in range(rng.randint(1, 3)):
            for b in range(rng.randint(1, 3)):
                m[i0 + a][j0 + b] = cor
    return tuple(tuple(l) for l in m)


# ══════════════════════════════════════════════ o extrator
def _casa(pares, fn) -> bool:
    return all(fn(a) == b for a, b in pares)


def extrair(pares: Sequence[Tuple[Grid, Grid]]
            ) -> Optional[Tuple[str, Callable[[Grid], Grid]]]:
    """
    Ve so (antes, depois). Tenta reconstruir a operacao dentro das familias
    concebiveis, e devolve a primeira que reproduz TODAS as transicoes.
    Verificacao exata — nao ha ajuste parcial.
    """
    # familia 1: translacao ate bloquear, direcao induzida
    for nome, (di, dj) in DIRS.items():
        fn = (lambda di, dj: (lambda g: _mover_ate_bloquear(g, di, dj)))(di, dj)
        if _casa(pares, fn):
            return f"mover_{nome}", fn

    # familia 2: preenchimento de regiao fechada, cor induzida
    cores = set()
    for a, b in pares:
        r = regioes_fechadas(a)
        for (i, j) in r:
            cores.add(b[i][j])
    if len(cores) == 1:
        cor = cores.pop()
        fn = (lambda cor: (lambda g: d_preenche(g, cor)))(cor)
        if _casa(pares, fn):
            return f"preenche_{cor}", fn

    # familia 3: remocao por propriedade, propriedade induzida
    for prop, fn in (("solto", d_remove_solto),):
        if _casa(pares, fn):
            return f"remove_{prop}", fn

    # familia 4: recoloracao global, mapa induzido
    mapa = ajustar_mapa(list(pares))
    if mapa is not None:
        fn = (lambda m: (lambda g: aplicar_mapa(g, m)))(mapa)
        if _casa(pares, fn):
            # o rotulo tem que distinguir recoloracoes DIFERENTES: duas
            # dinamicas distintas caiam no mesmo nome e uma sobrescrevia a
            # outra no dicionario, dando 5 primitivas em vez de 6.
            sig = "".join(str(mapa[k]) for k in sorted(mapa))
            return f"recolore_{sig}", fn
    return None


# ══════════════════════════════════════════════ ARC com orcamento casado
def _resolver(treino, nomes, dic, orcamento, prof=3):
    import itertools
    n = 0
    for d in range(prof + 1):
        for ops in itertools.product(nomes, repeat=d):
            n += 1
            if n > orcamento:
                return None
            saidas, ok = [], True
            for a, b in treino:
                r = a
                for o in ops:
                    r = dic[o](r)
                    if not valida(r):
                        ok = False
                        break
                if not ok:
                    break
                saidas.append(r)
            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, treino)):
                return ops, None
            m = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
            if m is not None and all(aplicar_mapa(r, m) == b
                                     for r, (_, b) in zip(saidas, treino)):
                return ops, m
    return None


def rodar_arc(nomes, dic, orcamento) -> Tuple[int, List[str]]:
    av = carregar(AVALIACAO)
    ok, quais = 0, []
    for nome, tr, te in av:
        r = _resolver(tr, nomes, dic, orcamento)
        if r is None:
            continue
        ops, m = r
        bom = True
        for a, b in te:
            x = a
            for o in ops:
                x = dic[o](x)
                if not valida(x):
                    bom = False
                    break
            if not bom or (aplicar_mapa(x, m) if m else x) != b:
                bom = False
                break
        if bom:
            ok += 1
            quais.append(f"{nome}  [{'.'.join(ops) or 'id'}"
                         f"{' +recolore' if m else ''}]")
    return ok, quais


def relatorio(orcamento: int = 14425) -> None:
    rng = random.Random(SEMENTE)
    print("=" * 78)
    print("MUNDO — primitivas nascidas de um mundo, testadas fora dele")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  W0  o extrator recupera as 6 dinamicas a partir das transicoes")
    print("  W1  ARC > 16/400 -> primitiva de mundo transfere")
    print("  W2  ARC = 16/400 -> mundo nao da primitiva relevante")
    print("  W3  ARC < 16/400 -> vocabulario dilui a busca")
    print()
    print("o extrator NAO ve o codigo da dinamica, so pares (antes, depois).")
    print("as FAMILIAS de hipotese sao minhas: o regresso esta empurrado um")
    print("nivel, nao fechado.")

    print("\n" + "-" * 78)
    print(f"FASE 1 · extrair, {N_TRANSICOES} transicoes por dinamica")
    extraidas: Dict[str, Callable[[Grid], Grid]] = {}
    w0 = 0
    for nome, fn in DINAMICAS.items():
        pares = []
        while len(pares) < N_TRANSICOES:
            a = cenario(rng)
            b = fn(a)
            if b != a:
                pares.append((a, b))
        r = extrair(pares)
        if r is None:
            print(f"  {nome:<15} NAO recuperada")
            continue
        rot, f2 = r
        conf = all(f2(a) == b for a, b in pares)
        print(f"  {nome:<15} -> {rot:<20} {'confere' if conf else 'DIVERGE'}")
        if conf:
            w0 += 1
            extraidas[f"M_{rot}"] = f2

    print(f"\nW0 · {w0}/{len(DINAMICAS)} dinamicas recuperadas")
    if w0 < len(DINAMICAS):
        print("     PRE-CONDICAO REPROVADA — o extrator nao recupera tudo.")
        print("     Um resultado abaixo seria defeito do instrumento.")
        return
    print("     aprovada")

    base_nomes = [n for n, _ in PRIMITIVAS]
    base_dic = dict(DICIONARIO)
    novo_nomes = base_nomes + sorted(extraidas)
    novo_dic = dict(base_dic)
    novo_dic.update(extraidas)

    print("\n" + "-" * 78)
    print(f"FASE 2 · ARC, orcamento casado em {orcamento} programas")
    print(f"  base   {len(base_nomes)} primitivas")
    print(f"  mundo  {len(novo_nomes)} primitivas (+{len(extraidas)} do mundo)")

    t0 = time.time()
    b_ok, b_quais = rodar_arc(base_nomes, base_dic, orcamento)
    print(f"\n  base    {b_ok}/400 = {b_ok/400:.1%}   ({time.time()-t0:.0f}s)")
    t0 = time.time()
    m_ok, m_quais = rodar_arc(novo_nomes, novo_dic, orcamento)
    print(f"  mundo   {m_ok}/400 = {m_ok/400:.1%}   ({time.time()-t0:.0f}s)")

    novas = [x for x in m_quais if x not in b_quais]
    perdidas = [x for x in b_quais if x not in m_quais]
    print(f"\n  ganhou {len(novas)} · perdeu {len(perdidas)}")
    for x in novas:
        print("   + " + x)
    for x in perdidas:
        print("   - " + x)

    print("\n" + "=" * 78)
    ref = 16
    total = m_ok + 2      # +2 da inducao do delta, familia disjunta
    print(f"referencia atual: busca {b_ok} + inducao 2 = {b_ok+2}/400")
    print(f"com mundo:        busca {m_ok} + inducao 2 = {total}/400")
    print()
    if total > ref:
        print(f"W1 · TRANSFERIU. Primitiva nascida num mundo manipulavel levantou")
        print(f"     um benchmark externo que o mundo nunca viu: {total}/400.")
    elif total == ref:
        print("W2 · NAO TRANSFERIU. O mundo nao produziu primitiva relevante ao")
        print("     ARC na escala mais barata. Mundo maior nao esta indicado por")
        print("     este resultado.")
    else:
        print(f"W3 · DILUIU ({total}/400 contra {ref}). As primitivas do mundo")
        print("     ocupam orcamento sem pagar — o custo de vocabulario que")
        print("     `gargalo.py` mediu em 12pp por molde ruim.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orcamento", type=int, default=14425)
    a = ap.parse_args()
    relatorio(orcamento=a.orcamento)

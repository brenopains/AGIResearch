"""
novaphys.arc_combinadores — a forma do espaco, nao o tamanho da busca
======================================================================

O QUE OS DADOS APONTARAM
--------------------------
    prof 3, orcamento = espaco (exaustivo)     base 14/400   guiado 14/400
    prof 4, orcamento = 14% do espaco          base 14/400   guiado 15/400

Ordenar 14% de um espaco de CADEIAS LINEARES comprou UMA tarefa, com uma rede
que preve primitivas a 10,6x o acaso. O guia nao e fraco — o espaco tem a
forma errada.

A evidencia esta nas proprias solucoes:

    pOU +recolore       x5
    pXOR +recolore      x2
    rot180.simH.simV    x2
    rot90.pOU.rot270    x1

**8 das 14 usam `pOU` ou `pXOR`** — as unicas primitivas onde RAMIFICACAO esta
escondida dentro de um simbolo que eu escrevi a mao. `combina(f(x), g(x))`
disfarcado de primitiva carrega a maioria das vitorias. E a familia "faca isso
em cada objeto" e ZERO de 14, apesar de ser uma fatia grande do ARC.

A MUDANCA E DE FORMA, NAO DE LISTA
------------------------------------
    P ::= cadeia(prims...)              o que existia
        | combina(op, P, P)             ramificacao, agora generica
        | por_objeto(P)                 aplica a cada peca e remonta

`pE`, `pOU`, `pXOR`, `pDIF` SAEM da lista de primitivas. Elas viram casos
particulares: `pXOR` = `combina(xor, painelA, painelB)`. Isso e teste, nao
conveniencia — se a generalizacao nao reproduzir o que a versao manual fazia,
a generalizacao esta errada e o numero cai.

E libera o que eu nunca escrevi: `combina(xor, id, rot180)` e diferenca de
simetria; `por_objeto(rot90)` gira cada peca no lugar.

ORCAMENTO CASADO COM O EXPERIMENTO ANTERIOR
---------------------------------------------
50.000 programas por tarefa — o mesmo da rodada de profundidade 4, onde a
forca bruta em cadeias deu 14/400 e o guiado 15/400. Mesma verificacao
exata, mesmas 400 tarefas de avaliacao, mesmo custo. So a FORMA do espaco
muda.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  K0  PRE-CONDICAO. A gramatica nova tem que reproduzir as tarefas de painel
      que a versao manual resolvia (`combina(ou|xor, painelA, painelB)`).
      Se perder essas, a generalizacao esta quebrada e o resto nao se le.

  K1  >= 18/400 (4 tarefas a mais, ~1pp) com o MESMO orcamento
      -> a forma do espaco era o gargalo

  K2  ~ 14/400
      -> a forma NAO era o gargalo; minha leitura das 8-de-14 estava errada
         e a hipotese das cadeias lineares cai

  K3  se `por_objeto` resolver tarefa que NENHUMA cadeia resolve
      -> confirma diretamente a familia ausente, independente do total

MINHA PREVISAO, escrita antes de rodar: espero K1 fraco, algo entre 16 e 22.
`por_objeto` deve trazer poucas porque so cobre "transforma cada peca no
lugar", e muita tarefa de objeto exige mover, contar ou comparar pecas — que
esta gramatica ainda nao expressa. Se passar de 30 vou procurar vazamento
antes de comemorar.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .arc import (Grid, PRIMITIVAS, DICIONARIO, ajustar_mapa, aplicar_mapa,
                  carregar, dims, fundo, valida)

AVALIACAO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/evaluation').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

# `pE/pOU/pXOR/pDIF` saem: viram casos de `combina`. `painelA/painelB` ficam.
FORA = {"pE", "pOU", "pXOR", "pDIF"}
BASE = [n for n, _ in PRIMITIVAS if n not in FORA]
OPS_COMB = ("e", "ou", "xor", "dif")


# ══════════════════════════════════════════════ objetos
def objetos(g: Grid) -> List[Tuple[int, int, int, int]]:
    """Caixas dos componentes conexos que nao sao fundo (4-vizinhanca)."""
    f = fundo(g)
    h, w = dims(g)
    visto = [[False] * w for _ in range(h)]
    caixas = []
    for i in range(h):
        for j in range(w):
            if g[i][j] == f or visto[i][j]:
                continue
            pilha, cel = [(i, j)], []
            visto[i][j] = True
            while pilha:
                a, b = pilha.pop()
                cel.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and not visto[x][y] \
                            and g[x][y] != f:
                        visto[x][y] = True
                        pilha.append((x, y))
            li = [a for a, _ in cel]
            lj = [b for _, b in cel]
            caixas.append((min(li), min(lj), max(li), max(lj)))
    return caixas


# ══════════════════════════════════════════════ o programa, agora arvore
class Prog:
    """
    Tres formas. `aplicar` e total: devolve None em vez de levantar, porque a
    busca avalia dezenas de milhares de candidatos invalidos por tarefa.
    """
    __slots__ = ("tipo", "ops", "op", "a", "b", "fora", "mapa")

    def __init__(self, tipo: str, ops=(), op="", a=None, b=None, fora=(),
                 mapa=None):
        self.tipo = tipo          # 'cadeia' | 'combina' | 'objeto'
        self.ops = tuple(ops)
        self.op = op
        self.a = a
        self.b = b
        self.fora = tuple(fora)   # cadeia externa aplicada ao final
        self.mapa = mapa          # recoloracao ajustada do dado, se houver

    # ---- avaliacao
    def _cadeia(self, g: Grid, ops: Sequence[str]) -> Optional[Grid]:
        for n in ops:
            g = DICIONARIO[n](g)          # type: ignore[assignment,arg-type]
            if not valida(g):
                return None
        return g

    def _nucleo(self, g: Grid) -> Optional[Grid]:
        if self.tipo == "cadeia":
            return self._cadeia(g, self.ops)

        if self.tipo == "combina":
            x = self.a.aplicar(g)
            y = self.b.aplicar(g)
            if x is None or y is None or dims(x) != dims(y):
                return None
            f = fundo(g)
            saida = []
            for lx, ly in zip(x, y):
                linha = []
                for u, v in zip(lx, ly):
                    pu, pv = u != f, v != f
                    if self.op == "e":
                        t = pu and pv
                    elif self.op == "ou":
                        t = pu or pv
                    elif self.op == "xor":
                        t = pu != pv
                    else:
                        t = pu and not pv
                    linha.append(1 if t else 0)
                saida.append(tuple(linha))
            return tuple(saida) if saida and saida[0] else None

        # 'objeto': transforma cada peca NO LUGAR
        caixas = objetos(g)
        if not caixas or len(caixas) > 40:
            return None
        f = fundo(g)
        m = [list(l) for l in g]
        mudou = False
        for (i0, j0, i1, j1) in caixas:
            sub = tuple(tuple(g[i][j] for j in range(j0, j1 + 1))
                        for i in range(i0, i1 + 1))
            r = self._cadeia(sub, self.ops)
            if r is None or dims(r) != dims(sub):
                continue                      # so cabe se preservar a caixa
            for a in range(i1 - i0 + 1):
                for b in range(j1 - j0 + 1):
                    if r[a][b] != f:
                        m[i0 + a][j0 + b] = r[a][b]
            mudou = True
        return tuple(tuple(l) for l in m) if mudou else None

    def aplicar(self, g: Grid) -> Optional[Grid]:
        r = self._nucleo(g)
        if r is None or not valida(r):
            return None
        if self.fora:
            r = self._cadeia(r, self.fora)
            if r is None:
                return None
        return r

    def __str__(self) -> str:
        if self.tipo == "cadeia":
            n = ".".join(self.ops) or "id"
        elif self.tipo == "combina":
            n = f"combina[{self.op}]({self.a}, {self.b})"
        else:
            n = f"por_objeto({'.'.join(self.ops)})"
        if self.fora:
            n = f"{n} > {'.'.join(self.fora)}"
        return n + (" +recolore" if self.mapa else "")

    def usa_objeto(self) -> bool:
        if self.tipo == "objeto":
            return True
        if self.tipo == "combina":
            return self.a.usa_objeto() or self.b.usa_objeto()
        return False


def cadeias(maxlen: int) -> List[Tuple[str, ...]]:
    fora: List[Tuple[str, ...]] = [()]
    for d in range(1, maxlen + 1):
        fora.extend(itertools.product(BASE, repeat=d))
    return fora


def enumerar() -> Iterator[Prog]:
    """
    Ordem de complexidade crescente. As cadeias puras vem primeiro para que a
    gramatica nova nunca chegue DEPOIS da antiga no orcamento — se ela perder,
    nao vai ser por ter empurrado o que ja funcionava para fora do teto.
    """
    c0 = cadeias(0)          # so ()
    c1 = cadeias(1)          # () + 20
    c2 = cadeias(2)          # () + 20 + 400

    for ops in c2:                                   # cadeia pura, ate 2
        yield Prog("cadeia", ops=ops)
    for ops in itertools.product(BASE, repeat=3):    # cadeia pura, 3
        yield Prog("cadeia", ops=ops)
    for ops in c1[1:]:                               # por_objeto(len 1)
        yield Prog("objeto", ops=ops)
    for op in OPS_COMB:                              # combina de cadeias curtas
        for a in c1:
            for b in c1:
                yield Prog("combina", op=op,
                           a=Prog("cadeia", ops=a), b=Prog("cadeia", ops=b))
    for ops in itertools.product(BASE, repeat=2):    # por_objeto(len 2)
        yield Prog("objeto", ops=ops)
    for f in c1[1:]:                                 # cadeia externa sobre combina
        for op in OPS_COMB:
            for a in c1:
                for b in c1:
                    yield Prog("combina", op=op, a=Prog("cadeia", ops=a),
                               b=Prog("cadeia", ops=b), fora=f)
    for f in c1[1:]:                                 # cadeia externa sobre objeto
        for ops in c1[1:]:
            yield Prog("objeto", ops=ops, fora=f)


def resolver(treino: Sequence[Tuple[Grid, Grid]],
             orcamento: int) -> Optional[Prog]:
    n = 0
    for p in enumerar():
        n += 1
        if n > orcamento:
            return None
        saidas, ok = [], True
        for a, b in treino:
            r = p.aplicar(a)
            if r is None:
                ok = False
                break
            saidas.append(r)
        if not ok:
            continue
        if all(r == b for r, (_, b) in zip(saidas, treino)):
            return p
        mapa = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
        if mapa is not None and all(aplicar_mapa(r, mapa) == b
                                    for r, (_, b) in zip(saidas, treino)):
            return Prog(p.tipo, ops=p.ops, op=p.op, a=p.a, b=p.b,
                        fora=p.fora, mapa=mapa)
    return None


def _saida(p: Prog, g: Grid) -> Optional[Grid]:
    r = p.aplicar(g)
    return aplicar_mapa(r, p.mapa) if (r is not None and p.mapa) else r


def relatorio(orcamento: int = 50000) -> None:
    print("=" * 78)
    print("ARC — combinadores: a forma do espaco, nao o tamanho da busca")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  K0  reproduzir as tarefas de painel que pXOR/pOU resolviam")
    print("  K1  >= 18/400 com o mesmo orcamento -> a forma era o gargalo")
    print("  K2  ~ 14/400 -> a forma nao era o gargalo")
    print("  K3  por_objeto resolve o que nenhuma cadeia resolvia")
    print()
    print(f"orcamento casado: {orcamento} (o mesmo da rodada prof 4)")
    print(f"primitivas de cadeia: {len(BASE)}  (pE/pOU/pXOR/pDIF removidas)")
    total = sum(1 for _ in enumerar())
    print(f"programas enumeraveis: {total}")

    av = carregar(AVALIACAO)
    t0 = time.time()
    ok, nomes, por_obj, comb = 0, [], 0, 0
    for nome, tr, te in av:
        p = resolver(tr, orcamento)
        if p is None:
            continue
        if all(_saida(p, a) == b for a, b in te):
            ok += 1
            nomes.append(f"{nome}  [{p}]")
            if p.usa_objeto():
                por_obj += 1
            if p.tipo == "combina":
                comb += 1
    dt = time.time() - t0

    print("\n" + "=" * 78)
    print(f"RESOLVIDAS  {ok}/400 = {ok/400:.1%}   ({dt:.0f}s)")
    print(f"  via combina     {comb}")
    print(f"  via por_objeto  {por_obj}")
    print(f"\nreferencia — cadeias lineares, mesmo orcamento: 14/400 = 3.5%")
    print()
    for x in nomes[:50]:
        print("   " + x)

    print("\n" + "=" * 78)
    if ok >= 18:
        print(f"K1 · A FORMA DO ESPACO ERA O GARGALO. {ok}/400 contra 14/400 com")
        print("     o MESMO orcamento e o MESMO verificador. Ramificacao e")
        print("     operacao por objeto nao sao mais primitivas escritas a mao.")
    elif abs(ok - 14) <= 3:
        print(f"K2 · A FORMA NAO ERA O GARGALO ({ok}/400 contra 14/400).")
        print("     Minha leitura de que 8-de-14 vinham de ramificacao escondida")
        print("     nao se traduziu em ganho ao generalizar. A hipotese cai.")
    else:
        print(f"Parcial: {ok}/400 contra 14/400. Entre os limiares declarados.")
    if por_obj > 0:
        print(f"\nK3 · `por_objeto` aparece em {por_obj} solucoes — familia que")
        print("     era ZERO nas cadeias lineares.")
    else:
        print("\nK3 · `por_objeto` nao resolveu nenhuma. A familia continua fora")
        print("     de alcance, e a forma implementada (transformar cada peca no")
        print("     LUGAR) nao e a que o ARC pede.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orcamento", type=int, default=50000)
    a = ap.parse_args()
    relatorio(orcamento=a.orcamento)

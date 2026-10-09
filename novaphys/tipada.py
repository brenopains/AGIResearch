"""
novaphys.tipada — o gargalo nunca foi profundidade, foi TIPO
==============================================================

A VIRADA DE CONCEPCAO
-----------------------
O dia inteiro eu tratei isto como problema de profundidade de busca. Todos os
numeros de hoje sao consistentes com outra leitura, e ela e melhor:

    ampliar vocabulario de 24 para 80 operadores  ->  +1 tarefa
    guiar a busca, tres mecanismos diferentes     ->  0
    33 das 51 solucoes tem UM passo

Se o problema fosse profundidade, ampliar vocabulario teria ajudado e as
solucoes seriam longas. Nao e o caso. O que esses numeros descrevem e uma
linguagem que resolve o que resolve em um passo e e MUDA para o resto.

E ela e muda por um motivo estrutural: as 24 primitivas sao TODAS
`Grade -> Grade`. Nao existe `contar`. Nao existe `a cor do maior objeto`.
Nao existe `o objeto diferente dos outros`.

    "pinte o maior objeto da cor que aparece uma vez so"

nao e uma composicao dificil de achar na nossa linguagem. E INEXPRIMIVEL, em
qualquer profundidade, porque nao ha valor intermediario do tipo Objeto nem do
tipo Cor por onde o programa possa passar. Cavar mais fundo num espaco que nao
contem a resposta e o que estivemos fazendo.

O QUE MUDA
------------
Multiplos tipos, e funcoes entre eles:

    Grade   Objeto   Lista[Objeto]   Cor   Int

    objetos    Grade -> Lista[Objeto]
    maior      Lista[Objeto] -> Objeto
    unico      Lista[Objeto] -> Objeto        o que difere dos demais
    contar     Lista[Objeto] -> Int
    corDe      Objeto -> Cor
    corRara    Grade -> Cor
    pintar     Grade x Objeto x Cor -> Grade
    recortar   Grade x Objeto -> Grade
    quadrado   Int x Cor -> Grade
    ...

Agora "pinte o maior objeto da cor rara" e:

    pintar(g, maior(objetos(g)), corRara(g))

Tres aplicacoes. Nao dez. A profundidade nunca foi o problema — a linguagem
e que nao tinha como dizer a frase.

ENUMERACAO ASCENDENTE, NAO DESCENDENTE
----------------------------------------
Em vez de enumerar PROGRAMAS e executa-los, constroi-se VALORES por tamanho
crescente: todos os valores de tamanho 1, depois combinacoes que dao tamanho
2, e assim por diante. Isso permite a peca decisiva:

EQUIVALENCIA OBSERVACIONAL
----------------------------
Dois programas que produzem EXATAMENTE os mesmos valores em todas as entradas
de treino sao intercambiaveis para sempre — qualquer coisa construida em cima
de um vale para o outro. Guarda-se um, descarta-se o outro.

Isso nao e heuristica nem aproximacao: e identidade sob observacao. E colapsa
o espaco de forma brutal, porque a esmagadora maioria das combinacoes produz
valores repetidos. E a tecnica que faz sintese de programas funcionar na
pratica (Bustle, DeepCoder, Transit) e nunca foi usada neste projeto.

O ORCAMENTO CONTINUA CASADO
-----------------------------
A moeda e a mesma dos outros experimentos: numero de valores construidos e
testados. O teto e 14.424, identico ao da enumeracao monolitica. Se a
linguagem tipada vencer, vencera gastando o mesmo.

ISTO E ESPECIFICO DO ARC?
---------------------------
Os tipos sao. As primitivas sao. O MECANISMO — enumeracao ascendente com
descarte por equivalencia observacional sobre uma linguagem multi-tipo — nao
e: vale para qualquer sintese com verificador. E a afirmacao sob teste e
sobre o mecanismo, nao sobre grades: "quando o gargalo e expressividade e nao
profundidade, ampliar TIPOS vence ampliar VOCABULARIO com o mesmo orcamento".
Essa frase e testavel e e o que este arquivo mede.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  T0  PRE-CONDICAO. A linguagem tipada tem que reproduzir pelo menos a maior
      parte das 14 tarefas que a monolitica resolve. Se perder muitas, a
      comparacao esta medindo troca de vocabulario e nao ganho de tipo.

  T1  tipada resolve >= 25 tarefas (vs 14 da monolitica) com o MESMO orcamento
      -> o gargalo era tipo, e a virada de concepcao esta certa

  T2  entre 18 e 24
      -> tipos ajudam de forma clara mas nao viram o jogo

  T3  <= 17
      -> tipos tambem nao eram o gargalo, e eu errei de novo o diagnostico

MINHA PREVISAO, escrita antes de rodar: T1 ou T2, e desta vez estou
genuinamente esperando o lado alto. Nao por otimismo: porque a familia
"consulte a estrutura e use o resultado" e uma fatia grande e visivel do ARC
que a linguagem atual nao consegue enunciar de jeito nenhum, e a equivalencia
observacional deve permitir alcancar tamanho 5 ou 6 dentro do mesmo orcamento
onde a monolitica alcanca 3.

O risco honesto: enumeracao ascendente com muitos tipos pode explodir na
combinacao de listas e ints antes de chegar em Grade util, e ai o orcamento
morre construindo valores que nunca viram saida.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import (Grid, ajustar_mapa, aplicar_mapa, carregar, dims,
                          fundo, resolver as resolver_mono, valida)
from novaphys.eter import PASTA_AVAL

K = 14424

# ── tipos
GRADE, OBJ, LISTA, COR, INT = "G", "O", "L", "C", "I"

Obj = Tuple[Tuple[int, int], ...]        # celulas de um componente


# ══════════════════════════════════════════════ operacoes por tipo
def _objs(g: Grid) -> Optional[Tuple[Obj, ...]]:
    h, w = dims(g)
    bg = fundo(g)
    visto = [[False] * w for _ in range(h)]
    fora: List[Obj] = []
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
            fora.append(tuple(sorted(cel)))
            if len(fora) > 60:
                return None
    return tuple(fora) or None


def _objs8(g: Grid) -> Optional[Tuple[Obj, ...]]:
    """Componentes 8-conexos — diagonal conta. Muda o que e 'um objeto'."""
    h, w = dims(g)
    bg = fundo(g)
    visto = [[False] * w for _ in range(h)]
    fora: List[Obj] = []
    viz = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    for i in range(h):
        for j in range(w):
            if visto[i][j] or g[i][j] == bg:
                continue
            pilha, cel = [(i, j)], []
            visto[i][j] = True
            while pilha:
                y, x = pilha.pop()
                cel.append((y, x))
                for dy, dx in viz:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not visto[ny][nx]
                            and g[ny][nx] != bg):
                        visto[ny][nx] = True
                        pilha.append((ny, nx))
            fora.append(tuple(sorted(cel)))
            if len(fora) > 60:
                return None
    return tuple(fora) or None


def _por_cor(g: Grid) -> Optional[Tuple[Obj, ...]]:
    """Cada cor nao-fundo vira um objeto, mesmo desconexo."""
    bg = fundo(g)
    d: Dict[int, List[Tuple[int, int]]] = {}
    for i, l in enumerate(g):
        for j, c in enumerate(l):
            if c != bg:
                d.setdefault(c, []).append((i, j))
    return tuple(tuple(sorted(v)) for _k, v in sorted(d.items())) or None


def _caixa(g: Grid, o: Obj) -> Optional[Grid]:
    if not o:
        return None
    ys = [y for y, _ in o]
    xs = [x for _, x in o]
    return tuple(tuple(g[y][x] for x in range(min(xs), max(xs) + 1))
                 for y in range(min(ys), max(ys) + 1))


def _mascara(g: Grid, o: Obj) -> Optional[Grid]:
    """Recorta a caixa mas apaga o que nao pertence ao objeto."""
    if not o:
        return None
    bg = fundo(g)
    ys = [y for y, _ in o]
    xs = [x for _, x in o]
    s = set(o)
    return tuple(tuple(g[y][x] if (y, x) in s else bg
                       for x in range(min(xs), max(xs) + 1))
                 for y in range(min(ys), max(ys) + 1))


def _unico(objs: Sequence[Obj], g: Grid) -> Optional[Obj]:
    """O objeto cuja FORMA e unica entre os demais — 'ache o diferente'."""
    if len(objs) < 2:
        return None
    def forma(o):
        ys = [y for y, _ in o]
        xs = [x for _, x in o]
        return tuple(sorted((y - min(ys), x - min(xs)) for y, x in o))
    cont: Dict[tuple, int] = {}
    for o in objs:
        cont[forma(o)] = cont.get(forma(o), 0) + 1
    raros = [o for o in objs if cont[forma(o)] == 1]
    return raros[0] if len(raros) == 1 else None


def _cor_unica(g: Grid) -> Optional[int]:
    """A cor nao-fundo que aparece em menos celulas."""
    bg = fundo(g)
    c: Dict[int, int] = {}
    for l in g:
        for x in l:
            if x != bg:
                c[x] = c.get(x, 0) + 1
    if len(c) < 2:
        return None
    return min(c, key=lambda k: c[k])


def _cor_dom(g: Grid) -> Optional[int]:
    bg = fundo(g)
    c: Dict[int, int] = {}
    for l in g:
        for x in l:
            if x != bg:
                c[x] = c.get(x, 0) + 1
    return max(c, key=lambda k: c[k]) if c else None


def _quadrado(n: int, c: int) -> Optional[Grid]:
    if not 1 <= n <= 30:
        return None
    return tuple(tuple(c for _ in range(n)) for _ in range(n))


def _linha(n: int, c: int) -> Optional[Grid]:
    if not 1 <= n <= 30:
        return None
    return (tuple(c for _ in range(n)),)


def _pintar(g: Grid, o: Obj, c: int) -> Optional[Grid]:
    s = set(o)
    return tuple(tuple(c if (i, j) in s else v for j, v in enumerate(l))
                 for i, l in enumerate(g))


def _so_obj(g: Grid, o: Obj) -> Optional[Grid]:
    """Apaga tudo menos o objeto, mantendo o tamanho da grade."""
    bg = fundo(g)
    s = set(o)
    return tuple(tuple(v if (i, j) in s else bg for j, v in enumerate(l))
                 for i, l in enumerate(g))


def _sem_obj(g: Grid, o: Obj) -> Optional[Grid]:
    bg = fundo(g)
    s = set(o)
    return tuple(tuple(bg if (i, j) in s else v for j, v in enumerate(l))
                 for i, l in enumerate(g))


def _recolore(g: Grid, a: int, b: int) -> Optional[Grid]:
    return tuple(tuple(b if v == a else v for v in l) for l in g)


# ── assinaturas: (nome, tipos dos argumentos, tipo do resultado, funcao)
UNARIAS: List[Tuple[str, str, str, Callable]] = [
    ("objs4", GRADE, LISTA, _objs),
    ("objs8", GRADE, LISTA, _objs8),
    ("objsCor", GRADE, LISTA, _por_cor),
    ("maior", LISTA, OBJ, lambda L: max(L, key=len) if L else None),
    ("menor", LISTA, OBJ, lambda L: min(L, key=len) if L else None),
    ("contar", LISTA, INT, lambda L: len(L) if L else None),
    ("corRara", GRADE, COR, _cor_unica),
    ("corDom", GRADE, COR, _cor_dom),
    ("fundo", GRADE, COR, fundo),
]

BINARIAS: List[Tuple[str, str, str, str, Callable]] = [
    ("recorta", GRADE, OBJ, GRADE, _caixa),
    ("mascara", GRADE, OBJ, GRADE, _mascara),
    ("soObj", GRADE, OBJ, GRADE, _so_obj),
    ("semObj", GRADE, OBJ, GRADE, _sem_obj),
    ("unico", LISTA, GRADE, OBJ, _unico),
    ("corDe", OBJ, GRADE, COR,
     lambda o, g: g[o[0][0]][o[0][1]] if o else None),
    ("quadrado", INT, COR, GRADE, _quadrado),
    ("linha", INT, COR, GRADE, _linha),
]

TERNARIAS: List[Tuple[str, str, str, str, str, Callable]] = [
    ("pintar", GRADE, OBJ, COR, GRADE, _pintar),
    ("recolore", GRADE, COR, COR, GRADE, _recolore),
]


# ══════════════════════════════════════════════ enumeracao ascendente
def sintetizar(treino: Sequence[Tuple[Grid, Grid]], orcamento: int,
               tam_max: int = 6) -> Tuple[Optional[str], int]:
    """
    Constroi VALORES por tamanho crescente. Cada valor e uma tupla com o
    resultado em CADA entrada de treino — e essa tupla inteira e a chave de
    equivalencia observacional.

    Se duas expressoes diferentes produzem a mesma tupla, sao intercambiaveis
    em qualquer contexto futuro. Guarda-se a primeira (menor) e descarta-se a
    outra. E identidade exata, nao aproximacao, e e o que torna a enumeracao
    viavel.
    """
    ents = [a for a, _ in treino]
    alvos = [b for _, b in treino]
    n = len(ents)

    # banco[tipo] = {chave_observacional: (expressao, valores)}
    banco: Dict[str, Dict[tuple, Tuple[str, tuple]]] = {
        t: {} for t in (GRADE, OBJ, LISTA, COR, INT)}
    gasto = 0

    def testar(vals: tuple) -> bool:
        return all(v == b for v, b in zip(vals, alvos))

    def guardar(tipo: str, expr: str, vals: tuple) -> Optional[str]:
        """
        CONTABILIDADE. A primeira versao so cobrava do orcamento os valores do
        tipo Grade — construir Objeto, Lista, Cor e Int saia de graca. Isso e
        injusto contra o braco monolitico, que paga por cada candidato: a
        tipada estaria fazendo muito mais trabalho pelo mesmo numero declarado.

        Agora TODO valor construido custa 1, do tipo que for. Se a tipada
        vencer, vence tendo construido o mesmo numero de coisas.
        """
        nonlocal gasto
        gasto += 1
        if any(v is None for v in vals):
            return None
        chave = vals
        if chave in banco[tipo]:
            return None
        banco[tipo][chave] = (expr, vals)
        if tipo == GRADE:
            if not all(valida(v) for v in vals):
                return None
            if testar(vals):
                return expr
            m = ajustar_mapa(list(zip(vals, alvos)))
            if m is not None and all(aplicar_mapa(v, m) == b
                                     for v, b in zip(vals, alvos)):
                return f"{expr} +recolore{m}"
        return None

    r = guardar(GRADE, "x", tuple(ents))
    if r:
        return r, gasto

    for _tam in range(2, tam_max + 1):
        if gasto >= orcamento:
            break
        novos: List[Tuple[str, str, tuple]] = []
        estourou = False

        def emitir(tipo, expr, vals):
            """Um so ponto de saida, para o corte por orcamento valer em
            todos os niveis de laco — a versao anterior tinha `break` no laco
            mais interno e os externos continuavam girando."""
            novos.append((tipo, expr, vals))

        for nome, ta, tr, fn in UNARIAS:
            if len(novos) + gasto >= orcamento:
                estourou = True
                break
            for _ch, (e, vs) in list(banco[ta].items()):
                if len(novos) + gasto >= orcamento:
                    estourou = True
                    break
                try:
                    saida = tuple(fn(v) for v in vs)
                except Exception:
                    continue
                emitir(tr, f"{nome}({e})", saida)

        for nome, ta, tb, tr, fn in BINARIAS:
            if estourou:
                break
            for _c1, (e1, v1) in list(banco[ta].items()):
                if estourou:
                    break
                for _c2, (e2, v2) in list(banco[tb].items()):
                    if len(novos) + gasto >= orcamento:
                        estourou = True
                        break
                    try:
                        saida = tuple(fn(a, b) for a, b in zip(v1, v2))
                    except Exception:
                        continue
                    emitir(tr, f"{nome}({e1},{e2})", saida)

        for nome, ta, tb, tc, tr, fn in TERNARIAS:
            if estourou:
                break
            for _c1, (e1, v1) in list(banco[ta].items()):
                if estourou:
                    break
                for _c2, (e2, v2) in list(banco[tb].items()):
                    if estourou:
                        break
                    for _c3, (e3, v3) in list(banco[tc].items()):
                        if len(novos) + gasto >= orcamento:
                            estourou = True
                            break
                        try:
                            saida = tuple(fn(a, b, c)
                                          for a, b, c in zip(v1, v2, v3))
                        except Exception:
                            continue
                        emitir(tr, f"{nome}({e1},{e2},{e3})", saida)

        achou = None
        for tipo, expr, vals in novos:
            r = guardar(tipo, expr, vals)
            if r and achou is None:
                achou = r
        if achou:
            return achou, gasto
        if not novos or estourou:
            break
    return None, gasto


def aplicar_expr(expr: str, g: Grid) -> Optional[Grid]:
    """Reexecuta a expressao numa grade nova. Parser minimo do que geramos."""
    mapa = None
    if " +recolore" in expr:
        expr, resto = expr.split(" +recolore", 1)
        mapa = eval(resto)                       # noqa: S307 — nosso proprio
    U = {n: f for n, _a, _r, f in UNARIAS}
    B = {n: f for n, _a, _b, _r, f in BINARIAS}
    T = {n: f for n, _a, _b, _c, _r, f in TERNARIAS}

    def val(e: str):
        e = e.strip()
        if e == "x":
            return g
        i = e.index("(")
        nome, dentro = e[:i], e[i + 1:-1]
        args, prof, atual = [], 0, ""
        for ch in dentro:
            if ch == "," and prof == 0:
                args.append(atual)
                atual = ""
                continue
            if ch == "(":
                prof += 1
            elif ch == ")":
                prof -= 1
            atual += ch
        args.append(atual)
        vs = [val(x) for x in args]
        if any(v is None for v in vs):
            return None
        try:
            if nome in U:
                return U[nome](vs[0])
            if nome in B:
                return B[nome](vs[0], vs[1])
            return T[nome](vs[0], vs[1], vs[2])
        except Exception:
            return None

    r = val(expr)
    if r is None or not valida(r):
        return None
    return aplicar_mapa(r, mapa) if mapa else r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--orcamento", type=int, default=K)
    ap.add_argument("--tam", type=int, default=6)
    a = ap.parse_args()

    print("=" * 76)
    print("TIPADA — o gargalo nunca foi profundidade, foi TIPO")
    print("=" * 76)
    print(f"orcamento IDENTICO ao dos outros experimentos: {a.orcamento}")
    print(f"tipos: Grade, Objeto, Lista, Cor, Int")
    print(f"{len(UNARIAS)} unarias, {len(BINARIAS)} binarias, "
          f"{len(TERNARIAS)} ternarias\n")

    tarefas = carregar(a.pasta)

    mono: Set[str] = set()
    for nome, tr, te in tarefas:
        p = resolver_mono(tr, prof=3, orcamento=a.orcamento)
        if p is not None and all(p.aplicar(x) == y for x, y in te):
            mono.add(nome)
    print(f"monolitica Grade->Grade : {len(mono):3d}/400  "
          f"({len(mono)/len(tarefas):.2%})")

    tip: Set[str] = set()
    exprs: Dict[str, str] = {}
    t0 = time.time()
    for nome, tr, te in tarefas:
        e, _g = sintetizar(tr, a.orcamento, a.tam)
        if e is None:
            continue
        if all(aplicar_expr(e, x) == y for x, y in te):
            tip.add(nome)
            exprs[nome] = e
    print(f"tipada multi-tipo       : {len(tip):3d}/400  "
          f"({len(tip)/len(tarefas):.2%})   ({time.time()-t0:.0f}s)")

    uniao = mono | tip
    print(f"uniao (2x orcamento)    : {len(uniao):3d}/400  "
          f"({len(uniao)/len(tarefas):.2%})")

    print(f"\nT0 · das {len(mono)} da monolitica, a tipada mantem "
          f"{len(mono & tip)}")
    print(f"so a monolitica: {sorted(mono - tip)}")
    print(f"\nso a TIPADA resolve ({len(tip - mono)}):")
    for n in sorted(tip - mono)[:25]:
        print(f"   {n}  ->  {exprs[n][:100]}")

    print("\n" + "=" * 76)
    t = len(tip)
    if t >= 25:
        print("T1 · o gargalo era TIPO. A virada de concepcao estava certa.")
    elif t >= 18:
        print("T2 · tipos ajudam de forma clara, mas nao viram o jogo.")
    else:
        print("T3 · tipos tambem nao eram o gargalo. Errei o diagnostico "
              "de novo.")


if __name__ == "__main__":
    main()

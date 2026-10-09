"""
novaphys.priors — quanto vale cada um dos quatro priors de Conhecimento Nuclear
================================================================================

O FATO QUE ORIGINA ESTE ARQUIVO
---------------------------------
Chollet construiu o ARC deliberadamente sobre quatro priors da psicologia do
desenvolvimento, adquiridos por criancas antes dos quatro anos:

    GEOMETRIA    posicao, simetria, translacao, rotacao, escala
    OBJETIDADE   grades como objetos coesos, persistentes, que interagem
    NUMERO       contar, somar, subtrair, ordenar, maior, menor
    AGENCIA      objetos como agentes agindo para atingir um objetivo

E a afirmacao do material da ARC Prize: um solucionador que entenda apenas
esses quatro conceitos consegue resolver o ARC. Conhecimento de linguagem ou
fatos do mundo e desnecessario.

A AUDITORIA QUE MOTIVOU TUDO
------------------------------
Classificando nossas 24 primitivas contra essa lista:

    GEOMETRIA     16/24
    logica painel  6/24
    OBJETIDADE     1/24   (so `maiorObj`)
    NUMERO         0/24
    AGENCIA        0/24

Implementamos UM dos quatro priors com esmero e tres com nada. E fizemos 3,5%.
Isso deixou de ser misterio — e explica por que ampliar o vocabulario de 24
para 80 rendeu UMA tarefa: eu adicionei mais geometria e cor, ou seja, mais do
balde que ja estava cheio.

O QUE ESTE ARQUIVO MEDE
-------------------------
Nao "o total sobe?". Isso e um numero sem eixo. Mede-se a CONTRIBUICAO
MARGINAL de cada prior, num desenho fatorial:

    cada prior sozinho          quanto ele alcanca isolado
    todos menos um              quanto se perde ao remover so aquele
    todos                       o teto do conjunto

A segunda linha e a que informa: um prior pode parecer fraco sozinho e ser
insubstituivel em combinacao, ou o contrario. Sem o "todos menos um" nao da
para distinguir os dois casos.

Este numero nao existe em lugar nenhum do nosso projeto, e ele diz onde esta o
resto do ARC.

MOTOR
-------
Enumeracao ascendente com descarte por equivalencia observacional, o mesmo de
`tipada.py`: constroi-se VALORES por tamanho crescente, e dois programas que
produzem os mesmos valores em todas as entradas de treino sao intercambiaveis
— guarda-se um. Nao e heuristica, e identidade sob observacao.

ORCAMENTO
-----------
14.424 construcoes por tarefa, o mesmo teto de todos os experimentos da
sessao, e TODA construcao custa 1 — de qualquer tipo. A primeira versao de
`tipada.py` so cobrava valores do tipo Grade, o que era injusto contra o braco
monolitico; esta corrigido.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO. So GEOMETRIA tem que ficar perto das 14 tarefas da busca
      monolitica — e o prior que ja tinhamos. Se ficar muito abaixo, o motor
      de sintese esta pior que a enumeracao e a comparacao entre priors nao
      vale.

  P1  algum prior ausente vale >= 8 tarefas de contribuicao marginal
      -> ha uma fatia grande do ARC atras de um prior especifico, e ela tem
         nome

  P2  contribuicoes marginais espalhadas, nenhuma >= 8
      -> o ARC exige os quatro juntos e nenhum sozinho compra muito

  P3  o total com os quatro fica <= 17
      -> os priors nao eram o gargalo e eu errei o diagnostico pela quarta vez

MINHA PREVISAO, escrita antes de rodar: OBJETIDADE com a maior contribuicao
marginal, entre 6 e 15 tarefas; NUMERO pequeno, 2 a 5; AGENCIA pequeno no
ARC-AGI-1, 1 a 4 — porque agencia e o prior que o ARC-AGI-3 exige e este
conjunto e o 1. Total esperado entre 20 e 35.

Se AGENCIA vier alto aqui eu estava errado sobre a divisao entre os
benchmarks. Se OBJETIDADE vier baixo, o diagnostico dos quatro priors nao
sobrevive e a conclusao honesta e que a classificacao explicava menos do que
parecia.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import (Grid, ajustar_mapa, aplicar_mapa, carregar, dims,
                          fundo, valida)

Obj = Tuple[Tuple[int, int], ...]
GRADE, OBJ, LISTA, COR, INT = "G", "O", "L", "C", "I"
K = 14424
RAIZ = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))


# ══════════════════════════════════════════════ utilidades de objeto
def _comps(g: Grid, bg: int, diag: bool) -> Optional[Tuple[Obj, ...]]:
    h, w = dims(g)
    viz = ([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
           if diag else [(1, 0), (-1, 0), (0, 1), (0, -1)])
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
                for dy, dx in viz:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not visto[ny][nx]
                            and g[ny][nx] != bg):
                        visto[ny][nx] = True
                        pilha.append((ny, nx))
            fora.append(tuple(sorted(cel)))
            if len(fora) > 50:
                return None
    return tuple(fora) or None


def _caixa(g: Grid, o: Obj) -> Optional[Grid]:
    if not o:
        return None
    ys = [y for y, _ in o]
    xs = [x for _, x in o]
    return tuple(tuple(g[y][x] for x in range(min(xs), max(xs) + 1))
                 for y in range(min(ys), max(ys) + 1))


def _forma(o: Obj) -> tuple:
    ys = [y for y, _ in o]
    xs = [x for _, x in o]
    return tuple(sorted((y - min(ys), x - min(xs)) for y, x in o))


# ══════════════════════════════════════════════ GEOMETRIA
def _rot90(g):
    return tuple(zip(*g[::-1]))


def _rot180(g):
    return tuple(tuple(l[::-1]) for l in g[::-1])


def _rot270(g):
    return tuple(zip(*g))[::-1]


def _espH(g):
    return tuple(tuple(l[::-1]) for l in g)


def _espV(g):
    return tuple(g[::-1])


def _transp(g):
    return tuple(zip(*g))


def _ladriH(g):
    return tuple(l + l for l in g)


def _ladriV(g):
    return g + g


def _amplia(k):
    def fn(g):
        h, w = dims(g)
        if h * k > 30 or w * k > 30:
            return None
        fora = []
        for l in g:
            linha = tuple(c for c in l for _ in range(k))
            for _ in range(k):
                fora.append(linha)
        return tuple(fora)
    return fn


def _reduz_bloco(g):
    h, w = dims(g)
    for k in (2, 3):
        if h % k == 0 and w % k == 0:
            fora = []
            bom = True
            for i in range(0, h, k):
                linha = []
                for j in range(0, w, k):
                    b = {g[i + a][j + c] for a in range(k) for c in range(k)}
                    if len(b) != 1:
                        bom = False
                        break
                    linha.append(b.pop())
                if not bom:
                    break
                fora.append(tuple(linha))
            if bom:
                return tuple(fora)
    return None


def _recorta(g):
    bg = fundo(g)
    cel = [(i, j) for i, l in enumerate(g) for j, c in enumerate(l) if c != bg]
    return _caixa(g, tuple(cel)) if cel else None


def _simH(g):
    """Espelha e cola: simetriza horizontalmente."""
    return tuple(l + tuple(reversed(l)) for l in g)


def _simV(g):
    return g + g[::-1]


def _fractal(g):
    """Cada celula nao-fundo vira uma copia da grade inteira."""
    h, w = dims(g)
    if h * h > 30 or w * w > 30:
        return None
    bg = fundo(g)
    fora = []
    for i in range(h):
        for _ in range(h):
            fora.append([bg] * (w * w))
    for i in range(h):
        for j in range(w):
            if g[i][j] == bg:
                continue
            for a in range(h):
                for b in range(w):
                    fora[i * h + a][j * w + b] = g[a][b]
    return tuple(tuple(l) for l in fora)


GEOMETRIA_U = [
    ("simH", GRADE, GRADE, _simH), ("simV", GRADE, GRADE, _simV),
    ("fractal", GRADE, GRADE, _fractal),
    ("rot90", GRADE, GRADE, _rot90), ("rot180", GRADE, GRADE, _rot180),
    ("rot270", GRADE, GRADE, _rot270), ("espH", GRADE, GRADE, _espH),
    ("espV", GRADE, GRADE, _espV), ("transp", GRADE, GRADE, _transp),
    ("ladriH", GRADE, GRADE, _ladriH), ("ladriV", GRADE, GRADE, _ladriV),
    ("amplia2", GRADE, GRADE, _amplia(2)), ("amplia3", GRADE, GRADE, _amplia(3)),
    ("reduzBloco", GRADE, GRADE, _reduz_bloco),
    ("recorta", GRADE, GRADE, _recorta),
]
GEOMETRIA_B: List = []
GEOMETRIA_T: List = []


# ══════════════════════════════════════════════ OBJETIDADE
def _unico_forma(L):
    if len(L) < 2:
        return None
    c: Dict[tuple, int] = {}
    for o in L:
        c[_forma(o)] = c.get(_forma(o), 0) + 1
    raros = [o for o in L if c[_forma(o)] == 1]
    return raros[0] if len(raros) == 1 else None


def _so_obj(g, o):
    bg = fundo(g)
    s = set(o)
    return tuple(tuple(v if (i, j) in s else bg for j, v in enumerate(l))
                 for i, l in enumerate(g))


def _sem_obj(g, o):
    bg = fundo(g)
    s = set(o)
    return tuple(tuple(bg if (i, j) in s else v for j, v in enumerate(l))
                 for i, l in enumerate(g))


def _mascara(g, o):
    if not o:
        return None
    bg = fundo(g)
    ys = [y for y, _ in o]
    xs = [x for _, x in o]
    s = set(o)
    return tuple(tuple(g[y][x] if (y, x) in s else bg
                       for x in range(min(xs), max(xs) + 1))
                 for y in range(min(ys), max(ys) + 1))


def _pintar(g, o, c):
    s = set(o)
    return tuple(tuple(c if (i, j) in s else v for j, v in enumerate(l))
                 for i, l in enumerate(g))


OBJETIDADE_U = [
    ("objs4", GRADE, LISTA, lambda g: _comps(g, fundo(g), False)),
    ("objs8", GRADE, LISTA, lambda g: _comps(g, fundo(g), True)),
    ("maior", LISTA, OBJ, lambda L: max(L, key=len) if L else None),
    ("menor", LISTA, OBJ, lambda L: min(L, key=len) if L else None),
    ("formaUnica", LISTA, OBJ, _unico_forma),
]
OBJETIDADE_B = [
    ("recortaObj", GRADE, OBJ, GRADE, _caixa),
    ("mascara", GRADE, OBJ, GRADE, _mascara),
    ("soObj", GRADE, OBJ, GRADE, _so_obj),
    ("semObj", GRADE, OBJ, GRADE, _sem_obj),
    ("corDe", OBJ, GRADE, COR,
     lambda o, g: g[o[0][0]][o[0][1]] if o else None),
]
OBJETIDADE_T = [("pintar", GRADE, OBJ, COR, GRADE, _pintar)]


# ══════════════════════════════════════════════ NUMERO
def _cor_rara(g):
    bg = fundo(g)
    c: Dict[int, int] = {}
    for l in g:
        for x in l:
            if x != bg:
                c[x] = c.get(x, 0) + 1
    return min(c, key=lambda k: c[k]) if len(c) >= 2 else None


def _cor_dom(g):
    bg = fundo(g)
    c: Dict[int, int] = {}
    for l in g:
        for x in l:
            if x != bg:
                c[x] = c.get(x, 0) + 1
    return max(c, key=lambda k: c[k]) if c else None


def _quadrado(n, c):
    return (tuple(tuple(c for _ in range(n)) for _ in range(n))
            if 1 <= n <= 30 else None)


def _linha(n, c):
    return (tuple(c for _ in range(n)),) if 1 <= n <= 30 else None


def _repetirH(g, n):
    h, w = dims(g)
    if not 1 <= n <= 30 or w * n > 30:
        return None
    return tuple(l * n for l in g)


def _repetirV(g, n):
    h, w = dims(g)
    if not 1 <= n <= 30 or h * n > 30:
        return None
    return g * n


NUMERO_U = [
    ("contar", LISTA, INT, lambda L: len(L) if L else None),
    ("tamObj", OBJ, INT, lambda o: len(o) if o else None),
    ("corRara", GRADE, COR, _cor_rara),
    ("corDom", GRADE, COR, _cor_dom),
    ("nCores", GRADE, INT,
     lambda g: len({c for l in g for c in l}) or None),
]
NUMERO_B = [
    ("quadrado", INT, COR, GRADE, _quadrado),
    ("linha", INT, COR, GRADE, _linha),
    ("repetirH", GRADE, INT, GRADE, _repetirH),
    ("repetirV", GRADE, INT, GRADE, _repetirV),
]
NUMERO_T: List = []


# ══════════════════════════════════════════════ AGENCIA
# O prior que temos ZERO. Objetos como agentes: caem, deslizam, esticam
# ate encontrar obstaculo, sao atraidos. E o prior que o ARC-AGI-3 exige.
def _gravidade(dy: int, dx: int):
    """Todos os objetos deslizam nesta direcao ate colidir com algo ou a borda."""
    def fn(g: Grid) -> Optional[Grid]:
        bg = fundo(g)
        L = _comps(g, bg, True)
        if not L:
            return None
        h, w = dims(g)
        ocup = [[bg] * w for _ in range(h)]
        # move primeiro os que estao mais adiante na direcao do movimento
        ordem = sorted(L, key=lambda o: -max(y * dy + x * dx for y, x in o))
        for o in ordem:
            cor = {(y, x): g[y][x] for y, x in o}
            d = 0
            while True:
                nd = d + 1
                ok = True
                for y, x in o:
                    ny, nx = y + dy * nd, x + dx * nd
                    if not (0 <= ny < h and 0 <= nx < w) or ocup[ny][nx] != bg:
                        ok = False
                        break
                if not ok:
                    break
                d = nd
                if d > 40:
                    return None
            for y, x in o:
                ocup[y + dy * d][x + dx * d] = cor[(y, x)]
        return tuple(tuple(l) for l in ocup)
    return fn


def _estender(dy: int, dx: int):
    """Cada celula nao-fundo emite um raio ate bater em algo ou na borda."""
    def fn(g: Grid) -> Optional[Grid]:
        bg = fundo(g)
        h, w = dims(g)
        fora = [list(l) for l in g]
        for i in range(h):
            for j in range(w):
                if g[i][j] == bg:
                    continue
                y, x = i + dy, j + dx
                while 0 <= y < h and 0 <= x < w and g[y][x] == bg:
                    fora[y][x] = g[i][j]
                    y += dy
                    x += dx
        return tuple(tuple(l) for l in fora)
    return fn


def _mover_obj(dy: int, dx: int):
    """UM objeto escolhido desliza ate colidir. Agencia sobre alvo."""
    def fn(g: Grid, o: Obj) -> Optional[Grid]:
        bg = fundo(g)
        h, w = dims(g)
        s = set(o)
        resto = [[bg if (i, j) in s else v for j, v in enumerate(l)]
                 for i, l in enumerate(g)]
        d = 0
        while True:
            nd = d + 1
            ok = True
            for y, x in o:
                ny, nx = y + dy * nd, x + dx * nd
                if not (0 <= ny < h and 0 <= nx < w) or resto[ny][nx] != bg:
                    ok = False
                    break
            if not ok:
                break
            d = nd
            if d > 40:
                return None
        for y, x in o:
            resto[y + dy * d][x + dx * d] = g[y][x]
        return tuple(tuple(l) for l in resto)
    return fn


AGENCIA_U = [
    (f"cai{dy}_{dx}", GRADE, GRADE, _gravidade(dy, dx))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
] + [
    (f"raio{dy}_{dx}", GRADE, GRADE, _estender(dy, dx))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
]
AGENCIA_B = [
    (f"move{dy}_{dx}", GRADE, OBJ, GRADE, _mover_obj(dy, dx))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
]
AGENCIA_T: List = []


# ══════════════════════════════════════════════ PAINEL
# As 6 primitivas de logica entre paineis. Na auditoria eu as classifiquei
# como "logica de painel" e depois nao as coloquei em prior nenhum — sumiram
# da implementacao. Elas nao mapeiam limpo nos quatro de Chollet (sao
# composicao de topologia com logica), entao viram grupo proprio e rotulado
# em vez de desaparecer da medicao.
def _dividir(g: Grid):
    h, w = dims(g)
    for j in range(1, w - 1):
        if len({g[i][j] for i in range(h)}) == 1 and j == w - 1 - j:
            a = tuple(l[:j] for l in g)
            b = tuple(l[j + 1:] for l in g)
            if dims(a) == dims(b) and a and a[0]:
                return a, b
    for i in range(1, h - 1):
        if len(set(g[i])) == 1 and i == h - 1 - i:
            a, b = tuple(g[:i]), tuple(g[i + 1:])
            if dims(a) == dims(b) and a and a[0]:
                return a, b
    if w % 2 == 0:
        a = tuple(l[:w // 2] for l in g)
        b = tuple(l[w // 2:] for l in g)
        if a and a[0]:
            return a, b
    if h % 2 == 0:
        return tuple(g[:h // 2]), tuple(g[h // 2:])
    return None


def _bin(op):
    def fn(g: Grid) -> Optional[Grid]:
        d = _dividir(g)
        if d is None:
            return None
        a, b = d
        bg = fundo(g)
        cor = next((c for l in g for c in l if c != bg), 1)
        return tuple(tuple(cor if op(x != bg, y != bg) else bg
                           for x, y in zip(la, lb))
                     for la, lb in zip(a, b))
    return fn


def _painel(k):
    def fn(g: Grid) -> Optional[Grid]:
        d = _dividir(g)
        return d[k] if d else None
    return fn


PAINEL_U = [
    ("pE", GRADE, GRADE, _bin(lambda x, y: x and y)),
    ("pOU", GRADE, GRADE, _bin(lambda x, y: x or y)),
    ("pXOR", GRADE, GRADE, _bin(lambda x, y: x != y)),
    ("pDIF", GRADE, GRADE, _bin(lambda x, y: x and not y)),
    ("painelA", GRADE, GRADE, _painel(0)),
    ("painelB", GRADE, GRADE, _painel(1)),
]
PAINEL_B: List = []
PAINEL_T: List = []


PRIORS = {
    "GEOMETRIA": (GEOMETRIA_U, GEOMETRIA_B, GEOMETRIA_T),
    "PAINEL": (PAINEL_U, PAINEL_B, PAINEL_T),
    "OBJETIDADE": (OBJETIDADE_U, OBJETIDADE_B, OBJETIDADE_T),
    "NUMERO": (NUMERO_U, NUMERO_B, NUMERO_T),
    "AGENCIA": (AGENCIA_U, AGENCIA_B, AGENCIA_T),
}


def montar(ativos: Sequence[str]):
    u, b, t = [], [], []
    for p in ativos:
        pu, pb, pt = PRIORS[p]
        u.extend(pu)
        b.extend(pb)
        t.extend(pt)
    return u, b, t


# ══════════════════════════════════════════════ motor de sintese
def sintetizar(treino, ativos: Sequence[str], orcamento: int = K,
               tam_max: int = 5) -> Tuple[Optional[str], int]:
    """
    Enumeracao ascendente por tamanho, com descarte por equivalencia
    observacional. TODA construcao custa 1, de qualquer tipo.
    """
    U, B, T = montar(ativos)
    ents = [a for a, _ in treino]
    alvos = [b for _, b in treino]
    banco: Dict[str, Dict[tuple, Tuple[str, tuple]]] = {
        x: {} for x in (GRADE, OBJ, LISTA, COR, INT)}
    gasto = 0

    def guardar(tipo, expr, vals):
        nonlocal gasto
        gasto += 1
        if any(v is None for v in vals):
            return None
        if vals in banco[tipo]:
            return None
        banco[tipo][vals] = (expr, vals)
        if tipo == GRADE:
            if not all(valida(v) for v in vals):
                return None
            if all(v == b for v, b in zip(vals, alvos)):
                return expr
            m = ajustar_mapa(list(zip(vals, alvos)))
            if m is not None and all(aplicar_mapa(v, m) == b
                                     for v, b in zip(vals, alvos)):
                return f"{expr}|MAPA{m}"
        return None

    r = guardar(GRADE, "x", tuple(ents))
    if r:
        return r, gasto

    for _tam in range(2, tam_max + 1):
        if gasto >= orcamento:
            break
        novos = []
        cheio = False
        for nome, ta, tr, fn in U:
            if cheio:
                break
            for _c, (e, vs) in list(banco[ta].items()):
                if gasto + len(novos) >= orcamento:
                    cheio = True
                    break
                try:
                    novos.append((tr, f"{nome}({e})", tuple(fn(v) for v in vs)))
                except Exception:
                    pass
        for nome, ta, tb, tr, fn in B:
            if cheio:
                break
            for _c1, (e1, v1) in list(banco[ta].items()):
                if cheio:
                    break
                for _c2, (e2, v2) in list(banco[tb].items()):
                    if gasto + len(novos) >= orcamento:
                        cheio = True
                        break
                    try:
                        novos.append((tr, f"{nome}({e1},{e2})",
                                      tuple(fn(p, q) for p, q in zip(v1, v2))))
                    except Exception:
                        pass
        for nome, ta, tb, tc, tr, fn in T:
            if cheio:
                break
            for _c1, (e1, v1) in list(banco[ta].items()):
                if cheio:
                    break
                for _c2, (e2, v2) in list(banco[tb].items()):
                    if cheio:
                        break
                    for _c3, (e3, v3) in list(banco[tc].items()):
                        if gasto + len(novos) >= orcamento:
                            cheio = True
                            break
                        try:
                            novos.append(
                                (tr, f"{nome}({e1},{e2},{e3})",
                                 tuple(fn(p, q, s)
                                       for p, q, s in zip(v1, v2, v3))))
                        except Exception:
                            pass
        achou = None
        for tipo, expr, vals in novos:
            got = guardar(tipo, expr, vals)
            if got and achou is None:
                achou = got
        if achou:
            return achou, gasto
        if not novos or cheio:
            break
    return None, gasto


def aplicar_expr(expr: str, g: Grid, ativos: Sequence[str]) -> Optional[Grid]:
    U, B, T = montar(ativos)
    du = {n: f for n, _a, _r, f in U}
    db = {n: f for n, _a, _b, _r, f in B}
    dt = {n: f for n, _a, _b, _c, _r, f in T}
    mapa = None
    if "|MAPA" in expr:
        expr, resto = expr.split("|MAPA", 1)
        mapa = eval(resto)                     # noqa: S307 — nosso proprio

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
            if nome in du:
                return du[nome](vs[0])
            if nome in db:
                return db[nome](vs[0], vs[1])
            return dt[nome](vs[0], vs[1], vs[2])
        except Exception:
            return None

    r = val(expr)
    if r is None or not valida(r):
        return None
    return aplicar_mapa(r, mapa) if mapa else r


def resolver_tarefa(args):
    nome, tr, te, ativos = args
    e, g = sintetizar(tr, ativos)
    if e is None:
        return nome, False, None
    try:
        return nome, all(aplicar_expr(e, x, ativos) == y for x, y in te), e
    except Exception:
        return nome, False, None


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--conj", default="evaluation")
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    tarefas = carregar(RAIZ + a.conj)
    nomes = ["GEOMETRIA", "PAINEL", "OBJETIDADE", "NUMERO", "AGENCIA"]

    print("=" * 76)
    print("PRIORS — quanto vale cada Conhecimento Nuclear, isolado e em falta")
    print("=" * 76)
    print(f"conjunto {a.conj} · {len(tarefas)} tarefas · orcamento {K}\n")
    for p in nomes:
        u, b, t = PRIORS[p]
        print(f"   {p:11s} {len(u)+len(b)+len(t):2d} operadores")
    print()

    def rodar(ativos, rot):
        t0 = time.time()
        with Pool(a.nucleos) as pool:
            res = pool.map(resolver_tarefa,
                           [(n, tr, te, ativos) for n, tr, te in tarefas],
                           chunksize=4)
        ok = {n for n, o, _e in res if o}
        ex = {n: e for n, o, e in res if o}
        print(f"{rot:26s} {len(ok):3d}/{len(tarefas)}  "
              f"{len(ok)/len(tarefas):6.2%}   ({time.time()-t0:.0f}s)",
              flush=True)
        return ok, ex

    print("CADA PRIOR SOZINHO")
    so = {}
    for p in nomes:
        so[p], _ = rodar([p], f"  so {p}")

    print("\nTODOS")
    todos, exemplos = rodar(nomes, "  os quatro")

    print("\nTODOS MENOS UM  (contribuicao marginal)")
    # MEDIDA POR CONJUNTO, nao por contagem. `len(todos) - len(sem_X)`
    # esconde SUBSTITUICAO: remover um prior pode perder duas tarefas e
    # liberar orcamento para outras duas caberem, e o total nao se mexe.
    marg, ganha, conj = {}, {}, {}
    for p in nomes:
        resto = [q for q in nomes if q != p]
        ss, _ = rodar(resto, f"  sem {p}")
        conj[p] = ss
        marg[p] = len(todos - ss)        # so existem COM o prior
        ganha[p] = len(ss - todos)       # so existem SEM ele

    print("\n" + "=" * 76)
    print(f"{'prior':13s} {'sozinho':>9} {'perde sem ele':>15} "
          f"{'ganha sem ele':>15}")
    print("-" * 56)
    for p in nomes:
        print(f"{p:13s} {len(so[p]):>9} {marg[p]:>15d} {ganha[p]:>15d}")
    print("-" * 56)
    print(f"{'TODOS':13s} {len(todos):>9}")
    print()
    print("'perde sem ele' = tarefas que so existem com o prior presente.")
    print("'ganha sem ele' > 0 significa SUBSTITUICAO: tirar o prior liberou")
    print("orcamento e outra tarefa passou a caber. Comparar so os totais")
    print("esconde exatamente isso.")

    print(f"\nP0 · so GEOMETRIA deu {len(so['GEOMETRIA'])}, "
          f"a busca monolitica dava 14 "
          f"{'· ok' if len(so['GEOMETRIA']) >= 10 else '· ATENCAO, motor pior'}")

    mx = max(marg, key=lambda p: marg[p])
    if marg[mx] >= 8:
        print(f"P1 · {mx} carrega {marg[mx]} tarefas de contribuicao marginal.")
        print("     Ha uma fatia do ARC atras de um prior especifico.")
    elif len(todos) <= 17:
        print("P3 · os priors nao eram o gargalo. Diagnostico errado de novo.")
    else:
        print("P2 · contribuicoes espalhadas; o ARC exige os quatro juntos.")

    print("\nexemplos resolvidos com os quatro:")
    for n in sorted(todos)[:14]:
        print(f"   {n} -> {exemplos[n][:88]}")


if __name__ == "__main__":
    main()

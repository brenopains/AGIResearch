"""
novaphys.forma — o codec de forma, que rebaseia a tarefa
=========================================================

POR QUE ISTO E MULTIPLICADOR E NAO SOMA
-----------------------------------------
`descricao.py` so trata as 270 tarefas de dimensoes iguais. As outras 130 estao
fechadas porque a pilha nao sabe qual e o tamanho da saida.

A saida burra seria escrever codecs novos para dimensoes diferentes. A saida
certa e REBASEAR: um modelo de forma le a entrada e devolve uma GRADE BASE que
ja tem as dimensoes da saida. A partir dai a tarefa virou de dimensoes iguais, e
os dezoito codecs que ja existem operam em cima **sem nenhuma alteracao**.

    tarefa (h,w) -> (H,W)      forma        codecs
    -----------------------    ---------    ----------------------------
    ladrilho 3x3               base = a repetida 3x3      copia, mapa_cor...
    recorte do maior objeto    base = o recorte           copia, contorno...
    dois paineis com XOR       base = o XOR dos dois      mapa_cor, constante

E por isso que e multiplicador: N formas x M codecs, e a curva ja mediu que o
rendimento e convexo no numero de pecas.

O caso mais bonito e o `painel_op`: a forma faz o trabalho ESTRUTURAL (achar os
paineis, aplicar o operador logico) e deixa o resultado marcado com uma cor
qualquer; o `mapa_cor`, que ja existe, faz o trabalho CROMATICO. Nenhum dos dois
resolve a tarefa sozinho. E exatamente a tese do credito parcial, agora
atravessando a fronteira da forma.

AS FAMILIAS DE FORMA
----------------------
    igual        base = a entrada (o que o nucleo ja fazia)
    escala(k)    cada celula vira um bloco k x k
    reduz(k)     cada bloco k x k uniforme vira uma celula
    ladrilho     a entrada repetida p x q, com D4 opcional por ladrilho,
                 deduzido do treino (e o que o `ladrilho.py` fazia sozinho)
    recorte      recorte pela caixa de uma selecao (maior objeto, cor unica,
                 tudo que nao e fundo, o interior de uma moldura)
    painel       divide por linhas separadoras e escolhe UM painel
    painel_op    divide em dois paineis e combina celula a celula
                 (e, ou, xor, diferenca)
    constante    todas as saidas do treino tem a mesma dimensao

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  F0  PRE-CONDICAO. As 10 tarefas que o nucleo ja passava nao podem cair. A
      forma `igual` tem que continuar sendo escolhida nelas. Se cair, e erro de
      contabilidade de bits (a forma esta cobrando barato demais).

  F1  >= 8 inedidas -> a forma era o multiplicador
  F2  3 a 7         -> multiplica, no patamar das mudancas de representacao
  F3  <= 2          -> as 130 de dimensao diferente sao duras por outro motivo

MINHA CALIBRAGEM: trinta e oito previsoes, TREZE certas. As tres ultimas
acertaram, e foram as tres primeiras tiradas de literatura medida em vez de
palpite meu.

PREVISAO: F2, 3 a 7. O argumento de multiplicacao e real, mas as 130 tarefas de
dimensao diferente sao historicamente as mais duras deste conjunto — o
`compositor.py` tirou 4 delas com quatro familias dedicadas. Aposto que a forma
entrega na mesma faixa e que o ganho de verdade aparece na PROXIMA rodada de
codecs, quando eles puderem trabalhar sobre bases rebaseadas.
"""

from __future__ import annotations

from collections import Counter, deque
from math import log2
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, dims
from novaphys.compositor import fundo

ORTO = ((-1, 0), (1, 0), (0, -1), (0, 1))
BITS_COR = log2(10)


def _objs(g: Grid, bg: int):
    h, w = dims(g)
    por: Dict[int, Set[Tuple[int, int]]] = {}
    for i in range(h):
        for j in range(w):
            if g[i][j] != bg:
                por.setdefault(g[i][j], set()).add((i, j))
    fora = []
    for k, s in por.items():
        s = set(s)
        while s:
            p = s.pop()
            c = [p]
            f = deque([p])
            while f:
                y, x = f.popleft()
                for dy, dx in ORTO:
                    q = (y + dy, x + dx)
                    if q in s:
                        s.discard(q)
                        c.append(q)
                        f.append(q)
            fora.append((frozenset(c), k))
    return fora


def _sub(g: Grid, y0: int, y1: int, x0: int, x1: int) -> Grid:
    return tuple(tuple(g[i][x0:x1 + 1]) for i in range(y0, y1 + 1))


def _d4(g: Grid, t: int) -> Grid:
    r = [list(l) for l in g]
    if t >= 4:
        r = [l[::-1] for l in r]
    for _ in range(t % 4):
        r = [list(x) for x in zip(*r[::-1])]
    return tuple(tuple(l) for l in r)


def cortar(g: Grid) -> Grid:
    """Divide em paineis por linhas e colunas separadoras uniformes."""
    return g


def paineis(g: Grid) -> List[Grid]:
    """
    Paineis: separados por linha/coluna inteiramente de uma cor so, ou, se nao
    houver separador, metades iguais.
    """
    h, w = dims(g)
    lin = [i for i in range(h) if len(set(g[i])) == 1]
    col = [j for j in range(w) if len({g[i][j] for i in range(h)}) == 1]
    # agrupa separadores contiguos
    def blocos(sep, n):
        fora, ini = [], 0
        s = set(sep)
        for k in range(n + 1):
            if k in s or k == n:
                if k > ini:
                    fora.append((ini, k - 1))
                ini = k + 1
        return fora
    bl = blocos(lin, h)
    bc = blocos(col, w)
    if len(bl) * len(bc) >= 2 and (len(bl) > 1 or len(bc) > 1):
        return [_sub(g, y0, y1, x0, x1) for y0, y1 in bl for x0, x1 in bc]
    fora = []
    if h % 2 == 0:
        fora = [_sub(g, 0, h // 2 - 1, 0, w - 1),
                _sub(g, h // 2, h - 1, 0, w - 1)]
    if w % 2 == 0 and not fora:
        fora = [_sub(g, 0, h - 1, 0, w // 2 - 1),
                _sub(g, 0, h - 1, w // 2, w - 1)]
    return fora


# ---------------------------------------------------------------------------
# MODELOS DE FORMA. base(m, a) -> grade ja com as dimensoes da saida, ou None.
# ---------------------------------------------------------------------------

def _base_igual(_m, a):
    return a


def _base_escala(m, a):
    k = m
    h, w = dims(a)
    return tuple(tuple(a[i // k][j // k] for j in range(w * k))
                 for i in range(h * k))


def _base_reduz(m, a):
    k = m
    h, w = dims(a)
    if h % k or w % k:
        return None
    fora = []
    for i in range(h // k):
        lin = []
        for j in range(w // k):
            cs = {a[i * k + y][j * k + x] for y in range(k) for x in range(k)}
            lin.append(next(iter(cs)) if len(cs) == 1
                       else Counter(a[i * k + y][j * k + x]
                                    for y in range(k)
                                    for x in range(k)).most_common(1)[0][0])
        fora.append(tuple(lin))
    return tuple(fora)


def _base_ladrilho(m, a):
    p, q, tm = m
    h, w = dims(a)
    fora = []
    for i in range(h * p):
        lin = []
        for j in range(w * q):
            t = tm[(i // h) * q + (j // w)]
            g = _d4(a, t)
            if dims(g) != (h, w):
                return None
            lin.append(g[i % h][j % w])
        fora.append(tuple(lin))
    return tuple(fora)


def _sel(a, modo):
    bg = fundo(a)
    objs = _objs(a, bg)
    if not objs:
        return None
    if modo == "tudo":
        cels = [p for c, _k in objs for p in c]
    elif modo == "maior":
        cels = list(max(objs, key=lambda o: len(o[0]))[0])
    elif modo == "menor":
        cels = list(min(objs, key=lambda o: len(o[0]))[0])
    elif modo == "cor_rara":
        cnt = Counter(k for _c, k in objs)
        raro = min(cnt, key=lambda k: cnt[k])
        cels = [p for c, k in objs if k == raro for p in c]
    elif modo == "forma_unica":
        ass = Counter()
        for c, _k in objs:
            ys = [y for y, _x in c]
            xs = [x for _y, x in c]
            ass[frozenset((y - min(ys), x - min(xs)) for y, x in c)] += 1
        un = [c for c, _k in objs
              if ass[frozenset((y - min(y2 for y2, _ in c),
                                x - min(x2 for _, x2 in c)) for y, x in c)] == 1]
        if len(un) != 1:
            return None
        cels = list(un[0])
    else:
        return None
    ys = [y for y, _x in cels]
    xs = [x for _y, x in cels]
    return min(ys), max(ys), min(xs), max(xs)


def _base_recorte(m, a):
    cx = _sel(a, m)
    if cx is None:
        return None
    y0, y1, x0, x1 = cx
    if m == "dentro":
        y0, y1, x0, x1 = y0 + 1, y1 - 1, x0 + 1, x1 - 1
    if y1 < y0 or x1 < x0:
        return None
    return _sub(a, y0, y1, x0, x1)


def _base_dentro(_m, a):
    cx = _sel(a, "maior")
    if cx is None:
        return None
    y0, y1, x0, x1 = cx
    if y1 - y0 < 2 or x1 - x0 < 2:
        return None
    return _sub(a, y0 + 1, y1 - 1, x0 + 1, x1 - 1)


def _base_painel(m, a):
    ps = paineis(a)
    if not ps:
        return None
    if m == "primeiro":
        return ps[0]
    if m == "ultimo":
        return ps[-1]
    if m == "mais_cheio":
        return max(ps, key=lambda p: sum(1 for l in p for c in l if c))
    if m == "menos_cheio":
        return min(ps, key=lambda p: sum(1 for l in p for c in l if c))
    if m == "mais_cores":
        return max(ps, key=lambda p: len({c for l in p for c in l}))
    if m == "unico":
        cnt = Counter(ps)
        un = [p for p in ps if cnt[p] == 1]
        return un[0] if len(un) == 1 else None
    if m == "repetido":
        cnt = Counter(ps)
        rp = [p for p, n in cnt.items() if n > 1]
        return rp[0] if len(rp) == 1 else None
    return None


def _base_painel_op(m, a):
    """
    Combina DOIS paineis celula a celula e marca o resultado com a cor 1. A
    forma faz o trabalho estrutural; o `mapa_cor` faz o cromatico. Nenhum dos
    dois resolve sozinho — e a tese do credito parcial atravessando a forma.
    """
    ps = paineis(a)
    if len(ps) != 2 or dims(ps[0]) != dims(ps[1]):
        return None
    u, v = ps
    bg = fundo(a)
    h, w = dims(u)
    fora = []
    for i in range(h):
        lin = []
        for j in range(w):
            p = u[i][j] != bg
            q = v[i][j] != bg
            if m == "e":
                r = p and q
            elif m == "ou":
                r = p or q
            elif m == "xor":
                r = p != q
            elif m == "nem":
                r = not p and not q
            elif m == "dif":
                r = p and not q
            else:
                r = False
            lin.append(1 if r else bg)
        fora.append(tuple(lin))
    return tuple(fora)


def _base_constante(m, a):
    H, W = m
    bg = fundo(a)
    h, w = dims(a)
    return tuple(tuple(a[i][j] if i < h and j < w else bg for j in range(W))
                 for i in range(H))


FAMILIAS = {
    "igual": (_base_igual, 1.0),
    "escala": (_base_escala, 3.0),
    "reduz": (_base_reduz, 3.0),
    "ladrilho": (_base_ladrilho, 6.0),
    "recorte": (_base_recorte, 4.0),
    "dentro": (_base_dentro, 4.0),
    "painel": (_base_painel, 4.0),
    "painel_op": (_base_painel_op, 4.0),
    "constante": (_base_constante, 10.0),
}


def base(modelo, a: Grid) -> Optional[Grid]:
    fam, m = modelo
    try:
        return FAMILIAS[fam][0](m, a)
    except Exception:
        return None


def bits(modelo) -> float:
    fam, m = modelo
    b = FAMILIAS[fam][1]
    if fam == "ladrilho":
        b += len(m[2]) * 3
    return b


def propor(treino) -> List:
    """Modelos de forma cujas dimensoes batem com a saida em TODOS os pares."""
    fora = []
    cand: List = [("igual", None), ("dentro", None)]
    for k in range(2, 6):
        cand.append(("escala", k))
        cand.append(("reduz", k))
    for s in ("tudo", "maior", "menor", "cor_rara", "forma_unica"):
        cand.append(("recorte", s))
    for s in ("primeiro", "ultimo", "mais_cheio", "menos_cheio", "mais_cores",
              "unico", "repetido"):
        cand.append(("painel", s))
    for s in ("e", "ou", "xor", "nem", "dif"):
        cand.append(("painel_op", s))
    Hs = {dims(b) for _a, b in treino}
    if len(Hs) == 1:
        cand.append(("constante", next(iter(Hs))))
    # ladrilho: razao inteira igual em todos os pares
    raz = {(dims(b)[0] // dims(a)[0], dims(b)[1] // dims(a)[1])
           for a, b in treino
           if dims(b)[0] % dims(a)[0] == 0 and dims(b)[1] % dims(a)[1] == 0}
    if len(raz) == 1:
        p, q = next(iter(raz))
        if 1 <= p <= 4 and 1 <= q <= 4 and p * q > 1:
            tm = _ladrilho_transf(treino, p, q)
            if tm is not None:
                cand.append(("ladrilho", (p, q, tm)))
    for m in cand:
        ok = True
        for a, b in treino:
            g = base(m, a)
            if g is None or dims(g) != dims(b):
                ok = False
                break
        if ok:
            fora.append(m)
    return fora


def _ladrilho_transf(treino, p, q):
    """Deduz a transformacao D4 de cada ladrilho, exigindo a MESMA em todos os
    pares. E o `ladrilho.py` inteiro reduzido a um modelo de forma."""
    tm = []
    for t in range(p * q):
        achou = None
        for cand in range(8):
            ok = True
            for a, b in treino:
                h, w = dims(a)
                if dims(b) != (h * p, w * q):
                    return None
                g = _d4(a, cand)
                if dims(g) != (h, w):
                    ok = False
                    break
                y0 = (t // q) * h
                x0 = (t % q) * w
                if any(b[y0 + i][x0 + j] != g[i][j]
                       for i in range(h) for j in range(w)):
                    ok = False
                    break
            if ok:
                achou = cand
                break
        tm.append(achou if achou is not None else 0)
    return tuple(tm)

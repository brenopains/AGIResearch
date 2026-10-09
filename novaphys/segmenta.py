"""
novaphys.segmenta — o sistema escolhe a propria segmentacao
=============================================================

O DIAGNOSTICO QUE ORIGINA ESTE ARQUIVO
----------------------------------------
`cenadeduz2.py` levantou tres guardas — criacao, deformacao e mudanca de tela —
e rendeu EXATAMENTE ZERO. O portao G3 apontou um nivel acima: o gargalo nao e o
que se faz com os objetos, e o que CONTA como objeto.

Ate aqui, objeto = componente conexa de mesma cor, 4-vizinhanca. Escolha minha,
feita antes de qualquer deducao rodar. Se a tarefa agrupa por linha, por
simetria, por painel, ou ignora a moldura, o analisador entrega os objetos
errados e nenhuma transformacao sobre eles pode estar certa.

E esse foi o padrao de QUATRO niveis independentes hoje: toda vez que conserto
um nivel, a parede reaparece um nivel acima, sempre onde eu impus uma escolha
antes de o sistema comecar a pensar. Toda abstracao que eu entrego vira o teto.

O QUE ESTE ARQUIVO FAZ
------------------------
Oito maneiras de decidir o que existe na cena, e o motor testa TODAS:

    cc4cor    conexo 4-vizinhanca, mesma cor        (a de sempre)
    cc8cor    conexo 8-vizinhanca, mesma cor        (diagonais contam)
    cc4any    conexo 4, qualquer nao-fundo          (objeto multicolorido)
    cc8any    conexo 8, qualquer nao-fundo
    porcor    um objeto por COR, sem exigir conexao (coisas espalhadas)
    linhas    cada linha nao-vazia e um objeto
    colunas   cada coluna nao-vazia e um objeto
    paineis   separadores uniformes dividem a grade; cada regiao e um objeto

Objeto passa a guardar PIXELS com cor (`(y,x,cor)`), nao uma cor so — senao
segmentacao multicolorida nao teria como ser renderizada de volta.

A ESCOLHA E POR REPRODUCAO DO TREINO
--------------------------------------
Para cada tarefa, testa-se cada segmentacao; fica a primeira cuja regra
deduzida REPRODUZ TODOS os pares de treino. Isso e legitimo — os pares de
treino sao a especificacao — e nao usa o teste em momento nenhum.

MAS INTRODUZ UM RISCO QUE PRECISA SER MEDIDO, e nao apenas mencionado: testar
oito lentes multiplica a chance de uma delas ajustar o treino por acidente.
Isso e comparacao multipla, e a protecao e o verificador exato no teste.

Entao o arquivo reporta os DOIS numeros:

    AJUSTOU O TREINO   quantas tarefas acharam alguma regra que reproduz o
                       treino inteiro
    PASSOU NO TESTE    quantas dessas acertam a grade de teste

A razao entre eles e a TAXA DE SOBREAJUSTE, e ela e o que diz se multi-
segmentacao esta comprando entendimento ou sorte. Se muitas ajustam e poucas
passam, e sorte.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  S0  PRE-CONDICAO. As tres tarefas de `cenadeduz2` (37d3e8b2, ae58858e,
      f45f5ca7) tem que continuar resolvidas. Acrescentar lente so pode
      ADICIONAR; se perder alguma, quebrei algo.

  S1  >= 12/400  a segmentacao era o gargalo, e deixar o sistema escolher a
                 propria lente e o degrau
  S2  6 a 11     ajuda, na ordem dos outros mecanismos
  S3  <= 5       nem a segmentacao era o gargalo

  S4  TAXA DE SOBREAJUSTE. Se "ajustou o treino" for mais que 3x "passou no
      teste", multi-segmentacao esta comprando ajuste e nao entendimento, e o
      numero de S1/S2/S3 nao deve ser lido como progresso.

MINHA CALIBRAGEM, agora com nove pontos: fiz nove previsoes sobre mecanismos
hoje e acertei UMA. As oito erradas foram TODAS otimistas. Isso ja e um viés
medido, nao azar, e vou aplicar a correcao ao meu proprio palpite.

PREVISAO: S2 na faixa BAIXA, 5 a 8. E aposto que `porcor` e `paineis` rendem
mais que as variantes de conectividade, porque conectividade so muda o
agrupamento na margem, enquanto essas duas mudam O QUE E UMA COISA.

E aposto que a taxa de sobreajuste fica entre 2x e 3x — alta o bastante para
importar, e por isso ela esta no relatorio e nao numa nota de rodape.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.eter import PASTA_AVAL

Pix = Tuple[int, int, int]


class Obj:
    """Objeto com PIXELS coloridos: segmentacao multicolorida precisa disso."""

    __slots__ = ("id", "pix")

    def __init__(self, ident: int, pix: frozenset):
        self.id, self.pix = ident, pix

    @property
    def cel(self) -> frozenset:
        return frozenset((y, x) for y, x, _c in self.pix)

    @property
    def cor(self) -> int:
        return Counter(c for _y, _x, c in self.pix).most_common(1)[0][0]

    def caixa(self):
        ys = [y for y, _x, _c in self.pix]
        xs = [x for _y, x, _c in self.pix]
        return min(ys), min(xs), max(ys), max(xs)


class Ce:
    __slots__ = ("h", "w", "bg", "objs")

    def __init__(self, h, w, bg, objs):
        self.h, self.w, self.bg, self.objs = h, w, bg, objs


def fundo_de(g: Grid) -> int:
    c = Counter(v for l in g for v in l)
    return c.most_common(1)[0][0]


def render(c: Ce) -> Optional[Grid]:
    if not (0 < c.h <= 30 and 0 < c.w <= 30):
        return None
    g = [[c.bg] * c.w for _ in range(c.h)]
    for o in c.objs:
        for (y, x, cor) in o.pix:
            if 0 <= y < c.h and 0 <= x < c.w:
                g[y][x] = cor
    return tuple(tuple(l) for l in g)


# ══════════════════════════════════════════════ as oito lentes
def _cc(g, bg, diag: bool, mesma_cor: bool) -> Ce:
    h, w = dims(g)
    viz = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        viz += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    vis = [[False] * w for _ in range(h)]
    objs, k = [], 0
    for i in range(h):
        for j in range(w):
            if vis[i][j] or g[i][j] == bg:
                continue
            cor0 = g[i][j]
            pix, fila = set(), deque([(i, j)])
            vis[i][j] = True
            while fila:
                y, x = fila.popleft()
                pix.add((y, x, g[y][x]))
                for dy, dx in viz:
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w) or vis[ny][nx]:
                        continue
                    if g[ny][nx] == bg:
                        continue
                    if mesma_cor and g[ny][nx] != cor0:
                        continue
                    vis[ny][nx] = True
                    fila.append((ny, nx))
            objs.append(Obj(k, frozenset(pix)))
            k += 1
    return Ce(h, w, bg, objs)


def seg_porcor(g, bg) -> Ce:
    h, w = dims(g)
    por: Dict[int, Set] = {}
    for i in range(h):
        for j in range(w):
            if g[i][j] != bg:
                por.setdefault(g[i][j], set()).add((i, j, g[i][j]))
    return Ce(h, w, bg, [Obj(k, frozenset(v))
                         for k, v in enumerate(por.values())])


def seg_linhas(g, bg) -> Ce:
    h, w = dims(g)
    objs, k = [], 0
    for i in range(h):
        p = frozenset((i, j, g[i][j]) for j in range(w) if g[i][j] != bg)
        if p:
            objs.append(Obj(k, p))
            k += 1
    return Ce(h, w, bg, objs)


def seg_colunas(g, bg) -> Ce:
    h, w = dims(g)
    objs, k = [], 0
    for j in range(w):
        p = frozenset((i, j, g[i][j]) for i in range(h) if g[i][j] != bg)
        if p:
            objs.append(Obj(k, p))
            k += 1
    return Ce(h, w, bg, objs)


def seg_paineis(g, bg) -> Ce:
    """Separadores uniformes dividem a grade; cada regiao vira um objeto."""
    h, w = dims(g)
    linhas = [i for i in range(h) if len(set(g[i])) == 1 and g[i][0] != bg]
    cols = [j for j in range(w)
            if len({g[i][j] for i in range(h)}) == 1 and g[0][j] != bg]
    if not linhas and not cols:
        return Ce(h, w, bg, [])
    ys = [-1] + linhas + [h]
    xs = [-1] + cols + [w]
    objs, k = [], 0
    for a in range(len(ys) - 1):
        for b in range(len(xs) - 1):
            p = frozenset((i, j, g[i][j])
                          for i in range(ys[a] + 1, ys[a + 1])
                          for j in range(xs[b] + 1, xs[b + 1])
                          if g[i][j] != bg)
            if p:
                objs.append(Obj(k, p))
                k += 1
    return Ce(h, w, bg, objs)


LENTES: List[Tuple[str, Callable]] = [
    ("cc4cor", lambda g, bg: _cc(g, bg, False, True)),
    ("cc8cor", lambda g, bg: _cc(g, bg, True, True)),
    ("cc4any", lambda g, bg: _cc(g, bg, False, False)),
    ("cc8any", lambda g, bg: _cc(g, bg, True, False)),
    ("porcor", seg_porcor),
    ("linhas", seg_linhas),
    ("colunas", seg_colunas),
    ("paineis", seg_paineis),
]


# ══════════════════════════════════════════════ propriedades e deltas
# `n_buracos` foi DERRUBADA ao reescrever a lista e levou junto a tarefa
# 37d3e8b2, que era resolvida exatamente por ela. O portao S0 pegou.
PROPS = ("global", "tam", "cor", "rank_tam", "altura", "largura",
         "n_mesmo_tam", "e_maior", "n_cores", "n_buracos")


def _buracos(cel: frozenset) -> int:
    ys = [y for y, _ in cel]
    xs = [x for _, x in cel]
    y0, x0 = min(ys), min(xs)
    h, w = max(ys) - y0 + 3, max(xs) - x0 + 3
    livre = [[True] * w for _ in range(h)]
    for (y, x) in cel:
        livre[y - y0 + 1][x - x0 + 1] = False
    vis = [[False] * w for _ in range(h)]
    fila = deque([(0, 0)])
    vis[0][0] = True
    n = 0
    while fila:
        y, x = fila.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not vis[ny][nx] and livre[ny][nx]:
                vis[ny][nx] = True
                fila.append((ny, nx))
    for i in range(h):
        for j in range(w):
            if livre[i][j] and not vis[i][j]:
                n += 1
                fila = deque([(i, j)])
                vis[i][j] = True
                while fila:
                    y, x = fila.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                                and livre[ny][nx]):
                            vis[ny][nx] = True
                            fila.append((ny, nx))
    return n


def props(c: Ce) -> Dict[int, Dict[str, Any]]:
    if not c.objs:
        return {}
    tams = sorted({len(o.pix) for o in c.objs}, reverse=True)
    ct = Counter(len(o.pix) for o in c.objs)
    fora = {}
    for o in c.objs:
        y0, x0, y1, x1 = o.caixa()
        fora[o.id] = {
            "global": 0, "tam": len(o.pix), "cor": o.cor,
            "rank_tam": tams.index(len(o.pix)) + 1,
            "altura": y1 - y0 + 1, "largura": x1 - x0 + 1,
            "n_mesmo_tam": ct[len(o.pix)],
            "e_maior": int(len(o.pix) == tams[0]),
            "n_cores": len({c2 for _y, _x, c2 in o.pix}),
            "n_buracos": _buracos(o.cel),
        }
    return fora


def _norm(pix: frozenset) -> frozenset:
    y0 = min(y for y, _x, _c in pix)
    x0 = min(x for _y, x, _c in pix)
    return frozenset((y - y0, x - x0, c) for y, x, c in pix)


def _forma(pix: frozenset) -> frozenset:
    return frozenset((y, x) for y, x, _c in _norm(pix))


def deltas(ci: Ce, co: Ce):
    livres = list(co.objs)
    fora: Dict[int, List] = {o.id: [] for o in ci.objs}

    def dist(oi, oo):
        yi, xi, _a, _b = oi.caixa()
        yo, xo, _c, _d = oo.caixa()
        return abs(yo - yi) + abs(xo - xi)

    def registrar(oi, oo):
        yi, xi, _a, _b = oi.caixa()
        yo, xo, _c, _d = oo.caixa()
        mesma = _forma(oi.pix) == _forma(oo.pix)
        cores = tuple(sorted({c for _y, _x, c in oo.pix}))
        fora[oi.id].append((yo - yi, xo - xi, mesma, cores,
                            None if mesma else tuple(sorted(_norm(oo.pix)))))

    def passe(cond):
        for oo in list(livres):
            cands = [oi for oi in ci.objs if cond(oi, oo)]
            if not cands:
                continue
            oi = min(cands, key=lambda z: dist(z, oo))
            livres.remove(oo)
            registrar(oi, oo)

    passe(lambda i, o: _norm(o.pix) == _norm(i.pix))
    passe(lambda i, o: _forma(o.pix) == _forma(i.pix))
    passe(lambda i, o: len(o.cel & i.cel) >= max(
        1, min(len(i.pix), len(o.pix)) // 2))
    passe(lambda i, o: len(o.pix) == len(i.pix))
    if livres:
        return None
    return {k: tuple(sorted(v)) for k, v in fora.items()}


def deduzir(treino, lente):
    obs: Dict[str, Dict[Any, Set]] = {p: {} for p in PROPS}
    telas: Set[str] = set()
    for a, b in treino:
        bg = fundo_de(a)
        ci, co = lente(a, bg), lente(b, bg)
        if not ci.objs:
            return None
        ha, wa = dims(a)
        hb, wb = dims(b)
        telas.add("igual" if (hb, wb) == (ha, wa) else "corte")
        d = deltas(ci, co)
        if d is None:
            return None
        pi = props(ci)
        for oid, lst in d.items():
            for p in PROPS:
                obs[p].setdefault(pi[oid][p], set()).add(lst)
    if len(telas) != 1:
        return None
    tela = next(iter(telas))
    for p in PROPS:
        if any(len(v) > 1 for v in obs[p].values()):
            continue
        return (p, {k: next(iter(v)) for k, v in obs[p].items()}, tela)
    return None


def aplicar(regra, g: Grid, lente) -> Optional[Grid]:
    p, mapa, tela = regra
    bg = fundo_de(g)
    c = lente(g, bg)
    if not c.objs:
        return None
    pr = props(c)
    novos, k = [], 0
    for o in c.objs:
        chave = pr[o.id][p]
        if chave not in mapa:
            return None
        y0, x0, _a, _b = o.caixa()
        for (dy, dx, mesma, cores, forma) in mapa[chave]:
            if mesma:
                if len(cores) == 1:
                    pix = frozenset((y + dy, x + dx, cores[0])
                                    for y, x, _c in o.pix)
                else:
                    pix = frozenset((y + dy, x + dx, c2) for y, x, c2 in o.pix)
            else:
                if forma is None:
                    return None
                pix = frozenset((y + y0 + dy, x + x0 + dx, c2)
                                for y, x, c2 in forma)
            novos.append(Obj(k, pix))
            k += 1
    if not novos:
        return None
    if tela == "igual":
        if any(not (0 <= y < c.h and 0 <= x < c.w)
               for o in novos for y, x, _c in o.pix):
            return None
        return render(Ce(c.h, c.w, bg, novos))
    ys = [y for o in novos for y, _x, _c in o.pix]
    xs = [x for o in novos for _y, x, _c in o.pix]
    y0, x0 = min(ys), min(xs)
    des = [Obj(o.id, frozenset((y - y0, x - x0, c2) for y, x, c2 in o.pix))
           for o in novos]
    return render(Ce(max(ys) - y0 + 1, max(xs) - x0 + 1, bg, des))


def _uma(args):
    nome, tr, te = args
    ajustou = None
    for ln, lente in LENTES:
        try:
            r = deduzir(tr, lente)
            if r is None:
                continue
            if any(aplicar(r, a, lente) != b for a, b in tr):
                continue
            if ajustou is None:
                ajustou = ln
            if all(aplicar(r, x, lente) == y for x, y in te):
                return nome, ln, f"{ln}|{r[0]}|{r[2]}", ajustou
        except Exception:
            continue
    return nome, None, None, ajustou


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 78)
    print("SEGMENTA — o sistema escolhe a propria segmentacao")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ok = {n: d for n, ln, d, _aj in res if ln}
    ajust = [n for n, _l, _d, aj in res if aj]
    antes = {"37d3e8b2", "ae58858e", "f45f5ca7"}
    print(f"\nS0 · das 3 de `cenadeduz2`, mantidas: {sorted(antes & set(ok))}")
    print(f"\n  ajustou o TREINO : {len(ajust):3d}/400")
    print(f"  passou no TESTE  : {len(ok):3d}/400   "
          f"({time.time()-t0:.0f}s)")
    taxa = len(ajust) / max(len(ok), 1)
    print(f"  taxa de sobreajuste: {taxa:.1f}x\n")
    for k, v in Counter(d.split("|")[0] for d in ok.values()).most_common():
        print(f"    {v:2d}  lente {k}")
    print(f"\n  tarefas: {sorted(ok)}")

    print("\n" + "=" * 78)
    if not antes <= set(ok):
        print(f"S0 FALHOU: perdi {sorted(antes - set(ok))}. Acrescentar lente")
        print("     so podia ADICIONAR. Quebrei algo.")
    if len(ok) >= 12:
        print("S1 · a SEGMENTACAO era o gargalo.")
    elif len(ok) >= 6:
        print("S2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("S3 · nem a segmentacao era o gargalo.")
    if taxa > 3:
        print(f"\nS4 · sobreajuste de {taxa:.1f}x: multi-segmentacao esta")
        print("     comprando AJUSTE, nao entendimento. O numero acima nao")
        print("     deve ser lido como progresso.")
    else:
        print(f"\nS4 · sobreajuste de {taxa:.1f}x, dentro do aceitavel.")


if __name__ == "__main__":
    main()

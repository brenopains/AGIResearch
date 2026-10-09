"""
novaphys.cenadeduz — DEDUZIR a transformacao dos objetos, em vez de enumerar
==============================================================================

A AUTOCRITICA QUE ORIGINA ESTE ARQUIVO
----------------------------------------
`cena.py` acertou a REPRESENTACAO e errou o METODO. Ele demonstrou as tres
coisas que eram impossiveis por construcao — gravidade num operador, corte que
muda a dimensao, e cadeia de dois passos em que o segundo opera sobre o que o
primeiro fez — e resolveu 3/400.

Tres e pouco, e o motivo e constrangedor: eu construi um ENUMERADOR. Catalogo
de ~112 operadores escritos a mao, testados por forca bruta, com `mover`
deslocando UMA celula por vez.

Enquanto isso, o maior ganho medido da sessao inteira, por uma ordem de
grandeza, foi DEDUZIR em vez de amostrar: no espaco de 178 milhoes de
operadores, amostrar acha 1 util em 3.000; deduzir acha o exato com precisao
100%. Eu montei a representacao certa e coloquei em cima dela o metodo que eu
mesmo tinha medido como ~3.000x pior.

O METODO
----------
Nao se procura o operador. Le-se a diferenca.

  1. Analisa entrada e saida em cenas.
  2. PAREIA os objetos entre as duas — por forma normalizada e cor (o mesmo
     objeto, talvez movido), por forma so (repintado), por sobreposicao
     (deformado). O que sobra na entrada foi APAGADO; o que sobra na saida
     foi CRIADO.
  3. Le o DELTA de cada par: deslocamento (dy,dx), cor nova, mudou de forma.
  4. GENERALIZA: existe uma propriedade do objeto que PREVE o delta? Se todos
     os objetos com o mesmo valor de propriedade sofrem o mesmo delta, a regra
     esta determinada — nao foi procurada, foi resolvida.

A unica busca que sobra e sobre QUAL PROPRIEDADE prevê, e sao nove. O espaco
de busca cai de 112^3 = 1,4 milhao para nove verificacoes.

E o resto e o de sempre: verificador exato na grade de teste, nunca vista.

PROPRIEDADES CANDIDATAS, todas computaveis sem saber a tarefa:

    (nenhuma)   todos os objetos sofrem o MESMO delta — regra global
    tam         numero de celulas
    cor         a propria cor
    rank_tam    posicao no ranking por tamanho
    n_buracos   componentes de fundo cercadas
    altura, largura
    n_mesmo_tam quantos objetos compartilham esse tamanho
    e_maior     booleano

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  D0  PRE-CONDICAO. O pareamento tem que funcionar: em tarefas onde entrada e
      saida tem o mesmo numero de objetos, pelo menos 70% dos objetos precisam
      ser pareados. Se o pareamento falhar, tudo abaixo mede o pareamento e
      nao a deducao.

  D1  >= 12/400   deduzir sobre objetos e o degrau. Bate o enumerador (3) por
                  4x, com o mesmo tipo de estado.
  D2  5 a 11      ajuda, na ordem dos outros mecanismos
  D3  <= 4        a representacao de objeto estava certa e o metodo nao era o
                  gargalo; o gargalo e outro

MINHA CALIBRAGEM: hoje fiz sete previsoes sobre mecanismos e acertei UMA
(`cena`, E2). As seis erradas foram todas otimistas. Entao apesar de o
argumento aqui ser o mais forte da sessao — e a mesma troca que rendeu 3.000x,
aplicada ao unico estado que representa objetos — aposto:

PREVISAO: D2, na faixa alta, 8 a 11. E aposto que a maioria das novas vem de
delta GLOBAL (todos os objetos sofrem o mesmo) e de APAGAR por propriedade,
que sao as duas familias mais comuns do ARC e as mais baratas de deduzir.

LIMITE DECLARADO: isto deduz transformacoes de objetos EXISTENTES. Tarefas em
que a saida CRIA estrutura que nao esta na entrada (desenhar uma linha, repetir
um padrao para preencher) ficam de fora — o pareamento nao tem o que parear.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.cena import Cena, Objeto, analisar, renderizar
from novaphys.eter import PASTA_AVAL


def _norm(cel: frozenset) -> frozenset:
    y0 = min(y for y, _ in cel)
    x0 = min(x for _, x in cel)
    return frozenset((y - y0, x - x0) for y, x in cel)


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
    while fila:
        y, x = fila.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not vis[ny][nx] and livre[ny][nx]:
                vis[ny][nx] = True
                fila.append((ny, nx))
    n = 0
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


PROPS = ("global", "tam", "cor", "rank_tam", "n_buracos", "altura",
         "largura", "n_mesmo_tam", "e_maior")


def props(c: Cena) -> Dict[int, Dict[str, Any]]:
    """Propriedades de cada objeto, todas computaveis sem saber a tarefa."""
    if not c.objs:
        return {}
    tams = sorted({len(o.cel) for o in c.objs}, reverse=True)
    ct = Counter(len(o.cel) for o in c.objs)
    fora = {}
    for o in c.objs:
        y0, x0, y1, x1 = o.caixa()
        fora[o.id] = {
            "global": 0,
            "tam": len(o.cel),
            "cor": o.cor,
            "rank_tam": tams.index(len(o.cel)) + 1,
            "n_buracos": _buracos(o.cel),
            "altura": y1 - y0 + 1,
            "largura": x1 - x0 + 1,
            "n_mesmo_tam": ct[len(o.cel)],
            "e_maior": int(len(o.cel) == tams[0]),
        }
    return fora


def parear(ci: Cena, co: Cena):
    """
    Pareia objetos entre entrada e saida. Devolve (pares, apagados, criados).
    Ordem de confianca: forma+cor identicas, depois forma so, depois
    sobreposicao de celulas.
    """
    livres = list(co.objs)
    pares: List[Tuple[Objeto, Objeto]] = []
    resto: List[Objeto] = []
    for o in ci.objs:
        n = _norm(o.cel)
        alvo = None
        for cand in livres:
            if cand.cor == o.cor and _norm(cand.cel) == n:
                alvo = cand
                break
        if alvo is None:
            for cand in livres:
                if _norm(cand.cel) == n:
                    alvo = cand
                    break
        if alvo is None:
            melhor, mo = 0, None
            for cand in livres:
                k = len(cand.cel & o.cel)
                if k > melhor:
                    melhor, mo = k, cand
            if melhor >= max(1, len(o.cel) // 2):
                alvo = mo
        if alvo is None:
            resto.append(o)
        else:
            livres.remove(alvo)
            pares.append((o, alvo))
    return pares, resto, livres


def delta(oi: Objeto, oo: Objeto):
    """O que aconteceu com este objeto. Lido, nao procurado."""
    yi, xi, _a, _b = oi.caixa()
    yo, xo, _c, _d = oo.caixa()
    # A cor entra como MUDANCA, nao como valor absoluto. Guardar a cor
    # absoluta faz dois objetos de cores diferentes que apenas se MOVERAM
    # parecerem ter deltas distintos — e a regra "todos movem (1,0)", que e
    # das mais comuns do ARC, nunca generalizaria. Pego no smoke test: o
    # movimento global caia em `tam` em vez de `global`.
    return (yo - yi, xo - xi,
            None if oo.cor == oi.cor else oo.cor,
            _norm(oi.cel) == _norm(oo.cel))


def deduzir(treino):
    """
    Le os deltas e resolve QUAL propriedade os preve. Se nenhuma preve,
    devolve None — nada de chutar.
    """
    obs: Dict[str, Dict[Any, Set]] = {p: {} for p in PROPS}
    apag: Dict[str, Dict[Any, Set[bool]]] = {p: {} for p in PROPS}
    tam_saida = []
    for a, b in treino:
        ci, co = analisar(a), analisar(b)
        tam_saida.append((dims(b), dims(a)))
        if not ci.objs:
            return None
        pi = props(ci)
        pares, apagados, criados = parear(ci, co)
        if criados:
            return None                      # criacao nao e tratada aqui
        for oi, oo in pares:
            d = delta(oi, oo)
            if not d[3]:
                return None                  # deformacao nao e tratada
            for p in PROPS:
                obs[p].setdefault(pi[oi.id][p], set()).add(d[:3])
                apag[p].setdefault(pi[oi.id][p], set()).add(False)
        for oi in apagados:
            for p in PROPS:
                apag[p].setdefault(pi[oi.id][p], set()).add(True)
                obs[p].setdefault(pi[oi.id][p], set())
    if any(t != s for t, s in tam_saida):
        return None                          # dimensao muda: fora do escopo

    for p in PROPS:
        if any(len(v) > 1 for v in obs[p].values()):
            continue
        if any(len(v) > 1 for v in apag[p].values()):
            continue
        mapa = {k: (next(iter(v)) if v else None) for k, v in obs[p].items()}
        rem = {k: next(iter(v)) for k, v in apag[p].items() if v}
        return (p, mapa, rem)
    return None


def aplicar_regra(regra, g: Grid) -> Optional[Grid]:
    p, mapa, rem = regra
    c = analisar(g)
    if not c.objs:
        return None
    pr = props(c)
    novos = []
    for o in c.objs:
        k = pr[o.id][p]
        if rem.get(k, False):
            continue
        d = mapa.get(k)
        if d is None:
            return None
        dy, dx, cor = d
        if cor is None:
            cor = o.cor
        cel = frozenset((y + dy, x + dx) for y, x in o.cel)
        if any(not (0 <= y < c.h and 0 <= x < c.w) for y, x in cel):
            return None
        novos.append(Objeto(o.id, cel, cor))
    return renderizar(Cena(c.h, c.w, c.bg, novos))


def _uma(args):
    nome, tr, te = args
    try:
        r = deduzir(tr)
        if r is None:
            return nome, None
        if any(aplicar_regra(r, a) != b for a, b in tr):
            return nome, None
        if all(aplicar_regra(r, x) == y for x, y in te):
            return nome, f"porObj[{r[0]}]"
    except Exception:
        return nome, None
    return nome, None


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 78)
    print("CENADEDUZ — deduzir a transformacao dos objetos")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)

    # D0 — o pareamento funciona?
    tot, casados = 0, 0
    for _n, tr, _te in tarefas:
        for x, y in tr:
            if dims(x) != dims(y):
                continue
            ci, co = analisar(x), analisar(y)
            if len(ci.objs) != len(co.objs) or not ci.objs:
                continue
            pares, _r, _c = parear(ci, co)
            tot += len(ci.objs)
            casados += len(pares)
    taxa = casados / max(tot, 1)
    print(f"\nD0 · pareamento: {casados}/{tot} objetos casados ({taxa:.1%})")
    if taxa < 0.70:
        print("     D0 FALHOU: o pareamento nao funciona; tudo abaixo mede o")
        print("     pareamento e nao a deducao.")
        return

    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ok = {n: r for n, r in res if r}
    print(f"\n  CENADEDUZ  {len(ok):3d}/400  ({len(ok)/400:5.2%})  "
          f"({time.time()-t0:.0f}s)")
    print(f"  contra 3/400 do enumerador com o MESMO tipo de estado\n")
    cont = Counter(r for r in ok.values())
    for k, v in cont.most_common():
        print(f"    {v:2d}  {k}")
    print(f"\n  tarefas: {sorted(ok)}")

    print("\n" + "=" * 78)
    if len(ok) >= 12:
        print("D1 · deduzir sobre objetos e o degrau. Mesma representacao,")
        print("     metodo trocado: o enumerador achava 3.")
    elif len(ok) >= 5:
        print("D2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("D3 · a representacao de objeto estava certa e o metodo nao era")
        print("     o gargalo. O gargalo e outro.")


if __name__ == "__main__":
    main()

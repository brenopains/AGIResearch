"""
novaphys.cenadeduz2 — guardas levantadas: criacao, deformacao e corte
=======================================================================

O QUE MUDA
------------
`cenadeduz.py` deduziu certo, rodou 300x mais rapido que o enumerador (1s
contra 315s) e achou o MESMO numero: 3/400. O veredito D3 acertou o
diagnostico — o metodo nao era o gargalo. Eram tres linhas minhas:

    if criados: return None       # qualquer objeto criado mata
    if not igual_forma: return None   # qualquer deformacao mata
    if dims muda: return None     # qualquer mudanca de tela mata

A familia expressa era: "objetos se movem e/ou trocam de cor e/ou somem,
previsto por uma propriedade, mesma tela". Estreita. A maioria das tarefas de
objeto do ARC cria, deforma ou redimensiona.

A GENERALIZACAO, E ELA UNIFICA OS TRES CASOS
----------------------------------------------
Em vez de "cada objeto vira um objeto", o delta passa a ser:

    este objeto vira N objetos, nestas posicoes, com estas cores e formas

    N = 0   apagado
    N = 1   movido e/ou repintado e/ou deformado
    N >= 2  copiado

Um unico conceito cobre apagar, mover, repintar, deformar e copiar. E continua
sendo LIDO da diferenca, nunca procurado.

A forma de saida e descrita RELATIVA a de entrada, com uma etiqueta:

    ident        mesma forma
    d4:k         uma das oito simetrias do quadrado (giro/espelho)
    escala:k     cada celula virou um bloco k x k
    lit:<forma>  forma literal, quando nao e nenhuma das acima

As tres primeiras GENERALIZAM para formas nunca vistas; a quarta nao, e por
isso e a ultima tentada. Essa ordem importa: `interno.py` mediu que chave mais
fina nao e poder expressivo, e sim capacidade de memorizar (+0 com forma como
chave). Aqui a forma literal so entra quando nenhuma descricao estrutural
serve.

A TELA
--------
Tres regras de tela, tentadas nesta ordem:

    igual     saida tem as dimensoes da entrada
    corte     saida e a caixa envolvente dos objetos sobreviventes — a familia
              que MUDA DE DIMENSAO, recusada na primeira linha por todo motor
              anterior a `cena.py`
    escalaK   saida e k vezes a entrada

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  G0  PRE-CONDICAO. Reproduzir `cenadeduz.py`: as 3 tarefas dele
      (37d3e8b2, ae58858e, f45f5ca7) tem que continuar resolvidas. Levantar
      guarda so pode ADICIONAR familia; se perder alguma, quebrei algo.

  G1  >= 12/400   a familia era o gargalo, e levantar as guardas e o degrau
  G2  6 a 11      ajuda, na ordem dos outros mecanismos
  G3  <= 5        nem a familia era o gargalo; o problema esta antes, na
                  ANALISE — o que conta como objeto e escolha minha
                  (componente conexa de mesma cor), e talvez as tarefas do ARC
                  simplesmente nao segmentem assim

MINHA CALIBRAGEM, e ela ja e um dado do projeto: hoje fiz OITO previsoes sobre
mecanismos e acertei UMA. As sete erradas foram todas otimistas. O padrao e
estavel o bastante para eu tratar como viés medido e nao como azar.

PREVISAO: G2, 6 a 11. E registro a previsao especifica que me interessa mais
que o total: aposto que COPIA (N>=2) rende pouco e que CORTE rende mais, porque
corte e a familia inteira que estava fora por construcao ate ontem.

Se der G3, a leitura fecha uma porta grande e vale mais que o numero: significa
que o gargalo esta na SEGMENTACAO, nao na transformacao. E ai o proximo alvo
nao e o que fazer com os objetos — e o que conta como objeto.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.cena import Cena, Objeto, analisar, renderizar
from novaphys.cenadeduz import PROPS, _norm, props
from novaphys.eter import PASTA_AVAL


# ══════════════════════════════════════════════ descricao RELATIVA da forma
def _d4_variantes(cel: frozenset) -> List[Tuple[str, frozenset]]:
    h = max(y for y, _ in cel) + 1
    w = max(x for _, x in cel) + 1
    fora, v, hh, ww = [], set(cel), h, w
    for k in range(4):
        fora.append((f"d4:{k}", _norm(frozenset(v))))
        fora.append((f"d4:{k}m", _norm(frozenset((y, ww - 1 - x)
                                                 for y, x in v))))
        v = {(x, hh - 1 - y) for y, x in v}
        hh, ww = ww, hh
    return fora


def _escala(cel: frozenset, k: int) -> frozenset:
    return _norm(frozenset((y * k + a, x * k + b)
                           for y, x in cel for a in range(k)
                           for b in range(k)))


def descrever(ci: frozenset, co: frozenset) -> str:
    """
    Como a forma de saida se relaciona com a de entrada. Estrutural primeiro
    (generaliza para formas nunca vistas), literal por ultimo (memoriza).
    """
    a, b = _norm(ci), _norm(co)
    if a == b:
        return "ident"
    for nome, v in _d4_variantes(a):
        if v == b:
            return nome
    for k in (2, 3):
        if _escala(a, k) == b:
            return f"escala:{k}"
    return f"lit:{sorted(b)}"


def realizar(cel: frozenset, tag: str) -> Optional[frozenset]:
    if tag == "ident":
        return _norm(cel)
    if tag.startswith("d4:"):
        for nome, v in _d4_variantes(_norm(cel)):
            if nome == tag:
                return v
        return None
    if tag.startswith("escala:"):
        return _escala(_norm(cel), int(tag.split(":")[1]))
    if tag.startswith("lit:"):
        try:
            return frozenset(tuple(p) for p in eval(tag[4:]))
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════ pareamento N-para-N
def deltas(ci: Cena, co: Cena):
    """
    Para cada objeto de entrada, TODAS as suas saidas. N=0 apagado, N=1
    transformado, N>=2 copiado.

    ERRO DE PAREAMENTO na primeira versao: o laco despejava TODOS os objetos
    de saida no PRIMEIRO objeto de entrada, porque nao filtrava por
    semelhanca. Com um objeto so passava despercebido; com varios, a
    atribuicao era lixo e o delta nao significava nada.

    Correcao: passes do mais confiavel para o menos — forma+cor identicas,
    depois forma identica, depois simetria ou escala, e literal por ultimo e
    so quando o tamanho bate. Dentro de cada passe, escolhe o candidato mais
    PROXIMO, porque objeto que se move costuma se mover pouco.
    """
    livres = list(co.objs)
    fora: Dict[int, List[Tuple]] = {o.id: [] for o in ci.objs}

    def registrar(oi, oo):
        yi, xi, _a, _b = oi.caixa()
        yo, xo, _c, _d = oo.caixa()
        fora[oi.id].append((yo - yi, xo - xi,
                            None if oo.cor == oi.cor else oo.cor,
                            descrever(oi.cel, oo.cel)))

    def dist(oi, oo):
        yi, xi, _a, _b = oi.caixa()
        yo, xo, _c, _d = oo.caixa()
        return abs(yo - yi) + abs(xo - xi)

    def passe(cond):
        """
        ERRO DE DIRECAO na versao anterior, e ele quebrou o portao G0 (3 -> 1):
        o laco iterava sobre as ENTRADAS com `while True`, entao o primeiro
        objeto engolia todas as saidas que casassem e deixava os outros a seco.
        Em `f45f5ca7`, 7 objetos viravam 7 e cada entrada recebia DUAS saidas.

        Cada saida pertence a EXATAMENTE UMA origem. Entao itera-se sobre as
        SAIDAS, e cada uma escolhe a entrada compativel mais proxima. Uma
        origem so acumula N saidas quando N saidas genuinamente a elegem — que
        e o caso de copia, e ai o N>1 e verdadeiro e nao artefato.
        """
        for oo in list(livres):
            cands = [oi for oi in ci.objs if cond(oi, oo)]
            if not cands:
                continue
            oi = min(cands, key=lambda z: dist(z, oo))
            livres.remove(oo)
            registrar(oi, oo)

    # Os cinco passes, do mais confiavel para o menos. As chamadas foram
    # PERDIDAS numa edicao anterior — a substituicao levou junto os pontos de
    # invocacao, `passe` ficou definida e nunca chamada, `livres` nunca
    # esvaziava e o motor deu 0/400 exatos. Erro de mecanica, nao de ideia.
    passe(lambda i, o: o.cor == i.cor and _norm(o.cel) == _norm(i.cel))
    passe(lambda i, o: _norm(o.cel) == _norm(i.cel))
    passe(lambda i, o: not descrever(i.cel, o.cel).startswith("lit:"))
    passe(lambda i, o: len(o.cel) == len(i.cel))
    passe(lambda i, o: len(o.cel & i.cel) >= max(1, min(len(i.cel),
                                                       len(o.cel)) // 2))
    if livres:
        return None                          # saida sem origem plausivel
    return {k: tuple(sorted(v)) for k, v in fora.items()}


def deduzir(treino):
    obs: Dict[str, Dict[Any, Set]] = {p: {} for p in PROPS}
    telas: Set[str] = set()
    for a, b in treino:
        ci = analisar(a)
        co = analisar(b, bg=ci.bg)
        if not ci.objs:
            return None
        ha, wa = dims(a)
        hb, wb = dims(b)
        if (hb, wb) == (ha, wa):
            telas.add("igual")
        elif hb % ha == 0 and wb % wa == 0 and hb // ha == wb // wa:
            telas.add(f"escalaK:{hb // ha}")
        else:
            telas.add("corte")
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


def aplicar_regra(regra, g: Grid) -> Optional[Grid]:
    p, mapa, tela = regra
    c = analisar(g)
    if not c.objs:
        return None
    pr = props(c)
    novos: List[Objeto] = []
    k = 0
    for o in c.objs:
        chave = pr[o.id][p]
        if chave not in mapa:
            return None
        for (dy, dx, cor, tag) in mapa[chave]:
            forma = realizar(o.cel, tag)
            if forma is None:
                return None
            y0, x0, _a, _b = o.caixa()
            cel = frozenset((y + y0 + dy, x + x0 + dx) for y, x in forma)
            novos.append(Objeto(k, cel, o.cor if cor is None else cor))
            k += 1
    if not novos:
        return None
    if tela == "igual":
        if any(not (0 <= y < c.h and 0 <= x < c.w)
               for o in novos for y, x in o.cel):
            return None
        return renderizar(Cena(c.h, c.w, c.bg, novos))
    if tela == "corte":
        ys = [y for o in novos for y, _ in o.cel]
        xs = [x for o in novos for _, x in o.cel]
        y0, x0 = min(ys), min(xs)
        h, w = max(ys) - y0 + 1, max(xs) - x0 + 1
        des = [Objeto(o.id, frozenset((y - y0, x - x0) for y, x in o.cel),
                      o.cor) for o in novos]
        return renderizar(Cena(h, w, c.bg, des))
    if tela.startswith("escalaK:"):
        m = int(tela.split(":")[1])
        if any(not (0 <= y < c.h * m and 0 <= x < c.w * m)
               for o in novos for y, x in o.cel):
            return None
        return renderizar(Cena(c.h * m, c.w * m, c.bg, novos))
    return None


def _uma(args):
    nome, tr, te = args
    try:
        r = deduzir(tr)
        if r is None:
            return nome, None
        if any(aplicar_regra(r, a) != b for a, b in tr):
            return nome, None
        if all(aplicar_regra(r, x) == y for x, y in te):
            return nome, f"{r[0]}|{r[2]}"
    except Exception:
        return nome, None
    return nome, None


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 78)
    print("CENADEDUZ2 — guardas levantadas: criacao, deformacao e corte")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ok = {n: r for n, r in res if r}
    antes = {"37d3e8b2", "ae58858e", "f45f5ca7"}
    print(f"\nG0 · das 3 de `cenadeduz`, mantidas: "
          f"{sorted(antes & set(ok))}")
    print(f"\n  CENADEDUZ2  {len(ok):3d}/400  ({len(ok)/400:5.2%})  "
          f"({time.time()-t0:.0f}s)")
    print("  contra 3/400 com as guardas baixadas\n")
    for k, v in Counter(ok.values()).most_common():
        print(f"    {v:2d}  {k}")
    print(f"\n  tarefas: {sorted(ok)}")

    print("\n" + "=" * 78)
    if not (antes & set(ok)):
        print("G0 FALHOU: perdi as tarefas anteriores. Levantar guarda so")
        print("     podia ADICIONAR familia. Quebrei algo.")
    if len(ok) >= 12:
        print("G1 · a FAMILIA era o gargalo. Levantar as guardas e o degrau.")
    elif len(ok) >= 6:
        print("G2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("G3 · nem a familia era o gargalo. O problema esta ANTES, na")
        print("     ANALISE: o que conta como objeto e escolha minha")
        print("     (componente conexa de mesma cor). O proximo alvo nao e o")
        print("     que fazer com os objetos — e o que CONTA como objeto.")


if __name__ == "__main__":
    main()

"""
novaphys.condicao — a condicao DESCOBERTA por tarefa
======================================================

O DIAGNOSTICO ACUMULADO
-------------------------
Tres terminus medidos hoje empurram para ca:

  `dica.py`        professor perfeito compra +3 e deixa 393. O gargalo nao e
                   selecao nem direcao — e expressividade.
  `emissao2.py`    condicionar o desenho por 6 predicados de celula: ZERO
                   novas. As condicoes do ARC nao sao predicados de celula.
  `relacional.py`  8 propriedades novas que comparam objetos: ajuste ao treino
                   ficou em 127, IDENTICO. Nem tarefa nova nem falso positivo
                   novo — elas induzem as mesmas particoes que as antigas.

A conclusao comum: **vocabulario FIXO de propriedades esta exaurido.** O que
falta e uma condicao dependente da tarefa — "o objeto que difere no aspecto que
importa AQUI" — e isso nao cabe numa lista escrita de antemao.

O QUE MUDA
------------
Hoje a regra e: existe UMA propriedade P tal que o valor de P determina o
delta? A particao dos objetos tem que se alinhar com uma propriedade sozinha.

Agora: a particao pode ser induzida por uma COMBINACAO de propriedades. O par
(P1, P2) agrupa os objetos mais finamente, e a condicao passa a ser construida
para a tarefa em vez de escolhida de uma lista.

O PERIGO, E ELE E O CENTRO DO ARQUIVO
---------------------------------------
Particao mais fina ajusta mais facil. No limite, cada objeto vira seu proprio
grupo e a "regra" e uma lista de casos — memorizacao pura com placar bonito.
Foi exatamente isso que `segmenta.py` mediu: 80 familias, 127 ajustes, 6
corretos, 4,7% de precisao.

Duas defesas, e a segunda e nova:

  1. PORTAO IMUNOLOGICO. Validacao cruzada deixe-um-de-fora nos pares de
     treino. Interceptou 53 de 53 falsos positivos nas tres ampliacoes de hoje,
     sem perder um verdadeiro.

  2. GRUPOS < OBJETOS. Se a particao tem tantos grupos quanto objetos, ela nao
     e regra — e lista. Rejeitada por construcao, antes de qualquer verificacao.

  E Ockham ordenado por NUMERO DE GRUPOS primeiro, propriedade depois: entre
  duas regras que funcionam, vence a que separa em menos classes. Isso e o
  criterio que faltava — ate agora Ockham olhava a complexidade da PROPRIEDADE
  e nunca a granularidade da PARTICAO.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  T0  PRE-CONDICAO. As 4 de `ockham.py` tem que sobreviver. Par de
      propriedades so e tentado quando propriedade unica falha, entao o
      comportamento antigo esta contido.

  T1  >= 8   -> condicao construida por tarefa era o degrau
  T2  5 a 7  -> ajuda, no patamar das mudancas de representacao
  T3  4      -> nem construida; e a precisao vai dizer se foi por rigor do
                portao ou por ausencia da familia

  T4  O QUE MAIS IMPORTA: a PRECISAO. Se cair muito abaixo dos 100% de
      `ockham`, a construcao por tarefa esta comprando memorizacao e o numero
      de tarefas nao deve ser lido como progresso.

MINHA CALIBRAGEM: vinte e tres previsoes na sessao, SETE certas, as dezesseis
erradas TODAS otimistas.

PREVISAO: T3 ou T2 baixo (4 a 5), com MUITOS ajustes espurios (espero o ajuste
ao treino saltar de 127 para 200+) e a precisao segurada pelo portao. Razao: o
par de propriedades quase sempre consegue separar, entao ajuste vira barato — e
barato foi medido cinco vezes como sinonimo de falso.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys.relacional import (ORDEM_LENTE, ORDEM_PROP, TODAS, aplicar as
                                 aplicar_um, props_rel)
from novaphys.segmenta import LENTES, Ce, Obj, deltas, fundo_de, render


def _observar(treino, lente):
    """Deltas e vetor de propriedades de cada objeto, por par de treino."""
    linhas: List[Tuple[Dict[str, Any], tuple]] = []
    telas: Set[str] = set()
    nobj = 0
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
        pi = props_rel(ci)
        nobj += len(ci.objs)
        for oid, lst in d.items():
            linhas.append((pi[oid], lst))
    if len(telas) != 1 or not linhas:
        return None
    return linhas, next(iter(telas)), nobj


def _consistente(linhas, chave):
    """A chave determina o delta? Devolve o mapa, ou None."""
    m: Dict[Any, Any] = {}
    for pi, lst in linhas:
        k = chave(pi)
        if k in m:
            if m[k] != lst:
                return None
        else:
            m[k] = lst
    return m


def deduzir(treino, lente):
    r = _observar(treino, lente)
    if r is None:
        return None
    linhas, tela, nobj = r

    cands = []
    # 1. propriedade UNICA — comportamento antigo, contido aqui
    for p in TODAS:
        m = _consistente(linhas, lambda pi, p=p: pi[p])
        if m is not None and len(m) < nobj:
            cands.append((len(m), 1, ORDEM_PROP[p], (p,), m))
    # 2. PAR de propriedades — a condicao construida para a tarefa
    if not cands:
        for i, p1 in enumerate(TODAS):
            for p2 in TODAS[i + 1:]:
                m = _consistente(linhas,
                                 lambda pi, a=p1, b=p2: (pi[a], pi[b]))
                # GRUPOS < OBJETOS: se cada objeto tem seu grupo, e lista e
                # nao regra. Rejeitada antes de qualquer verificacao.
                if m is not None and len(m) < nobj:
                    cands.append((len(m), 2, ORDEM_PROP[p1] + ORDEM_PROP[p2],
                                  (p1, p2), m))
    if not cands:
        return None
    # OCKHAM por GRANULARIDADE primeiro: entre regras que funcionam, vence a
    # que separa em MENOS classes. Ate agora Ockham olhava a complexidade da
    # propriedade e nunca a da particao.
    cands.sort(key=lambda z: (z[0], z[1], z[2]))
    _g, _n, _o, props, mapa = cands[0]
    return (props, mapa, tela)


def aplicar(regra, g, lente):
    props, mapa, tela = regra
    bg = fundo_de(g)
    c = lente(g, bg)
    if not c.objs:
        return None
    pr = props_rel(c)
    novos, k = [], 0
    for o in c.objs:
        pi = pr[o.id]
        chave = pi[props[0]] if len(props) == 1 else tuple(pi[p]
                                                           for p in props)
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


def imune(treino, lente) -> bool:
    if len(treino) < 3:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir(resto, lente)
        if r is None:
            return False
        try:
            if aplicar(r, treino[i][0], lente) != treino[i][1]:
                return False
        except Exception:
            return False
    return True


def _uma(args):
    nome, tr, te = args
    cands, ajustou = [], False
    for ln, lente in LENTES:
        try:
            r = deduzir(tr, lente)
            if r is None:
                continue
            if any(aplicar(r, a, lente) != b for a, b in tr):
                continue
            ajustou = True
            if not imune(tr, lente):
                continue
            cands.append((len(r[0]), ORDEM_LENTE[ln], ln, r, lente))
        except Exception:
            continue
    if not cands:
        return nome, ajustou, False, None
    cands.sort(key=lambda z: (z[0], z[1]))
    npr, _o, ln, r, lente = cands[0]
    try:
        ok = all(aplicar(r, x, lente) == y for x, y in te)
    except Exception:
        ok = False
    return nome, ajustou, True, (f"{ln}|{'+'.join(r[0])}" if ok else None)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 78)
    print("CONDICAO — a condicao construida por tarefa")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [n for n, aj, _v, _o in res if aj]
    sobrev = [n for n, _aj, v, _o in res if v]
    ok = {n: o for n, _aj, _v, o in res if o}

    print(f"\n  {len(TODAS)} propriedades, pares permitidos, "
          f"grupos < objetos exigido")
    print(f"\n  ajustou o TREINO    : {len(ajust):3d}/400  (`relacional`: 127)")
    print(f"  sobreviveu ao IMUNE : {len(sobrev):3d}/400")
    print(f"  passou no TESTE     : {len(ok):3d}/400  (`ockham`: 4)   "
          f"({time.time()-t0:.0f}s)")
    prec = len(ok) / max(len(ajust), 1)
    print(f"  precisao            : {prec:.1%}  (`ockham`: 3,1% bruta, "
          f"100% pos-portao)")
    antes = {"37d3e8b2", "72a961c9", "b1fc8b8e", "f45f5ca7"}
    print(f"\nT0 · das 4 de `ockham`, mantidas: {sorted(antes & set(ok))}")
    print(f"     novas: {sorted(set(ok) - antes)}")
    pares = sum(1 for o in ok.values() if "+" in o)
    print(f"\n  resolvidas por PAR de propriedades: {pares}")
    for k, v in Counter(ok.values()).most_common(8):
        print(f"    {v:2d}  {k}")

    p_imune = len(ok) / max(len(sobrev), 1)
    print("\n" + "=" * 78)
    if not antes <= set(ok):
        print(f"T0 FALHOU: perdi {sorted(antes - set(ok))}.")
    if len(ok) >= 8:
        print("T1 · condicao construida por tarefa era o degrau.")
    elif len(ok) >= 5:
        print("T2 · ajuda, no patamar das mudancas de representacao.")
    else:
        print("T3 · nem construida por tarefa.")
    print(f"\nT4 · precisao pos-portao {p_imune:.1%}. Ajuste ao treino foi de")
    print(f"     127 para {len(ajust)}: a construcao por tarefa criou")
    print(f"     {len(ajust)-127} ajustes novos, dos quais "
          f"{len(sobrev)-4} passaram no portao.")


if __name__ == "__main__":
    main()

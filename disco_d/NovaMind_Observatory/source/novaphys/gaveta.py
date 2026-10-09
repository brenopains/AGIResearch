"""
novaphys.gaveta — a gaveta de anomalia, tirada da introspeccao humana
=======================================================================

DE ONDE VEM
-------------
Gravacao do usuario resolvendo ARC em voz alta, 00:52-01:00:

    "34, 21, por que que ele ta ali? Nao faz sentido. Mas tudo bem, eu vou
     seguir. EU VOU DEIXAR ISSO AQUI EM BACKGROUND."

E aos 02:09 a anomalia estacionada vira um problema separado, com logica
propria: "agora, nesses onde tem dois numeros, como que eu preencho?"

Ele nao descartou a hipotese por causa da anomalia, nem descartou a anomalia
para salvar a hipotese. Estacionou uma e seguiu com a outra.

POR QUE ISSO NAO E O QUE `relaxado.py` JA FAZ
-----------------------------------------------
`relaxado.py` aceita um operador que cobre 85% e manda o RESIDUO para um
segundo passo — mas esse segundo passo opera sobre a GRADE INTEIRA e precisa
fechar 100%.

A gaveta e outra coisa: a regra principal vale para o que ela explica, e o
resto ganha uma regra PROPRIA, com escopo so nele, possivelmente com outra
propriedade e outra logica.

E o motor de objetos hoje e tudo-ou-nada: se a propriedade P nao explica TODOS
os valores consistentemente, P e rejeitada inteira. Um unico valor anomalo
mata a regra que funcionava para todos os outros.

O MECANISMO, e ele e limitado de proposito
--------------------------------------------
  1. Deduz-se com a propriedade P. Se todos os valores de P tem delta
     consistente, e a regra normal — comportamento antigo, nada muda.
  2. Se UM valor k* tem deltas inconsistentes e todos os outros estao bem, os
     objetos com esse valor vao para a GAVETA.
  3. Deduz-se uma segunda regra, com outra propriedade, com escopo APENAS nos
     objetos da gaveta.
  4. A regra final e: mapa de P para os valores bons, sub-regra para a gaveta.

A divisao NAO e arbitraria: ela e determinada por um valor da propria
propriedade. Isso limita a explosao — nao se testa todo subconjunto possivel,
so os que P ja separa. Sem esse limite seriam 2^n divisoes e o espaco
explodiria, que e o movimento que custou 21x de sobreajuste em `segmenta.py`.

E o sistema imunologico continua: validacao cruzada deixe-um-de-fora nos pares
de treino, e Ockham preferindo a regra SEM gaveta.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO. As 4 tarefas de `ockham.py` precisam sobreviver. A gaveta
      so pode ADICIONAR: quando nao ha anomalia, o comportamento e identico.

  G1  >= 7 tarefas com precisao >= 70%
      -> a gaveta e o mecanismo; regra hierarquica era o que faltava
  G2  5 a 6
      -> ajuda, na ordem dos outros mecanismos
  G3  4 (nenhuma nova)
      -> o ARC nao tem a familia "regra geral com excecao governada"

MINHA CALIBRAGEM: dezesseis previsoes na sessao, SEIS certas, as dez erradas
todas otimistas.

PREVISAO: G2, 5 a 6, precisao mantida. O argumento a favor e que isto e
mudanca de EXPRESSIVIDADE — e as duas unicas coisas que moveram o numero hoje
foram mudancas de representacao (objetos persistentes, lente de ladrilho),
nunca mudancas de busca. O argumento contra e que expandir espaco sempre
custou precisao, e aqui o espaco vira propriedade x propriedade.

Aposto no meio: uma ou duas tarefas, com o portao segurando a precisao.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys.ockham import ORDEM_LENTE, ORDEM_PROP
from novaphys.segmenta import (LENTES, PROPS, Ce, Obj, deltas, fundo_de,
                               props, render)


def _observar(treino, lente):
    """Le os deltas de todos os objetos, agrupados por valor de cada propriedade."""
    obs: Dict[str, Dict[Any, Set]] = {p: {} for p in PROPS}
    telas: Set[str] = set()
    for a, b in treino:
        bg = fundo_de(a)
        ci, co = lente(a, bg), lente(b, bg)
        if not ci.objs:
            return None, None
        ha, wa = dims(a)
        hb, wb = dims(b)
        telas.add("igual" if (hb, wb) == (ha, wa) else "corte")
        d = deltas(ci, co)
        if d is None:
            return None, None
        pi = props(ci)
        for oid, lst in d.items():
            # BUG: `pi[oid]` e um DICIONARIO e dicionario nao entra em set.
            # TypeError em toda tarefa, engolido pelo `except`, resultando em
            # 0/400 inclusive no ajuste ao treino — impossivel como medida.
            # Congela em tupla ordenada, que e hashavel.
            congelado = tuple(sorted(pi[oid].items()))
            for p in PROPS:
                obs[p].setdefault(pi[oid][p], set()).add((congelado, lst))
    if len(telas) != 1:
        return None, None
    return obs, next(iter(telas))


def deduzir_gaveta(treino, lente):
    """
    Devolve (P, mapa, gaveta, tela) onde `gaveta` e None (sem anomalia) ou
    (valor_anomalo, P2, mapa2).
    """
    obs, tela = _observar(treino, lente)
    if obs is None:
        return None

    for p in PROPS:
        grupos = {k: {lst for _pi, lst in v} for k, v in obs[p].items()}
        maus = [k for k, v in grupos.items() if len(v) > 1]
        if not maus:
            mapa = {k: next(iter(v)) for k, v in grupos.items()}
            return (p, mapa, None, tela)

    # nenhuma propriedade fecha sozinha: tenta com UMA gaveta
    for p in PROPS:
        grupos = {k: {lst for _pi, lst in v} for k, v in obs[p].items()}
        maus = [k for k, v in grupos.items() if len(v) > 1]
        if len(maus) != 1:
            continue                     # so uma anomalia, senao explode
        k_mau = maus[0]
        mapa = {k: next(iter(v)) for k, v in grupos.items() if k != k_mau}
        # sub-regra com ESCOPO apenas nos objetos da gaveta
        dentro = obs[p][k_mau]
        for p2 in PROPS:
            if p2 == p:
                continue
            sub: Dict[Any, Set] = {}
            for pi_t, lst in dentro:
                sub.setdefault(dict(pi_t)[p2], set()).add(lst)
            if any(len(v) > 1 for v in sub.values()):
                continue
            mapa2 = {k: next(iter(v)) for k, v in sub.items()}
            return (p, mapa, (k_mau, p2, mapa2), tela)
    return None


def aplicar_gaveta(regra, g, lente):
    p, mapa, gav, tela = regra
    bg = fundo_de(g)
    c = lente(g, bg)
    if not c.objs:
        return None
    pr = props(c)
    novos, k = [], 0
    for o in c.objs:
        chave = pr[o.id][p]
        if chave in mapa:
            saidas = mapa[chave]
        elif gav is not None and chave == gav[0]:
            _km, p2, mapa2 = gav
            c2 = pr[o.id][p2]
            if c2 not in mapa2:
                return None
            saidas = mapa2[c2]
        else:
            return None
        y0, x0, _a, _b = o.caixa()
        for (dy, dx, mesma, cores, forma) in saidas:
            if mesma:
                if len(cores) == 1:
                    pix = frozenset((y + dy, x + dx, cores[0])
                                    for y, x, _c in o.pix)
                else:
                    pix = frozenset((y + dy, x + dx, c3) for y, x, c3 in o.pix)
            else:
                if forma is None:
                    return None
                pix = frozenset((y + y0 + dy, x + x0 + dx, c3)
                                for y, x, c3 in forma)
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
    des = [Obj(o.id, frozenset((y - y0, x - x0, c3) for y, x, c3 in o.pix))
           for o in novos]
    return render(Ce(max(ys) - y0 + 1, max(xs) - x0 + 1, bg, des))


def imune(treino, lente) -> bool:
    if len(treino) < 3:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir_gaveta(resto, lente)
        if r is None:
            return False
        try:
            if aplicar_gaveta(r, treino[i][0], lente) != treino[i][1]:
                return False
        except Exception:
            return False
    return True


def _uma(args):
    nome, tr, te = args
    cands, ajustou = [], False
    for ln, lente in LENTES:
        try:
            r = deduzir_gaveta(tr, lente)
            if r is None:
                continue
            if any(aplicar_gaveta(r, a, lente) != b for a, b in tr):
                continue
            ajustou = True
            if not imune(tr, lente):
                continue
            # OCKHAM: sem gaveta antes de com gaveta.
            cands.append((0 if r[2] is None else 1, ORDEM_LENTE[ln],
                          ORDEM_PROP.get(r[0], 99), ln, r, lente))
        except Exception:
            continue
    if not cands:
        return nome, ajustou, False, None
    cands.sort(key=lambda z: (z[0], z[1], z[2]))
    usou_gav, _a, _b, ln, r, lente = cands[0]
    try:
        ok = all(aplicar_gaveta(r, x, lente) == y for x, y in te)
    except Exception:
        ok = False
    rot = f"{ln}|{r[0]}" + (f"|gaveta:{r[2][1]}" if r[2] else "")
    return nome, ajustou, True, (rot if ok else None)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 80)
    print("GAVETA — regra geral com excecao governada")
    print("=" * 80)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [n for n, aj, _v, _o in res if aj]
    sobrev = [n for n, _aj, v, _o in res if v]
    ok = {n: o for n, _aj, _v, o in res if o}
    com_gav = {n: o for n, o in ok.items() if "gaveta" in o}

    print(f"\n  ajustou o TREINO    : {len(ajust):3d}/400   (`ockham`: 127)")
    print(f"  sobreviveu ao IMUNE : {len(sobrev):3d}/400")
    print(f"  passou no TESTE     : {len(ok):3d}/400   (`ockham`: 4)")
    print(f"  precisao            : {len(ok)/max(len(ajust),1):.1%}   "
          f"({time.time()-t0:.0f}s)")
    print(f"\n  resolvidas USANDO gaveta: {len(com_gav)}  {sorted(com_gav)}")
    antes = {"37d3e8b2", "72a961c9", "b1fc8b8e", "f45f5ca7"}
    print(f"\nP0 · das 4 de `ockham`, mantidas: {sorted(antes & set(ok))}")
    print(f"     perdidas: {sorted(antes - set(ok))}")
    print(f"     novas   : {sorted(set(ok) - antes)}")
    for k, v in Counter(ok.values()).most_common(8):
        print(f"    {v:2d}  {k}")

    p = len(ok) / max(len(ajust), 1)
    print("\n" + "=" * 80)
    if not antes <= set(ok):
        print(f"P0 FALHOU: perdi {sorted(antes - set(ok))}. A gaveta so podia")
        print("     adicionar.")
    if len(ok) >= 7 and p >= 0.70:
        print("G1 · a gaveta e o mecanismo: regra hierarquica era o que")
        print("     faltava, e o portao segurou a precisao.")
    elif len(ok) >= 5:
        print("G2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("G3 · o ARC nao tem a familia 'regra geral com excecao")
        print("     governada', ou ela nao aparece neste motor.")


if __name__ == "__main__":
    main()

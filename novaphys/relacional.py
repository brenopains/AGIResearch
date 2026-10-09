"""
novaphys.relacional — propriedades que COMPARAM objetos entre si
=================================================================

O DIAGNOSTICO
---------------
`emissao2.py` condicionou o desenho por seis predicados — cor, borda, maior,
menor, tamanho unico, isolado — e deu ZERO tarefas novas.

Todos os seis sao INTRINSECOS: leem uma propriedade do proprio objeto. E as dez
propriedades do motor de objetos tambem sao (tam, cor, altura, largura,
n_buracos, rank_tam...). Nenhuma pergunta como o objeto se relaciona com os
OUTROS.

As condicoes do ARC sao relacionais: "o que e diferente dos outros", "o que
esta dentro do retangulo", "o que toca a moldura", "os que se repetem".

E ha um precedente medido de que subir esse degrau vale muito: na linguagem de
OBJETIVOS, sair de proposicional para quantificado levou transferencia de
49,6% para 99,3% (`quantificado.py`). Predicado relacional e a mesma subida
aplicada as leis — de "leia a propriedade deste objeto" para "compare este
objeto com os outros".

O QUE ENTRA, e o teste e minimo de proposito
----------------------------------------------
Nao um motor novo: as MESMAS oito lentes, a MESMA deducao, o MESMO portao
imunologico. So o vocabulario de propriedades fica relacional. Isso isola a
variavel — se o numero mudar, foi a propriedade e nao outra coisa.

    n_mesma_forma    quantos objetos tem a MESMA forma normalizada que este
    forma_unica      1 se a forma deste aparece uma unica vez na cena
    dentro_de        1 se a caixa deste esta contida na caixa de outro
    contem           1 se a caixa deste contem a caixa de outro
    toca_outro       1 se e 4-adjacente a outro objeto
    n_vizinhos       quantos objetos ele toca
    alinhado         1 se compartilha faixa de linha ou coluna com outro
    cor_maioria      1 se a cor dele e a mais comum entre os objetos

Oito relacionais somadas as dez intrinsecas: o espaco vai de 8x10=80 para
8x18=144 hipoteses.

O RISCO, MEDIDO
-----------------
Ampliar o espaco custou caro cinco vezes nesta sessao: 21x de sobreajuste com
8 lentes, 53 falsos positivos gerados por modificadores e gaveta,
recombinacao 7 -> 2. Mas o portao interceptou 53 de 53 sem perder um
verdadeiro. Ele e a razao de eu poder ampliar sem mentir no relatorio.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  R0  PRE-CONDICAO. As 4 de `ockham.py` tem que sobreviver. Acrescentar
      propriedade so pode ADICIONAR.

  R1  >= 8   -> relacional era o degrau, como foi na linguagem de objetivos
  R2  5 a 7  -> ajuda, no patamar das outras mudancas de representacao
  R3  4      -> nem relacional; o ARC exige comparacao mais rica que estas oito

MINHA CALIBRAGEM, e ela ja e um dado: vinte e duas previsoes na sessao, SETE
certas, as quinze erradas TODAS otimistas. Eu disse "este e o degrau" quinze
vezes e errei quinze.

PREVISAO: R2 na faixa baixa, 5 a 6 (ou seja, +1 a +2). Nao aposto alto apesar
do precedente do quantificador, porque o padrao do meu erro e exatamente
apostar alto quando tenho um argumento estrutural bonito — foi o que fiz em
`residuo`, `ordem1`, `cenadeduz` e `emissao2`, e as quatro deram abaixo do
piso que previ.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys.segmenta import (LENTES, PROPS, Ce, Obj, deltas, fundo_de,
                               props as props_base, render)

RELACIONAIS = ("n_mesma_forma", "forma_unica", "dentro_de", "contem",
               "toca_outro", "n_vizinhos", "alinhado", "cor_maioria")
TODAS = tuple(PROPS) + RELACIONAIS
ORDEM_PROP = {p: i for i, p in enumerate(TODAS)}
ORDEM_LENTE = {n: i for i, (n, _f) in enumerate(LENTES)}


def _forma(o: Obj) -> frozenset:
    y0, x0, _a, _b = o.caixa()
    return frozenset((y - y0, x - x0) for y, x, _c in o.pix)


def props_rel(c: Ce) -> Dict[int, Dict[str, Any]]:
    """
    As dez intrinsecas mais as oito relacionais. As relacionais so existem em
    relacao aos OUTROS objetos da cena — e essa e a diferenca inteira.
    """
    base = props_base(c)
    if not c.objs:
        return base
    formas = {o.id: _forma(o) for o in c.objs}
    cf = Counter(formas.values())
    caixas = {o.id: o.caixa() for o in c.objs}
    cel = {o.id: o.cel for o in c.objs}
    cores = Counter(o.cor for o in c.objs)
    cor_top = cores.most_common(1)[0][0]

    def contido(a, b):
        ay0, ax0, ay1, ax1 = a
        by0, bx0, by1, bx1 = b
        return by0 <= ay0 and bx0 <= ax0 and ay1 <= by1 and ax1 <= bx1

    for o in c.objs:
        ca = caixas[o.id]
        dentro = any(k != o.id and contido(ca, caixas[k]) for k in caixas)
        contem = any(k != o.id and contido(caixas[k], ca) for k in caixas)
        viz = 0
        for k, cs in cel.items():
            if k == o.id:
                continue
            if any((y + dy, x + dx) in cs
                   for y, x in cel[o.id]
                   for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                viz += 1
        alin = any(k != o.id and (
            not (ca[2] < caixas[k][0] or caixas[k][2] < ca[0]) or
            not (ca[3] < caixas[k][1] or caixas[k][3] < ca[1]))
            for k in caixas)
        base[o.id].update({
            "n_mesma_forma": cf[formas[o.id]],
            "forma_unica": int(cf[formas[o.id]] == 1),
            "dentro_de": int(dentro),
            "contem": int(contem),
            "toca_outro": int(viz > 0),
            "n_vizinhos": viz,
            "alinhado": int(alin),
            "cor_maioria": int(o.cor == cor_top),
        })
    return base


def deduzir(treino, lente):
    obs: Dict[str, Dict[Any, Set]] = {p: {} for p in TODAS}
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
        pi = props_rel(ci)
        for oid, lst in d.items():
            for p in TODAS:
                obs[p].setdefault(pi[oid][p], set()).add(lst)
    if len(telas) != 1:
        return None
    tela = next(iter(telas))
    for p in TODAS:
        if any(len(v) > 1 for v in obs[p].values()):
            continue
        return (p, {k: next(iter(v)) for k, v in obs[p].items()}, tela)
    return None


def aplicar(regra, g, lente):
    p, mapa, tela = regra
    bg = fundo_de(g)
    c = lente(g, bg)
    if not c.objs:
        return None
    pr = props_rel(c)
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
            cands.append((ORDEM_LENTE[ln], ORDEM_PROP.get(r[0], 99), ln, r,
                          lente))
        except Exception:
            continue
    if not cands:
        return nome, ajustou, False, None
    cands.sort(key=lambda z: (z[0], z[1]))
    _a, _b, ln, r, lente = cands[0]
    try:
        ok = all(aplicar(r, x, lente) == y for x, y in te)
    except Exception:
        ok = False
    return nome, ajustou, True, (f"{ln}|{r[0]}" if ok else None)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 78)
    print("RELACIONAL — propriedades que comparam objetos entre si")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [n for n, aj, _v, _o in res if aj]
    sobrev = [n for n, _aj, v, _o in res if v]
    ok = {n: o for n, _aj, _v, o in res if o}

    print(f"\n  {len(PROPS)} intrinsecas + {len(RELACIONAIS)} relacionais = "
          f"{len(TODAS)} propriedades x {len(LENTES)} lentes")
    print(f"\n  ajustou o TREINO    : {len(ajust):3d}/400  (`ockham`: 127)")
    print(f"  sobreviveu ao IMUNE : {len(sobrev):3d}/400")
    print(f"  passou no TESTE     : {len(ok):3d}/400  (`ockham`: 4)   "
          f"({time.time()-t0:.0f}s)")
    print(f"  precisao            : {len(ok)/max(len(ajust),1):.1%}")
    antes = {"37d3e8b2", "72a961c9", "b1fc8b8e", "f45f5ca7"}
    print(f"\nR0 · das 4 de `ockham`, mantidas: {sorted(antes & set(ok))}")
    print(f"     novas: {sorted(set(ok) - antes)}")
    usadas = Counter(o.split("|")[1] for o in ok.values())
    print(f"\n  propriedades usadas: {dict(usadas.most_common())}")
    rel = sum(v for k, v in usadas.items() if k in RELACIONAIS)
    print(f"  resolvidas por propriedade RELACIONAL: {rel}")

    print("\n" + "=" * 78)
    if not antes <= set(ok):
        print(f"R0 FALHOU: perdi {sorted(antes - set(ok))}.")
    if len(ok) >= 8:
        print("R1 · relacional era o degrau, como foi na linguagem de")
        print("     objetivos (49,6% -> 99,3%).")
    elif len(ok) >= 5:
        print("R2 · ajuda, no patamar das outras mudancas de representacao.")
    else:
        print("R3 · nem relacional. O ARC exige comparacao mais rica que")
        print("     estas oito.")


if __name__ == "__main__":
    main()

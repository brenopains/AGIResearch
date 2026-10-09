"""
novaphys.ancora — o mesmo simbolo vale nos dois mundos?
========================================================

POR QUE SO AGORA
------------------
A sessao inteira o usuario defendeu transferencia, ancoragem, curriculo e
vocabulario acumulado. Tudo empatou com placebo, e havia uma razao estrutural:

    ancoragem e IMENSURAVEL num dominio so.

Dentro do ARC todo simbolo ja esta disponivel em toda tarefa, entao nao ha o
que transportar — foi por isso que a biblioteca deu 30/30/30 contra placebo
sorteado e o professor perfeito do `dica.py` comprou so +3.

Desde o `dominio2.py` existem DOIS mundos: o ARC e o simulador de particulas do
`deducao.py`, que e anterior a todo o vocabulario construido hoje. E o segundo
ja deu sinal: 6/24 fisico contra 0/24 aleatorio, com `T:desl` e `P:vizinho` nas
seis solucoes.

AS DUAS PARTES
----------------
PARTE 1, SOBREPOSICAO. Quais familias de teste e quais pintores aparecem nas
solucoes vencedoras dos DOIS mundos? Um simbolo que aparece so num deles e um
conceito de dominio. Um que aparece nos dois e um CONCEITO. Isso e descritivo e
barato, e e a evidencia direta de que ha simbolo compartilhado.

PARTE 2, TRANSFERENCIA. A parte que decide. Extrair a biblioteca de abstracoes
das solucoes do ARC e medir no MUNDO DE PARTICULAS, contra dois controles:

    B0  sem biblioteca
    B1  biblioteca vinda do ARC
    B2  placebo pareado, mesma forma, conteudo sorteado

E o mesmo desenho do `holdout.py` que deu 30/30/30 — mas atravessando a
fronteira de dominio, que e a unica condicao em que a hipotese pode viver.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. B0 tem que dar 6/24 no braco fisico, que e a medida do
      `dominio2.py`. Se cair, e erro na troca de biblioteca.

  A1  B1 > B2  -> ANCORAGEM MEDIDA. Abstracoes formadas resolvendo quebra-
      cabecas ajudam a explicar fisica de particula, e o CONTEUDO delas importa.
      E a primeira transferencia entre dominios do projeto.
  A2  B1 ~ B2  -> semear ajuda mas o conteudo nao importa; nao e ancoragem.
  A3  B1 <= B0 -> o vocabulario do ARC atrapalha no outro mundo.

SEMENTE PRE-REGISTRADA: 20260909

MINHA CALIBRAGEM: cinquenta e uma previsoes, DEZOITO certas.

PREVISAO: A2, e desta vez sem otimismo. A biblioteca deu +0 em TODAS as
condicoes dentro do ARC, inclusive com vantagem maxima; nao ha razao para ela
ganhar potencia atravessando dominio. E a Parte 1 deve mostrar sobreposicao
pequena mas nao vazia — `desl` e `vizinho` aparecem nos dois, e sao os dois mais
genericos do vocabulario. Aposto que a SOBREPOSICAO existe (e ja e resultado) e
que a TRANSFERENCIA nao. Se A1 der, eu estava errado sobre a biblioteca a
sessao inteira, e da forma que interessa.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from collections import Counter
from itertools import combinations

from novaphys.arc import carregar, dims
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys import deducao as DD
from novaphys.abstrai import antiunifica, casa

SEMENTE = 20260909
LIB_ARC = "biblioteca_arc_para_fisica.json"
LIB_PLA = "biblioteca_placebo_fisica.json"
ARC, EPS = None, None


def _ini():
    global ARC, EPS
    ARC = carregar(PASTA_AVAL)[:150]
    EPS = []
    for s_ in range(6):
        for fis in (True, False):
            for k, obs in enumerate(DD.pares(fis, 8, 4, SEMENTE + s_)):
                EPS.append((f"{'fis' if fis else 'ale'}_k{k+1}_s{s_}",
                            fis, k + 1, obs))


def _simbolos(pilha):
    """As familias de teste e pintores que a pilha usa."""
    t, p = set(), set()
    for x in pilha:
        if D._marca(x):
            continue
        if x[0].nome == "programa":
            sel, pin = x[1]
            p.add(pin[0])
            for c in sel:
                t.add(c[0])
        elif x[0].nome == "condicao":
            for c in x[1][0]:
                t.add(c[0])
    return t, p


def _arc(ti):
    n, tr, te = ARC[ti]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4)
        if err != 0 or not pilha:
            return None
        if not all(D.aplicar((fm, pilha), x) == y for x, y in te):
            return None
        t, p = _simbolos(pilha)
        progs = [x[1] for x in pilha
                 if not D._marca(x) and x[0].nome == "programa"]
        return sorted(t), sorted(p), [[[list(c) for c in s], list(k)]
                                      for s, k in progs]
    except Exception:
        return None


def _fis(args):
    ti, lib = args
    D.usar_biblioteca(lib)
    nome, fisico, k, obs = EPS[ti]
    try:
        tr, te = obs[:-2], obs[-2:]
        if len(tr) < 3:
            return fisico, k, 0, (), ()
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4)
        if err != 0 or not pilha:
            return fisico, k, 0, (), ()
        ok = all(D.aplicar((fm, pilha), x) == y for x, y in te)
        t, p = _simbolos(pilha) if ok else (set(), set())
        return fisico, k, 1 if ok else 0, tuple(sorted(t)), tuple(sorted(p))
    except Exception:
        return fisico, k, 0, (), ()


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=2)
    a = ap.parse_args()
    _ini()
    rnd = random.Random(SEMENTE)

    print("=" * 76)
    print("ANCORA — o mesmo simbolo vale nos dois mundos?")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        ra = pool.map(_arc, range(len(ARC)), chunksize=2)
        rf = pool.map(_fis, [(i, None) for i in range(len(EPS))], chunksize=1)

    ok_arc = [x for x in ra if x]
    ta = Counter(t for t, _p, _g in ok_arc for t in t)
    pa = Counter(p for _t, ps, _g in ok_arc for p in ps)
    tf = Counter(t for f, _k, o, ts, _ps in rf if f and o for t in ts)
    pf = Counter(p for f, _k, o, _ts, ps in rf if f and o for p in ps)
    base = sum(o for f, _k, o, _t, _p in rf if f)

    print(f"\n  ({time.time()-t0:.0f}s)")
    print(f"  ARC resolvidas na amostra: {len(ok_arc)}/{len(ARC)}")
    print(f"  A0 . fisica sem biblioteca: {base}/24")
    print("")
    print("  PARTE 1 . SOBREPOSICAO DE SIMBOLOS")
    print("    familia          ARC  fisica")
    print("    --------------   ---  ------")
    for k in sorted(set(ta) | set(tf), key=lambda x: -(ta.get(x, 0))):
        m = "  <<< NOS DOIS" if ta.get(k) and tf.get(k) else ""
        print(f"    teste:{k:<10s} {ta.get(k,0):3d}  {tf.get(k,0):4d}{m}")
    for k in sorted(set(pa) | set(pf), key=lambda x: -(pa.get(x, 0))):
        m = "  <<< NOS DOIS" if pa.get(k) and pf.get(k) else ""
        print(f"    pintor:{k:<9s} {pa.get(k,0):3d}  {pf.get(k,0):4d}{m}")
    comuns = (set(ta) & set(tf)) | (set(pa) & set(pf))
    print(f"\n    simbolos nos DOIS mundos: {len(comuns)}  {sorted(comuns)}")

    # biblioteca do ARC
    progs = [(tuple(tuple(c) for c in s), tuple(k))
             for _t, _p, g in ok_arc for s, k in g]
    lib = []
    if len(progs) >= 2:
        cont = Counter()
        for p1, p2 in combinations(progs, 2):
            ab = antiunifica(p1, p2)
            if ab:
                cont[ab] += 1
        for ab, _n in cont.most_common():
            u = sum(1 for s, k in progs if casa(ab, (s, k)))
            if u >= 2 and ab[0]:
                lib.append({"comum": [list(c) for c in sorted(ab[0], key=repr)],
                            "buracos": ab[1], "pintor": list(ab[2]), "usos": u})
        lib = lib[:8]
    io.open(LIB_ARC, "w", encoding="utf-8").write(json.dumps(lib, indent=1))
    voc = sorted({c for _t, _p, g in ok_arc for s, _k in g for c in
                  [tuple(x) for x in s]}, key=repr) or [("fundo",)]
    pla = [{"comum": [list(x) for x in rnd.sample(voc, min(len(x2["comum"]),
                                                           len(voc)))],
            "buracos": x2["buracos"], "pintor": x2["pintor"],
            "usos": x2["usos"]} for x2 in lib]
    io.open(LIB_PLA, "w", encoding="utf-8").write(json.dumps(pla, indent=1))
    print(f"\n  biblioteca do ARC: {len(lib)} abstracoes; placebo pareado idem")

    print("\n  PARTE 2 . TRANSFERENCIA ENTRE DOMINIOS")
    with Pool(a.nucleos, initializer=_ini) as pool:
        r1 = pool.map(_fis, [(i, LIB_ARC) for i in range(len(EPS))], chunksize=1)
        r2 = pool.map(_fis, [(i, LIB_PLA) for i in range(len(EPS))], chunksize=1)
    B1 = sum(o for f, _k, o, _t, _p in r1 if f)
    B2 = sum(o for f, _k, o, _t, _p in r2 if f)
    print(f"    B0 sem biblioteca   : {base}/24")
    print(f"    B1 biblioteca do ARC: {B1}/24")
    print(f"    B2 placebo pareado  : {B2}/24")
    print(f"\n    A1 . B1 - B2 (o conteudo importa?): {B1-B2:+d}")

    print("")
    print("=" * 76)
    if base != 6:
        print(f"A0 . base {base}/24, esperado 6/24 do `dominio2.py`.")
    if B1 > B2 and B1 > base:
        print("A1 . ANCORAGEM MEDIDA. Abstracoes formadas resolvendo quebra-")
        print("     cabecas ajudam a explicar fisica, e o conteudo importa.")
    elif B1 <= base:
        print("A3 . o vocabulario do ARC nao ajuda no outro mundo.")
    else:
        print("A2 . semear ajuda, mas o conteudo nao importa; nao e ancoragem.")


if __name__ == "__main__":
    main()

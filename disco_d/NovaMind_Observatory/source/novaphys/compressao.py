"""
novaphys.compressao — os dez mundos sob o portao de compressao
===============================================================

O QUE MUDOU
-------------
`ruido.py` mediu, nas regras publicadas de Wolfram:

    ruido     razao     comprime   residuo 0
      0%      0,118      4/4        4/4
      3%      0,406      4/4        0/4
     10%      0,644      4/4        0/4
     20%      0,889      4/4        0/4

Sob `residuo == 0` o sistema morre a 3% de ruido. Sob compressao sobrevive a
20%. A unica coisa entre esta arquitetura e dado real era qual numero serve de
limiar.

O RISCO DO CRITERIO NOVO, E COMO ELE E CONTROLADO
---------------------------------------------------
Um modelo rico SEMPRE comprime um pouco. Se o portao for so "razao < 1", ele e
pior que o antigo: aceita qualquer coisa.

O controle e o braco NULO, que ja existe em todos os dez mundos: mesmas
entradas, saidas sorteadas com o mesmo histograma de cor. Saida sem estrutura
nao tem o que comprimir. Entao:

    razao do nulo ~ 1,0  ->  o criterio e honesto, e a diferenca real-nulo
                             mede ESTRUTURA CAPTURADA
    razao do nulo << 1   ->  o criterio esta quebrado e nao serve

E a margem `real - nulo` passa a ser a medida contínua que o portao binario
nunca deu: quanto de estrutura o vocabulario captura, mesmo onde nao fecha.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  Z0  PRE-CONDICAO. Sob `exato`, os mesmos 8 mundos de antes. Se mudar, a
      alteracao quebrou algo e o resto nao vale.

  Z1  razao do NULO >= 0,90 em todos os mundos -> o criterio e honesto
      abaixo disso -> comprimir ruido, criterio pior que o antigo

  Z2  `critters` e `greenberg`, que davam ZERO sob exato, comprimem?
      sim -> ha estrutura capturada que o portao binario escondia
      nao -> os dois estao fora por expressividade, nao por criterio

SEMENTE PRE-REGISTRADA: 20260909

MINHA CALIBRAGEM: cinquenta e cinco previsoes, VINTE certas.

PREVISAO: Z1 passa com folga (saida sorteada nao comprime, e isso e quase
tautologico). E Z2 PARCIAL: aposto que `greenberg` comprime — ele e primo do
`ciclico` e do `brian`, que agora fecham, entao deve haver estrutura parcial
capturada. E aposto que `critters` NAO comprime, porque particao em blocos nao
e uma afirmacao sobre celula: nenhum teste do vocabulario fala sobre "o bloco
2x2 a que eu pertenco", e nao ha razao para captura parcial de algo que nao e
dizivel nem em parte.
"""

from __future__ import annotations

import argparse
import time

from novaphys.arc import dims
from novaphys import descricao as D
from novaphys import nucleo as N


def _ini():
    N._ini()


def _uma(args):
    ti, crit = args
    nome, mundo, real, pr = N.TAR[ti]
    tr, te = pr[:-2], pr[-2:]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4, criterio=crit)
        if not pilha:
            return mundo, real, 1.0, 1.0, 0
        rz = D.razao(pilha, [(D.forma.base(fm, a), b) for a, b in tr])
        # no teste: a pilha comprime a saida que ela nunca viu?
        reb = []
        for x, y in te:
            g = D.forma.base(fm, x)
            if g is None or dims(g) != dims(y):
                return mundo, real, rz, 1.0, 0
            reb.append((g, y))
        rt = D.razao(pilha, reb)
        fech = 1 if (err == 0 and all(D.aplicar((fm, pilha), x) == y
                                      for x, y in te)) else 0
        return mundo, real, rz, rt, fech
    except Exception:
        return mundo, real, 1.0, 1.0, 0


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=4)
    a = ap.parse_args()
    _ini()
    mundos = sorted({m for _n, m, _r, _p in N.TAR})

    print("=" * 76)
    print("COMPRESSAO — os dez mundos sob o portao novo")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        ex = pool.map(_uma, [(i, "exato") for i in range(len(N.TAR))],
                      chunksize=1)
        co = pool.map(_uma, [(i, "compressao") for i in range(len(N.TAR))],
                      chunksize=1)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  mundo       fecha   razao TREINO      razao TESTE      margem")
    print("              exato   real    nulo     real    nulo      teste")
    print("  ----------  -----   -----   -----    -----   -----     ------")
    piores = []
    for m in mundos:
        fe = sum(f for mm, r, _z, _t, f in ex if mm == m and r)
        rr = [x for x in co if x[0] == m and x[1]]
        rn = [x for x in co if x[0] == m and not x[1]]
        if not rr or not rn:
            continue
        zr = sum(x[2] for x in rr) / len(rr)
        zn = sum(x[2] for x in rn) / len(rn)
        tr_ = sum(x[3] for x in rr) / len(rr)
        tn = sum(x[3] for x in rn) / len(rn)
        print(f"  {m:10s}  {fe:2d}      {zr:5.3f}   {zn:5.3f}    "
              f"{tr_:5.3f}   {tn:5.3f}     {tn-tr_:+6.3f}")
        piores.append((m, zn, tn - tr_))

    nulo_min = min(z for _m, z, _g in piores)
    print(f"\n  Z1 . menor razao do NULO no treino: {nulo_min:.3f}")
    for m in ("critters", "greenberg"):
        w = [x for x in co if x[0] == m and x[1]]
        if w:
            z = sum(x[2] for x in w) / len(w)
            print(f"  Z2 . {m:10s} razao {z:.3f}  "
                  f"{'COMPRIME' if z < 0.9 else 'nao comprime'}")

    ne = sum(1 for m, _z, _g in piores)
    fe_tot = sum(f for _m, r, _z, _t, f in ex if r)
    print(f"\n  Z0 . tarefas que fecham sob `exato`: {fe_tot}")

    print("")
    print("=" * 76)
    if nulo_min < 0.90:
        print("Z1 FALHOU: o nulo comprime; o criterio novo aceita ruido e e")
        print("     pior que o antigo.")
    else:
        print("Z1 . o criterio e honesto: saida sem estrutura nao comprime,")
        print("     e a margem real-nulo mede estrutura capturada mesmo onde")
        print("     a pilha nao fecha.")


if __name__ == "__main__":
    main()

"""
novaphys.zeta_explora — M4: a taxa de ALARME FALSO da maquina de descoberta
=============================================================================

POR QUE ISTO E O EXPERIMENTO QUE MAIS IMPORTA
-----------------------------------------------
Hoje, em dado real do cerebro, uma assimetria passou por TRES controles
independentes (nulo permutado, controle espacial, medida compartilhada) e ainda
estava errada NA DIRECAO — so o portao de extrapolacao contra rotulo medido
pegou. E ao longo da sessao QUATRO portoes meus deram falso positivo, todos
enviesados para aceitar.

Logo, antes de apontar o achador de leis para qualquer dado onde nao se saiba a
resposta, e preciso medir a taxa de alarme falso dele contra um gabarito
PERFEITO. Os zeros de zeta sao esse gabarito: o dado mais estudado da
matematica, com Odlyzko tendo computado zeros a altura 10^22 e a estatistica de
espacamentos verificada ha 50 anos. A probabilidade de haver lei nova aqui e
essencialmente ZERO.

Entao: se este modulo ANUNCIAR descoberta, a descoberta e um falso positivo, e o
numero de anuncios e a taxa de alarme falso da maquina.

O QUE E RODADO
----------------
O MESMO `busca()` de `neuro_leis.py` — aquele que no dado do Allen aceitou 0 de
1.892 pares no nulo e reencontrou 4 das 6 relacoes da literatura. Ajuste em
metade, aceitacao so se pagar bits na outra metade, com penalidade de estrutura.

Grandezas derivadas dos 8000 primeiros zeros t_n:

    t          a altura do zero
    n          o indice
    d          espacamento normalizado (t_{n+1}-t_n) * log(t/2pi)/2pi
    d_ant      o espacamento anterior (correlacao de curto alcance)
    logt       log(t)
    dens       densidade local de zeros, log(t/2pi)/2pi
    N_menos_n  N(t) - n, onde N(t) = t/2pi * log(t/2pi e) + 7/8 — e o famoso
               termo flutuante S(t), de media zero

LEIS QUE ELE DEVE REENCONTRAR (controle positivo do lado exploratorio)
------------------------------------------------------------------------
  n x t        Riemann-von Mangoldt: N(T) ~ (T/2pi) log(T/2pi e)
  dens x logt  identidade por construcao
  t x logt     identidade por construcao
Essas TEM de aparecer. Se nao aparecerem, o achador esta cego e o resto nao vale.

PRE-REGISTRO
--------------
  M4a  REENCONTRA O CONHECIDO: as tres relacoes acima aparecem entre as aceitas
  M4b  NULO: com as colunas embaralhadas independentemente, aceita <= 2 leis
  M4c  FALSO POSITIVO: descontadas as relacoes conhecidas e as que sao
       identidade algebrica entre as proprias colunas, o numero de leis
       "novas" anunciadas e ZERO
PREVISAO: M4a passa. M4b passa (foi 0/1892 no Allen). M4c **FALHA** — eu aposto
que ele vai anunciar de 2 a 6 relacoes que nao estao na minha lista de
conhecidas, e que serao todas consequencias algebricas das definicoes (d e
funcao de t e da diferenca; N_menos_n e funcao de t e n; etc). Se for isso, a
taxa de alarme falso NAO e zero, e a licao e que o achador precisa de um passo
de "isto e identidade algebrica das minhas proprias definicoes?" antes de
anunciar — e esse passo nao existe hoje, nem no dado do cerebro.
"""

from __future__ import annotations

import argparse
from math import log, pi

import numpy as np
import pandas as pd

from novaphys.neuro_leis import busca

ARQ = "D:/zeta/zeros.npy"
SEMENTE = 20261028

# as que a literatura ja estabeleceu, ou que sao identidade por construcao
CONHECIDAS = {
    frozenset(("n", "t")),          # Riemann-von Mangoldt
    frozenset(("dens", "logt")),    # definicao de densidade
    frozenset(("t", "logt")),       # identidade
    frozenset(("n", "logt")),       # consequencia de N(T) ~ T log T
    frozenset(("n", "dens")),       # idem
    frozenset(("t", "dens")),       # idem
}


def grandezas(t):
    d = np.diff(t) * (np.log(t[:-1] / (2 * pi)) / (2 * pi))
    t0 = t[1:-1]
    df = pd.DataFrame({
        "t": t0,
        "n": np.arange(2, len(t)),
        "d": d[1:],
        "d_ant": d[:-1],
        "logt": np.log(t0),
        "dens": np.log(t0 / (2 * pi)) / (2 * pi),
    })
    N = t0 / (2 * pi) * np.log(t0 / (2 * pi * np.e)) + 7 / 8
    df["N_menos_n"] = N - df["n"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=ARQ)
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    t = np.load(a.arquivo)
    df = grandezas(t)
    print("=" * 78)
    print(f"ZETA EXPLORA — M4, taxa de alarme falso. {len(df):,} zeros, "
          f"{len(df.columns)} grandezas")
    print("=" * 78)

    ach = busca(df, np.random.default_rng(SEMENTE), "REAL", min_n=2000)
    emb = df.copy()
    for c in emb.columns:
        emb[c] = emb[c].to_numpy()[rng.permutation(len(emb))]
    achn = busca(emb, np.random.default_rng(SEMENTE), "NULO embaralhado", min_n=2000)

    print(f"\n  {'ganho(bits)':>12s}  {'y':12s} {'x':12s} {'familia':10s} "
          f"{'':6s}")
    novas = []
    for g, cy, cx, fam, _c in ach[:20]:
        par = frozenset((cy, cx))
        etq = "CONHECIDA" if par in CONHECIDAS else "*** NOVA ***"
        if par not in CONHECIDAS:
            novas.append((g, cy, cx, fam))
        print(f"  {g:12.0f}  {cy:12s} {cx:12s} {fam:10s} {etq}")

    print("")
    print("=" * 78)
    achadas = {frozenset((cy, cx)) for _g, cy, cx, _f, _c in ach}
    falta = [p for p in (frozenset(("n", "t")), frozenset(("dens", "logt")),
                         frozenset(("t", "logt"))) if p not in achadas]
    print(f"M4a {'.' if not falta else 'FALHOU .'} reencontra o conhecido: "
          f"faltaram {[sorted(p) for p in falta]}")
    print(f"M4b {'.' if len(achn) <= 2 else 'FALHOU .'} nulo aceita "
          f"{len(achn)} leis (<= 2)")
    pares_novos = {frozenset((cy, cx)) for _g, cy, cx, _f in novas}
    print(f"M4c {'.' if not pares_novos else 'FALHOU .'} falso positivo: "
          f"{len(pares_novos)} relacoes 'novas' anunciadas (esperado 0)")
    if pares_novos:
        print(f"\n  as anunciadas como novas: "
              f"{[sorted(p) for p in sorted(pares_novos, key=lambda s: sorted(s))]}")
        print("  TODAS precisam ser examinadas: sao identidade algebrica das")
        print("  minhas proprias definicoes, ou sao alarme falso de verdade?")


if __name__ == "__main__":
    main()

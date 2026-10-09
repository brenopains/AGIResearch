"""
novaphys.neuro_leis — leis sobre 74 mil neuronios REAIS, com o alfabeto dos
neurocientistas
============================================================================

DE ONDE VEM
-------------
O gargalo de fundo do projeto: o unico canal de estrutura nova sou eu escrevendo
primitivas. Aqui o alfabeto NAO e meu — sao as ~58 grandezas que os
neurocientistas do Allen Institute mediram em 74.072 neuronios de camundongo
acordado (taxa de disparo sob 5 estimulos, fator de Fano, seletividade a
orientacao e direcao, campo receptivo, esparsidade, modulacao por corrida).
Dado real, publico, e nao gerado por mim.

O MECANISMO (o mesmo das orbitas, em forma estatistica)
--------------------------------------------------------
Para cada par (x, y) de grandezas, duas familias de lei:
    linear     y = a x + b
    potencia   log y = a log x + b        (so para grandezas positivas)
Ajuste em metade dos neuronios, ACEITACAO so se pagar bits na outra metade:
    ganho = (n/2) log2(variancia base / variancia do residuo) - bits da estrutura
Mais uma tripla declarada (controle positivo): area ~ largura x altura.
NULO: cada coluna embaralhada por conta propria — destroi a relacao entre
grandezas do MESMO neuronio e preserva as distribuicoes marginais.

PRE-REGISTRO, ANTES DE OLHAR O RESULTADO
------------------------------------------
Gabarito (relacoes que a literatura ja estabeleceu; escritas antes de rodar):
  K1  a taxa de disparo e propriedade do neuronio: taxa sob um estimulo prediz a
      taxa sob outro, com expoente perto de 1
  K2  o fator de Fano CAI com a taxa de disparo (expoente negativo)
  K3  a esparsidade de vida cai com a taxa de disparo
  K4  seletividade a orientacao em grades moveis prediz a de grades estaticas
  K5  a distribuicao de taxas e log-normal (assimetria em log << assimetria linear)
  K6  area do campo receptivo ~ largura x altura (identidade geometrica — controle
      positivo: se isto nao sair, o metodo esta quebrado)

  N1  o nulo aceita <= 2 leis (de milhares de pares testados)
  N2  o metodo recupera >= 4 das 6 do gabarito
PREVISAO: N1 passa. N2 passa (K6 e K1 saem facil; K2 e K4 provaveis; K3 incerta).
O que espero que ISTO mostre e o teto de sempre: ele acha as relacoes que EXISTEM
entre as grandezas dadas, e nao inventa a grandeza que falta.
"""

from __future__ import annotations

import argparse
import itertools
from math import log2

import numpy as np
import pandas as pd

ARQ = "D:/dados_cerebro/brain_observatory_1.1_analysis_metrics.csv"
MIN_N = 8000          # coluna so entra se tiver este tanto de medidas
SEMENTE = 20261017


def carrega(caminho):
    df = pd.read_csv(caminho, low_memory=False)
    num = df.select_dtypes(include=[np.number])
    bons = [c for c in num.columns if num[c].notna().sum() >= MIN_N
            and num[c].nunique() > 20 and c != "ecephys_unit_id"]
    return num[bons]


def _bits(res_ho, base_ho, n, estrutura):
    if res_ho <= 0 or base_ho <= 0:
        return -1e9
    return 0.5 * n * log2(base_ho / res_ho) - estrutura


def _ajusta(x, y, tr, te, estrutura):
    """Ajusta y = a x + b em `tr` e devolve os bits ganhos em `te`."""
    A = np.stack([x[tr], np.ones(int(tr.sum()))], 1)
    coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    pred = coef[0] * x[te] + coef[1]
    res = float(np.mean((y[te] - pred) ** 2))
    base = float(np.var(y[te]))
    return _bits(res, base, int(te.sum()), estrutura), coef


def busca(df, rng, rotulo, min_n=None):
    """`min_n` sobrepoe MIN_N. DEFEITO QUE ISTO CONSERTA: MIN_N = 8000 foi
    calibrado para os 74 mil neuronios do Allen, e ao rodar este mesmo achador
    sobre 7.998 zeros de zeta o `continue` descartou os 42 pares sem testar
    nenhum — 0 leis aceitas, inclusive Riemann-von Mangoldt, que vale +17.520
    bits. Constante de um dominio herdada por outro vira cegueira silenciosa, e
    esta enviesava para REJEITAR (as outras quatro do dia enviesavam para
    aceitar)."""
    lim = MIN_N if min_n is None else min_n
    cols = list(df.columns)
    m = len(cols)
    achados = []
    V = {c: df[c].to_numpy(dtype=float) for c in cols}
    for cy, cx in itertools.permutations(cols, 2):
        y, x = V[cy], V[cx]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < lim:
            continue
        idx = np.where(ok)[0]
        marca = rng.random(len(idx)) < 0.5
        tr = np.zeros(len(x), bool)
        te = np.zeros(len(x), bool)
        tr[idx[marca]] = True
        te[idx[~marca]] = True
        # estrutura: escolher o par (2 log2 m), a familia (log2 2), 2 parametros
        est = 2 * log2(m) + 1 + 2 * 0.5 * log2(int(te.sum()))
        melhor = (*_ajusta(x, y, tr, te, est), "linear")
        pos = ok & (x > 0) & (y > 0)
        if pos.sum() >= lim:
            lx = np.where(pos, np.log(np.where(pos, x, 1.0)), 0.0)
            ly = np.where(pos, np.log(np.where(pos, y, 1.0)), 0.0)
            trp, tep = tr & pos, te & pos
            if trp.sum() > 100 and tep.sum() > 100:
                g_pot, c_pot = _ajusta(lx, ly, trp, tep, est)
                if g_pot > melhor[0]:
                    melhor = (g_pot, c_pot, "potencia")
        if melhor[0] > 0:
            achados.append((melhor[0], cy, cx, melhor[2], melhor[1]))
    achados.sort(reverse=True, key=lambda t: t[0])
    print(f"  {rotulo}: {len(achados)} leis aceitas de {m*(m-1)} pares testados")
    return achados


def tripla(df, rng, alvo="area_rf", fontes=("width_rf", "height_rf")):
    """Controle positivo K6: area ~ largura x altura (em log, soma)."""
    if alvo not in df.columns or any(c not in df.columns for c in fontes):
        return None
    y = df[alvo].to_numpy(float)
    a = df[fontes[0]].to_numpy(float)
    b = df[fontes[1]].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b) & (y > 0) & (a > 0) & (b > 0)
    if ok.sum() < 1000:
        return None
    ly, la, lb = np.log(y[ok]), np.log(a[ok]), np.log(b[ok])
    marca = rng.random(int(ok.sum())) < 0.5
    A = np.stack([la[marca], lb[marca], np.ones(int(marca.sum()))], 1)
    coef, *_ = np.linalg.lstsq(A, ly[marca], rcond=None)
    pred = coef[0] * la[~marca] + coef[1] * lb[~marca] + coef[2]
    res = float(np.mean((ly[~marca] - pred) ** 2))
    base = float(np.var(ly[~marca]))
    est = 3 * log2(len(df.columns)) + 3 * 0.5 * log2(int((~marca).sum()))
    return _bits(res, base, int((~marca).sum()), est), coef, int(ok.sum())


def assimetria(df, col="firing_rate_dg"):
    """K5: a distribuicao de taxas e log-normal? assimetria em log vs linear."""
    if col not in df.columns:
        return None
    v = df[col].to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 1000:
        return None

    def sk(z):
        z = (z - z.mean()) / z.std()
        return float(np.mean(z ** 3))
    return sk(v), sk(np.log(v)), len(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=ARQ)
    a = ap.parse_args()
    df = carrega(a.arquivo)
    print("=" * 78)
    print(f"NEURO LEIS — {len(df)} neuronios reais, {len(df.columns)} grandezas medidas")
    print("=" * 78)

    real = busca(df, np.random.default_rng(SEMENTE), "REAL ")
    emb = df.copy()
    r2 = np.random.default_rng(SEMENTE + 1)
    for c in emb.columns:                     # nulo: cada coluna embaralhada
        emb[c] = emb[c].to_numpy()[r2.permutation(len(emb))]
    nulo = busca(emb, np.random.default_rng(SEMENTE), "NULO ")

    print("\n  as 12 leis mais fortes (bits ganhos fora da amostra):")
    for g, cy, cx, fam, coef in real[:12]:
        eq = (f"{cy} = {coef[0]:+.3f}*{cx} {coef[1]:+.3f}" if fam == "linear"
              else f"log {cy} = {coef[0]:+.3f}*log {cx} {coef[1]:+.3f}")
        print(f"    {g:9.0f} bits  [{fam:9s}] {eq}")

    print("\n  gabarito pre-registrado:")

    def achou(cy_pref, cx_pref, sinal=None):
        for g, cy, cx, fam, coef in real:
            if cy.startswith(cy_pref) and cx.startswith(cx_pref):
                if sinal is None or np.sign(coef[0]) == sinal:
                    return g, cy, cx, fam, coef
        return None

    k1 = achou("firing_rate_dg", "firing_rate_sg")
    k2 = achou("fano_dg", "firing_rate_dg", -1)
    k3 = achou("lifetime_sparseness_dg", "firing_rate_dg", -1)
    k4 = achou("g_osi_dg", "g_osi_sg")
    k5 = assimetria(df)
    k6 = tripla(df, np.random.default_rng(SEMENTE + 2))
    ok = 0
    for nome, r in (("K1 taxa prediz taxa", k1), ("K2 Fano cai com a taxa", k2),
                    ("K3 esparsidade cai com a taxa", k3),
                    ("K4 OSI movel x estatica", k4)):
        if r:
            ok += 1
            g, cy, cx, fam, coef = r
            print(f"    OK   {nome:32s} {g:8.0f} bits, expoente {coef[0]:+.3f} [{fam}]")
        else:
            print(f"    --   {nome:32s} nao achada")
    if k5:
        s_lin, s_log, n = k5
        bom = abs(s_log) < abs(s_lin) / 2
        ok += bom
        print(f"    {'OK  ' if bom else '--  '} K5 taxas log-normais          "
              f"assimetria linear {s_lin:.2f} vs log {s_log:.2f} (n={n})")
    if k6:
        g, coef, n = k6
        bom = g > 0 and abs(coef[0] - 1) < 0.25 and abs(coef[1] - 1) < 0.25
        ok += bom
        print(f"    {'OK  ' if bom else '--  '} K6 area ~ largura x altura    "
              f"{g:8.0f} bits, expoentes {coef[0]:+.3f} e {coef[1]:+.3f} (n={n})")
    print("")
    print("=" * 78)
    print(f"N1 {'.' if len(nulo) <= 2 else 'FALHOU .'} nulo aceitou {len(nulo)} leis (<= 2)")
    print(f"N2 {'.' if ok >= 4 else 'FALHOU .'} recuperou {ok}/6 do gabarito (>= 4)")


if __name__ == "__main__":
    main()

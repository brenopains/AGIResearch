"""
novaphys.conectoma — medidas num cerebro inteiro real (FlyWire 783)
=====================================================================

O DADO
--------
`proofread_connections_783.feather`, 16,8 milhoes de linhas: par pre->pos por
neuropilo, com contagem de sinapses e probabilidade media de cada
neurotransmissor. E o cerebro ADULTO INTEIRO de Drosophila, reconstruido sinapse a
sinapse — ~139 mil neuronios. Dado publico, real, que nao fui eu que gerei, e com
alfabeto que nao e meu.

POR QUE ISTO E DIFERENTE DO RESTO DO PROJETO
---------------------------------------------
Ate aqui, quase todo mundo que o sistema mediu foi mundo meu. Aqui as grandezas
sao as que a area usa (grau, peso sinaptico, neurotransmissor, particao por
neuropilo) e os valores esperados vem da literatura, nao de mim. O que o sistema
faz e o de sempre: ajustar em metade, aceitar so se pagar bits na outra metade, e
comparar com um nulo.

PRE-REGISTRO, ANTES DE OLHAR O RESULTADO
------------------------------------------
  R1  CAUDA PESADA NO GRAU: log-normal ganha de exponencial por > 1000 bits fora
      da amostra, nas distribuicoes de grau de entrada e de saida
  R2  COMPOSICAO QUIMICA: a fracao de neuronios GABAergicos (inibitorios) fica
      entre 10% e 25%, e a acetilcolina e o maior grupo
  R3  REGRA DE RENT entre neuropilos: conexoes que cruzam a fronteira E ~ N^p com
      p entre 0,5 e 0,95 e R2 > 0,8
  R4  RECIPROCIDADE: pares bidirecionais em excesso de pelo menos 2x sobre o
      esperado pelo modelo de configuracao (que preserva os graus)
PREVISAO: R1 passa (cauda pesada e consenso). R2 passa (a literatura poe ACh como
maior grupo e GABA perto de 15-20%). R3 e a duvidosa: Rent e definido sobre
particao ESPACIAL, e neuropilo e uma particao anatomica grosseira — pode dar
expoente fora da faixa. R4 passa.
"""

from __future__ import annotations

import argparse
from math import log2

import numpy as np
import pandas as pd
import pyarrow.feather as feather

ARQ = "D:/dados_cerebro/proofread_connections_783.feather"
SEMENTE = 20261019


def carrega(caminho):
    t = feather.read_table(caminho, memory_map=True,
                           columns=["pre_pt_root_id", "post_pt_root_id", "neuropil",
                                    "syn_count", "gaba_avg", "ach_avg", "glut_avg"])
    return t.to_pandas()


def _bits_ajuste(x, rng, familia):
    """Bits ganhos fora da amostra ao codificar x com `familia` em vez da media."""
    x = x[np.isfinite(x) & (x > 0)]
    m = rng.random(len(x)) < 0.5
    tr, te = x[m], x[~m]
    if len(tr) < 100 or len(te) < 100:
        return -1e9
    if familia == "lognormal":
        z_tr, z_te = np.log(tr), np.log(te)
        mu, sd = z_tr.mean(), z_tr.std()
        # -log2 p(x) = log2(x) + log2(sd sqrt(2pi)) + (z-mu)^2 / (2 sd^2 ln2)
        cod = (np.log2(te) + log2(sd * np.sqrt(2 * np.pi))
               + (z_te - mu) ** 2 / (2 * sd ** 2 * np.log(2)))
    else:                                   # exponencial
        lam = 1.0 / tr.mean()
        cod = -np.log2(lam) + te * lam / np.log(2)
    base = np.log2(te.max() + 1)            # codigo uniforme como referencia
    return float(np.sum(base - cod))


def graus(df):
    saida = df.groupby("pre_pt_root_id")["syn_count"].agg(["sum", "count"])
    entrada = df.groupby("post_pt_root_id")["syn_count"].agg(["sum", "count"])
    return saida, entrada


def neurotransmissor(df):
    """NT de cada neuronio: media das probabilidades ponderada por sinapse."""
    w = df["syn_count"].to_numpy(float)
    pre = df["pre_pt_root_id"].to_numpy()
    ids, inv = np.unique(pre, return_inverse=True)
    tot = np.bincount(inv, weights=w, minlength=len(ids))
    col = {}
    for c in ("gaba_avg", "ach_avg", "glut_avg"):
        col[c] = np.bincount(inv, weights=df[c].to_numpy(float) * w,
                             minlength=len(ids)) / np.maximum(tot, 1e-9)
    M = np.stack([col["gaba_avg"], col["ach_avg"], col["glut_avg"]], 1)
    escolha = np.array(["gaba", "ach", "glut"])[M.argmax(1)]
    return pd.Series(escolha).value_counts(normalize=True)


def rent(df):
    """E ~ N^p entre neuropilos. Cada neuronio e atribuido ao neuropilo onde tem
    MAIS sinapses; E = arestas que cruzam a fronteira daquele neuropilo."""
    lado = pd.concat([
        df[["pre_pt_root_id", "neuropil", "syn_count"]].rename(
            columns={"pre_pt_root_id": "id"}),
        df[["post_pt_root_id", "neuropil", "syn_count"]].rename(
            columns={"post_pt_root_id": "id"}),
    ])
    lado = lado[lado.neuropil.notna() & (lado.neuropil != "None")]
    por = lado.groupby(["id", "neuropil"], sort=False)["syn_count"].sum()
    casa = por.reset_index().sort_values("syn_count").drop_duplicates("id", keep="last")
    reg = dict(zip(casa["id"].to_numpy(), casa["neuropil"].to_numpy()))
    pares = df.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False).size()
    a = pares.index.get_level_values(0).to_numpy()
    b = pares.index.get_level_values(1).to_numpy()
    ra = pd.Series(a).map(reg)
    rb = pd.Series(b).map(reg)
    ok = ra.notna() & rb.notna()
    ra, rb = ra[ok], rb[ok]
    cruza = ra.to_numpy() != rb.to_numpy()
    E = (pd.Series(ra.to_numpy()[cruza]).value_counts()
         .add(pd.Series(rb.to_numpy()[cruza]).value_counts(), fill_value=0))
    N = casa.groupby("neuropil")["id"].nunique()
    d = pd.DataFrame({"N": N, "E": E}).dropna()
    d = d[(d.N >= 50) & (d.E > 0)].reset_index().rename(columns={"index": "neuropilo"})
    x, y = np.log(d.N.to_numpy(float)), np.log(d.E.to_numpy(float))
    p, _b = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return p, r2, len(d), d


def reciprocidade(df):
    """Pares bidirecionais, contra o esperado pelo modelo de configuracao."""
    pares = df.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False).size()
    a = pares.index.get_level_values(0).to_numpy()
    b = pares.index.get_level_values(1).to_numpy()
    ids, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    n = len(ids)
    ia, ib = inv[: len(a)], inv[len(a):]
    laco = ia != ib
    ia, ib = ia[laco], ib[laco]
    chave = ia.astype(np.int64) * n + ib
    inversa = ib.astype(np.int64) * n + ia
    recip = int(np.isin(inversa, chave, assume_unique=False).sum()) // 2
    M = len(chave)
    kout = np.bincount(ia, minlength=n).astype(float)
    kin = np.bincount(ib, minlength=n).astype(float)
    # esperado: soma_ij p_ij p_ji com p_ij ~ kout_i kin_j / M
    esperado = float((kout * kin).sum() ** 2) / (2 * M ** 2) if M else 0.0
    return recip, esperado, M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=ARQ)
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    print("=" * 78)
    print("CONECTOMA — cerebro adulto inteiro de Drosophila (FlyWire 783)")
    print("=" * 78)
    df = carrega(a.arquivo)
    print(f"  {len(df):,} linhas   neuronios distintos: "
          f"{len(set(df.pre_pt_root_id) | set(df.post_pt_root_id)):,}")

    saida, entrada = graus(df)
    r1 = {}
    for nome, g in (("grau de saida", saida["count"]), ("grau de entrada", entrada["count"])):
        v = g.to_numpy(float)
        b_ln = _bits_ajuste(v, np.random.default_rng(SEMENTE), "lognormal")
        b_ex = _bits_ajuste(v, np.random.default_rng(SEMENTE), "exponencial")
        r1[nome] = b_ln - b_ex
        print(f"  {nome:16s} n={len(v):7,}  mediana {np.median(v):6.0f}  "
              f"maximo {v.max():7.0f}   log-normal ganha da exponencial por "
              f"{b_ln - b_ex:12,.0f} bits")

    nt = neurotransmissor(df)
    print("\n  neurotransmissor predominante por neuronio:")
    for k, v in nt.items():
        print(f"    {k:6s} {100*v:5.1f}%")
    gaba = float(nt.get("gaba", 0.0))
    maior_ach = nt.idxmax() == "ach"

    p, r2, nnp, tab = rent(df)
    print(f"\n  regra de Rent entre {nnp} neuropilos: E ~ N^{p:.3f}  (R2={r2:.3f})")
    print("   " + tab.sort_values("N").tail(5).to_string(index=False).replace("\n", "\n   "))

    recip, esp, M = reciprocidade(df)
    razao = recip / esp if esp else 0.0
    print(f"\n  reciprocidade: {recip:,} pares bidirecionais em {M:,} conexoes; "
          f"esperado {esp:,.0f} pelo modelo de configuracao -> {razao:.1f}x")

    print("")
    print("=" * 78)
    ok1 = min(r1.values()) > 1000
    print(f"R1 {'.' if ok1 else 'FALHOU .'} cauda pesada: log-normal ganha por "
          f"{min(r1.values()):,.0f} bits no pior caso (> 1000)")
    ok2 = 0.10 <= gaba <= 0.25 and maior_ach
    print(f"R2 {'.' if ok2 else 'FALHOU .'} GABA {100*gaba:.1f}% (10-25%) e ACh o maior grupo: {maior_ach}")
    ok3 = 0.5 <= p <= 0.95 and r2 > 0.8
    print(f"R3 {'.' if ok3 else 'FALHOU .'} Rent p={p:.3f} (0,5-0,95) com R2={r2:.3f} (> 0,8)")
    ok4 = razao >= 2
    print(f"R4 {'.' if ok4 else 'FALHOU .'} reciprocidade {razao:.1f}x o esperado (>= 2x)")


if __name__ == "__main__":
    main()

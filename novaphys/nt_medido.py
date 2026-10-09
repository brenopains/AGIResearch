"""
novaphys.nt_medido — o portao de extrapolacao: a assimetria sobrevive ao rotulo MEDIDO?
=========================================================================================

DE ONDE VEIO A PREVISAO
-------------------------
Com os rotulos de neurotransmissor PREDITOS por aprendizado de maquina sobre
imagens de microscopia (Eckstein et al.), `nt_conectividade.py` mediu, em 68.896
neuronios escondidos do FlyWire 783:

    parceiro PRE (quem manda para mim)   +0,0397 bit/neuronio
    parceiro POS (para quem eu mando)    +0,0061 bit/neuronio      razao 6,5x
    so parceiros em OUTRO neuropilo      +0,0321 x +0,0028         razao 11,6x
    excluindo a sinapse com o escondido  +0,0397 x +0,0060         razao 6,6x

Tres controles passados (nulo permutado negativo para tudo; assimetria CRESCE ao
afastar os parceiros; medida compartilhada nao explica), e o confundidor de grau
empurra na direcao contraria (grau de saida mediano 85 > entrada 72).

Mas os portoes G3 e G4 falharam: o melhor preditor era ANATOMICO (neuropilo) e
nao era uma lei (gastava 26% da economia em estrutura — tabela de 61 entradas).
A conclusao pre-registrada foi que o dado com rotulo PREDITO nao separa biologia
de duto, e que o gap acionavel era **rotulo medido**.

O PORTAO, REGISTRADO ANTES DE OLHAR O DADO MEDIDO
---------------------------------------------------
  E1  A ASSIMETRIA SOBREVIVE: razao pre/pos >= 3x com rotulo MEDIDO
  E2  NULO: com rotulos permutados, nenhuma economia passa de 0,01 bit
  E3  MAGNITUDE: a economia `pre` fica >= 0,02 bit/neuronio
PREVISAO: E1 passa, E2 passa, E3 passa. Se E1 falhar, a assimetria era o duto de
rotulagem, e eu erro em publico uma previsao registrada antes de olhar — o que e
o proposito de registrar.

O DADO MEDIDO
---------------
`Supplemental_file1_neuron_annotations.tsv` (Schlegel et al. 2024) tem a coluna
`known_nt` com o neurotransmissor estabelecido EXPERIMENTALMENTE, e
`known_nt_source` com a fonte: 87.837 neuronios, 155 fontes distintas, entre elas
TAPIN-seq (Davis et al. 2020), EASI-FISH (Nern et al. 2024), RT-PCR (Takemura et
al. 2011), imuno-histoquimica (Ito et al. 2013), FACS RNA-seq (Konstantinides et
al. 2018). Nada de predicao de imagem: aqui o alvo E os parceiros usam rotulo
medido.

NORMALIZACAO, DECLARADA ANTES
-------------------------------
`known_nt` vem em texto livre e composto ("acetylcholine; acetylcholine",
"glutamate, gaba; glutamate", "gaba-negative"). Regra conservadora:
  - minuscula, quebra em ';' e ',';
  - coleta os tokens que sao transmissor rapido: acetylcholine, glutamate, gaba,
    histamine;
  - "gaba-negative" e EVIDENCIA DE AUSENCIA, nao rotulo: descartado;
  - fica so quem tem EXATAMENTE UM transmissor distinto. Ambiguo sai.

DUAS RESSALVAS QUE EU NAO VOU ESCONDER
----------------------------------------
1. `known_nt` e atribuido por TIPO CELULAR, entao neuronios do mesmo tipo
   compartilham rotulo — e neuronios do mesmo tipo se conectam de forma
   estereotipada. Parte do sinal de "parceiro" e portanto "mesmo tipo celular".
   Isso e biologia, nao artefato, mas muda o que a frase pode afirmar.
2. O lobulo optico domina (56.861 de 87.837) e e altamente colunar e
   estereotipado. Por isso o resultado e reportado tambem SO no cerebro central.
"""

from __future__ import annotations

import argparse
import time
from math import log2

import numpy as np
import pandas as pd
import pyarrow.feather as feather

CONEX = "D:/dados_cerebro/proofread_connections_783.feather"
ANOT = "D:/dados_cerebro/nt/anot.tsv"
SEMENTE = 20261027
RAPIDOS = ("acetylcholine", "glutamate", "gaba", "histamine")


def normaliza_nt(s):
    """Texto livre -> um transmissor rapido, ou None."""
    if not isinstance(s, str):
        return None
    t = s.lower()
    achados = set()
    for parte in t.replace(";", ",").split(","):
        p = parte.strip()
        if p.endswith("-negative") or p.endswith("negative"):
            continue                      # evidencia de AUSENCIA
        for r in RAPIDOS:
            if p == r:
                achados.add(r)
    return achados.pop() if len(achados) == 1 else None


def _entropia(conta):
    n = conta.sum()
    if n <= 0:
        return 0.0
    p = conta / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def bits_condicional(chave_tr, rot_tr, chave_te, rot_te, marg, k, piso=0.5):
    tab = {}
    for c, r in zip(chave_tr, rot_tr):
        if c < 0:
            continue
        v = tab.get(c)
        if v is None:
            v = np.zeros(k)
            tab[c] = v
        v[r] += 1
    pm = (marg + piso) / (marg.sum() + piso * k)
    tot = 0.0
    for c, r in zip(chave_te, rot_te):
        v = tab.get(c)
        p = pm if (v is None or v.sum() < 5) else (v + piso) / (v.sum() + piso * k)
        tot += -log2(max(p[r], 1e-12))
    return tot / max(len(rot_te), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-central", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()

    print("=" * 78)
    print("NT MEDIDO — o portao de extrapolacao da assimetria pre/pos"
          f"{'  [SO CEREBRO CENTRAL]' if a.so_central else ''}")
    print("=" * 78)

    d = pd.read_csv(ANOT, sep="\t", low_memory=False,
                    usecols=["root_id", "known_nt", "super_class", "cell_type"])
    d["nt"] = d.known_nt.map(normaliza_nt)
    d = d[d.nt.notna()]
    if a.so_central:
        d = d[d.super_class.isin(["central", "descending", "ascending",
                                  "visual_projection", "visual_centrifugal"])]
    print(f"  {len(d):,} neuronios com rotulo MEDIDO e nao ambiguo")
    print("  " + "  ".join(f"{k} {v:,}" for k, v in
                           d.nt.value_counts().items()))

    classes = sorted(d.nt.unique())
    cod = {c: k for k, c in enumerate(classes)}
    ids = d.root_id.to_numpy()
    rot = d.nt.map(cod).to_numpy().astype(np.int8)
    pos = {int(i): k for k, i in enumerate(ids)}
    marg = np.bincount(rot, minlength=len(classes)).astype(float)
    H = _entropia(marg)
    print(f"  LINHA DE BASE: {H:.4f} bits/neuronio ({len(classes)} classes)")

    df = feather.read_table(CONEX, memory_map=True,
                            columns=["pre_pt_root_id", "post_pt_root_id"]
                            ).to_pandas()
    a_ = df.pre_pt_root_id.to_numpy()
    b_ = df.post_pt_root_id.to_numpy()
    ia = np.array([pos.get(int(x), -1) for x in a_])
    ib = np.array([pos.get(int(x), -1) for x in b_])
    val = (ia >= 0) & (ib >= 0)
    print(f"  {int(val.sum()):,} conexoes entre neuronios com rotulo medido "
          f"({time.time()-t0:.0f}s)")

    ajuste = rng.random(len(ids)) < 0.5
    esconde = ~ajuste
    print(f"  ajuste {int(ajuste.sum()):,}  escondidos {int(esconde.sum()):,}")

    def economia(direcao, rotulos):
        ie, idl = (ia, ib) if direcao == "pos" else (ib, ia)
        m = val & ajuste[np.where(idl >= 0, idl, 0)]
        e, dd = ie[m], idl[m]
        conta = np.zeros((len(ids), len(classes)), dtype=np.int32)
        np.add.at(conta, (e, rotulos[dd]), 1)
        tem = conta.sum(1) > 0
        chave = np.where(tem, conta.argmax(1), -1)
        b = bits_condicional(chave[ajuste], rotulos[ajuste],
                             chave[esconde], rotulos[esconde], marg, len(classes))
        return H - b, int(tem.sum())

    print(f"\n  {'':34s} {'economia':>10s} {'com parceiro':>14s}")
    ep, np_ = economia("pre", rot)
    eo, no_ = economia("pos", rot)
    print(f"  {'pre (quem manda para mim)':34s} {ep:+10.4f} {np_:14,}")
    print(f"  {'pos (para quem eu mando)':34s} {eo:+10.4f} {no_:14,}")
    razao = ep / max(eo, 1e-9)
    print(f"  {'razao pre/pos':34s} {razao:10.1f}x")

    rp = rot[rng.permutation(len(rot))]
    epn, _ = economia("pre", rp)
    eon, _ = economia("pos", rp)
    print(f"\n  NULO permutado:  pre {epn:+.4f}   pos {eon:+.4f}")

    print("")
    print("=" * 78)
    print(f"E1 {'.' if razao >= 3 else 'FALHOU .'} assimetria sobrevive ao "
          f"rotulo MEDIDO: {razao:.1f}x (>= 3x)")
    print(f"E2 {'.' if max(epn, eon) <= 0.01 else 'FALHOU .'} nulo: melhor "
          f"{max(epn, eon):+.4f} (<= 0,01)")
    print(f"E3 {'.' if ep >= 0.02 else 'FALHOU .'} magnitude: pre {ep:+.4f} "
          f"(>= 0,02)")
    print(f"\n  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

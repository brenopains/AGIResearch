"""
novaphys.nt_conectividade — a identidade quimica cabe em quantos bits de fiacao?
==================================================================================

O ALVO, ESCOLHIDO A DEDO
--------------------------
O usuario pediu um problema em aberto de verdade, nao brinquedo, e pediu que eu
escolhesse aquele em que tenho confianca de que a maquina resolve OU revela gap.

Escolhi: **prever a identidade de neurotransmissor de um neuronio a partir da
estrutura de conexao dele**, no FlyWire 783 (cerebro adulto inteiro de
Drosophila, 138.639 neuronios, 16,8 milhoes de conexoes).

POR QUE ESTE
--------------
O subsistema maduro desta arquitetura e o achador de leis sobre grandezas
medidas: em `neuro_leis.py`, sobre 74 mil neuronios do Allen, o nulo aceitou
**0 de 1.892** pares e o metodo reencontrou **4 das 6** relacoes que eu
pre-registrei da literatura. E o unico subsistema com taxa de falso positivo
MEDIDA em dado real. Fisica gerada por mim e brinquedo; cura exige maquinario que
nao existe aqui; materiais exigiria dado novo. Este dado esta em disco.

E a pergunta tem portao de extrapolacao REAL: esconde-se metade dos neuronios e
preve-se o neurotransmissor deles so da estrutura.

A PERGUNTA EM BITS, QUE E O DIFERENCIAL
-----------------------------------------
Uma rede neural ganharia desta maquina em ACURACIA. O diferencial aqui nao e
acurácia, e COMPRESSAO: codificar o NT de N neuronios pela distribuicao marginal
(ACh 67,1%, glut 18,5%, GABA 14,4%) custa H ~ 1,23 bits por neuronio. A pergunta
e se uma regra CURTA baseada em fiacao reduz isso em neuronios ESCONDIDOS. A
afirmacao "a identidade quimica cabe em k bits de estrutura" e uma lei legivel;
"acerto 80%" nao e.

O CONFUNDIDOR, DITO ANTES DE RODAR
------------------------------------
Os rotulos de NT do FlyWire vem de PREDICAO por aprendizado de maquina sobre
imagens de microscopia eletronica, nao de medida direta. Entao parte do sinal
pode ser o viés espacial daquele preditor, e nao biologia. O desenho separa as
duas familias justamente por isso:

  PARCEIRO  maioria dos parceiros pos-sinapticos / pre-sinapticos / vizinhos
            (atravessa neuropilos distantes -> dificil de explicar por viés
             espacial do rotulador)
  ESPACO    moda do neuropilo onde o neuronio tem mais sinapses; decil de grau
            (se SO isto pagar, o achado pode ser o viés do rotulador)

VAZAMENTO, QUE DECIDE A VALIDADE INTEIRA
------------------------------------------
Ao prever um neuronio ESCONDIDO pela maioria dos parceiros, so podem entrar
parceiros da metade de AJUSTE. Usar qualquer parceiro deixaria os rotulos
escondidos entrarem pela porta de tras. O mesmo vale para a moda do neuropilo:
ela e estimada so com neuronios da metade de ajuste.

PRE-REGISTRO, ANTES DE OLHAR
------------------------------
  G0  SANIDADE: a linha de base (entropia marginal) fica entre 1,1 e 1,4
      bits/neuronio
  G1  NULO: com os rotulos de NT PERMUTADOS entre neuronios, nenhuma hipotese
      economiza mais de 0,01 bit/neuronio na metade escondida
  G2  ALGUMA LEI PAGA: pelo menos uma hipotese economiza >= 0,05 bit/neuronio
      escondido, descontada a estrutura
  G3  ESTRUTURA > ESPACO: a melhor regra de PARCEIRO economiza mais que a melhor
      regra de ESPACO — e este e o portao que separa lei de circuito do viés do
      rotulador de imagens
  G4  E CURTA: a regra vencedora gasta em estrutura menos de 1% do que economiza
      (lei, nao tabela de consulta)
PREVISAO: G0 passa. G1 passa — e se nao passar, o portao esta quebrado e nada
mais vale. G2 passa, aposto 0,1 a 0,4 bit. G3 **FALHA** — aposto que a regra
espacial (neuropilo) ganha, porque identidade de NT em Drosophila e fortemente
ligada a tipo celular, que e fortemente ligado a anatomia, e porque o rotulador
de EM tem correlacao espacial. G4 passa.
Se G3 falhar como eu previ, a conclusao pre-registrada NAO e "achamos uma lei":
e que o sinal e anatomico e o dado nao consegue distinguir biologia de viés do
rotulador — e o gap revelado e que **precisamos de rotulo de NT medido, nao
predito**, para responder a pergunta. Isso e um gap real e acionavel.
"""

from __future__ import annotations

import argparse
import time
from math import log2

import numpy as np
import pandas as pd
import pyarrow.feather as feather

ARQ = "D:/dados_cerebro/proofread_connections_783.feather"
SEMENTE = 20261026
NTS = ("gaba", "ach", "glut")


# ─── o dado ──────────────────────────────────────────────────────────────────

def carrega(caminho):
    t = feather.read_table(caminho, memory_map=True,
                           columns=["pre_pt_root_id", "post_pt_root_id",
                                    "neuropil", "syn_count",
                                    "gaba_avg", "ach_avg", "glut_avg"])
    return t.to_pandas()


def rotulo_nt(df):
    """NT de cada neuronio: media das probabilidades ponderada por sinapse,
    depois argmax. Mesma definicao usada em `conectoma.py`."""
    w = df["syn_count"].to_numpy(float)
    pre = df["pre_pt_root_id"].to_numpy()
    ids, inv = np.unique(pre, return_inverse=True)
    tot = np.bincount(inv, weights=w, minlength=len(ids))
    M = np.stack([np.bincount(inv, weights=df[f"{n}_avg"].to_numpy(float) * w,
                              minlength=len(ids)) / np.maximum(tot, 1e-9)
                  for n in NTS], 1)
    return ids, M.argmax(1).astype(np.int8)


def neuropilo_dominante(df):
    """Onde cada neuronio tem MAIS sinapses."""
    lado = pd.concat([
        df[["pre_pt_root_id", "neuropil", "syn_count"]].rename(
            columns={"pre_pt_root_id": "id"}),
        df[["post_pt_root_id", "neuropil", "syn_count"]].rename(
            columns={"post_pt_root_id": "id"}),
    ])
    lado = lado[lado.neuropil.notna() & (lado.neuropil != "None")]
    por = lado.groupby(["id", "neuropil"], sort=False)["syn_count"].sum()
    casa = por.reset_index().sort_values("syn_count").drop_duplicates(
        "id", keep="last")
    return dict(zip(casa["id"].to_numpy(), casa["neuropil"].to_numpy()))


# ─── a moeda: bits por neuronio escondido ────────────────────────────────────

def _entropia(conta):
    n = conta.sum()
    if n <= 0:
        return 0.0
    p = conta / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def bits_sob_modelo(chave_te, rot_te, tabela, marg, piso=0.5):
    """Bits por neuronio para codificar os rotulos ESCONDIDOS usando a
    distribuicao condicional estimada NA METADE DE AJUSTE. Quem nao tem chave
    conhecida cai na marginal — e paga a marginal, sem desconto."""
    if len(rot_te) == 0:
        return float("nan")
    tot = 0.0
    pm = (marg + piso) / (marg.sum() + piso * len(marg))
    for k, r in zip(chave_te, rot_te):
        c = tabela.get(k)
        if c is None or c.sum() < 5:
            p = pm
        else:
            p = (c + piso) / (c.sum() + piso * len(c))
        tot += -log2(max(p[r], 1e-12))
    return tot / len(rot_te)


def tabela_condicional(chave_tr, rot_tr):
    """conta[chave][nt], estimada SO na metade de ajuste."""
    t = {}
    for k, r in zip(chave_tr, rot_tr):
        c = t.get(k)
        if c is None:
            c = np.zeros(len(NTS))
            t[k] = c
        c[r] += 1
    return t


# ─── as hipoteses ────────────────────────────────────────────────────────────

def chaves_parceiro(df, ids, rot, ajuste, direcao):
    """Para cada neuronio, a MAIORIA do NT dos parceiros que estao na metade de
    ajuste. Parceiro escondido nao entra — e isto que impede o vazamento."""
    pos = {i: k for k, i in enumerate(ids)}
    a = df["pre_pt_root_id"].to_numpy()
    b = df["post_pt_root_id"].to_numpy()
    if direcao == "pos":
        eu, dele = a, b
    elif direcao == "pre":
        eu, dele = b, a
    else:
        eu = np.concatenate([a, b])
        dele = np.concatenate([b, a])
    ie = np.array([pos.get(x, -1) for x in eu])
    idl = np.array([pos.get(x, -1) for x in dele])
    ok = (ie >= 0) & (idl >= 0)
    ie, idl = ie[ok], idl[ok]
    vale = ajuste[idl]                       # so parceiro da metade de ajuste
    ie, idl = ie[vale], idl[vale]
    conta = np.zeros((len(ids), len(NTS)), dtype=np.int32)
    np.add.at(conta, (ie, rot[idl]), 1)
    tem = conta.sum(1) > 0
    maioria = np.where(tem, conta.argmax(1), -1)
    return maioria


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=ARQ)
    ap.add_argument("--permuta", action="store_true",
                    help="NULO: embaralha os rotulos de NT entre neuronios")
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()

    print("=" * 78)
    print(f"NT x CONECTIVIDADE — FlyWire 783"
          f"{'  [NULO PERMUTADO]' if a.permuta else ''}")
    print("=" * 78)
    df = carrega(a.arquivo)
    ids, rot = rotulo_nt(df)
    if a.permuta:
        rot = rot[rng.permutation(len(rot))]
    print(f"  {len(df):,} conexoes, {len(ids):,} neuronios com NT "
          f"({time.time()-t0:.0f}s)")

    marg = np.bincount(rot, minlength=len(NTS)).astype(float)
    H = _entropia(marg)
    print(f"  marginal: " + "  ".join(f"{n} {100*marg[k]/marg.sum():.1f}%"
                                      for k, n in enumerate(NTS)))
    print(f"  LINHA DE BASE (entropia marginal): {H:.4f} bits/neuronio")

    ajuste = rng.random(len(ids)) < 0.5
    esconde = ~ajuste
    print(f"  ajuste {ajuste.sum():,}  escondidos {esconde.sum():,}")

    npl = neuropilo_dominante(df)
    grau = np.zeros(len(ids))
    pos = {i: k for k, i in enumerate(ids)}
    gs = df.groupby("pre_pt_root_id")["syn_count"].count()
    for i, v in gs.items():
        k = pos.get(i)
        if k is not None:
            grau[k] = v

    hip = {}
    for d in ("pos", "pre", "amb"):
        hip[f"parceiro_{d}"] = ("PARCEIRO",
                                chaves_parceiro(df, ids, rot, ajuste, d),
                                len(NTS))
    hip["neuropilo"] = ("ESPACO",
                        np.array([hash(npl.get(i, "?")) % (2 ** 31) for i in ids]),
                        len(set(npl.values())))
    q = np.quantile(grau[grau > 0], np.linspace(0, 1, 11)) if (grau > 0).any() \
        else np.array([0, 1])
    hip["grau_decil"] = ("ESPACO", np.digitize(grau, q), 10)

    print(f"\n  {'hipotese':18s} {'familia':9s} {'bits/neur':>10s} "
          f"{'economia':>9s} {'estrutura':>10s} {'liquido':>9s}")
    res = {}
    for nome, (fam, chave, npar) in hip.items():
        ktr = chave[ajuste]
        rtr = rot[ajuste]
        bom = ktr >= 0 if chave.dtype.kind in "iu" else np.ones(len(ktr), bool)
        tab = tabela_condicional(ktr[bom], rtr[bom])
        b = bits_sob_modelo(chave[esconde], rot[esconde], tab, marg)
        econ = H - b
        est = npar * (len(NTS) - 1) * 0.5 * log2(max(int(esconde.sum()), 2)) \
            / max(int(esconde.sum()), 1)
        liq = econ - est
        res[nome] = (fam, econ, est, liq)
        print(f"  {nome:18s} {fam:9s} {b:10.4f} {econ:9.4f} {est:10.5f} "
              f"{liq:9.4f}")

    mp = max((v[3], k) for k, v in res.items() if v[0] == "PARCEIRO")
    me = max((v[3], k) for k, v in res.items() if v[0] == "ESPACO")
    melhor = max((v[3], k) for k, v in res.items())
    print("")
    print("=" * 78)
    ok0 = 1.1 <= H <= 1.4
    print(f"G0 {'.' if ok0 else 'FALHOU .'} linha de base {H:.4f} (1,1-1,4)")
    if a.permuta:
        pior = max(v[3] for v in res.values())
        print(f"G1 {'.' if pior <= 0.01 else 'FALHOU .'} NULO: melhor economia "
              f"{pior:+.4f} bit (<= 0,01)")
        print("\n  (rode sem --permuta para G2, G3 e G4)")
        return
    print(f"G2 {'.' if melhor[0] >= 0.05 else 'FALHOU .'} alguma lei paga: "
          f"{melhor[1]} com {melhor[0]:+.4f} bit (>= 0,05)")
    print(f"G3 {'.' if mp[0] > me[0] else 'FALHOU .'} PARCEIRO > ESPACO: "
          f"{mp[1]} {mp[0]:+.4f} vs {me[1]} {me[0]:+.4f}")
    v = res[melhor[1]]
    print(f"G4 {'.' if v[2] < 0.01 * max(v[1], 1e-9) else 'FALHOU .'} e curta: "
          f"estrutura {v[2]:.5f} < 1% de {v[1]:.4f}")
    print(f"\n  G1 (nulo) roda com --permuta")


if __name__ == "__main__":
    main()

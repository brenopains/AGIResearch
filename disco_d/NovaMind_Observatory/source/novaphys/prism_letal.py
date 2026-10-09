"""
novaphys.prism_letal — remedio que ja existe e mata seletivamente um genotipo de cancer
=========================================================================================

POR QUE ESTE ALVO, DEPOIS DO QUE A LETALIDADE SINTETICA ENSINOU
------------------------------------------------------------------
No DepMap CRISPR os portoes mecanicos passaram e deu 0/8 inedito: os sinais FORTES
de letalidade sintetica em dado publico ja foram achados e viraram farmaco (o
9p21 -> PELO/HBS1L virou o TNG961). Esse espaco esta garimpado.

A triagem de REPOSICIONAMENTO do PRISM e menos garimpada num eixo especifico:
centenas de remedios NAO oncologicos (antifungicos, antipsicoticos, anti-
hipertensivos...) testados contra centenas de linhagens de cancer. Um remedio
aprovado para outra doenca que mata seletivamente celulas com uma perda
genetica especifica e hipotese de reposicionamento — o tipo de resultado que
vira ensaio clinico mais rapido que alvo novo, porque o remedio ja existe.

O DESENHO, E POR QUE A REPLICACAO AGORA E DE VERDADE
-----------------------------------------------------
  DESCOBERTA   PRISM Repurposing Public 24Q2 — SO as triagens NOVAS de 2024
               (REP1M e REP300): 1.514 compostos, dose unica 2,5 uM, 5 dias,
               906 linhagens. O README avisa que a matriz "Extended Primary"
               MISTURA essas triagens com a primaria de 2019; a primaria de 2019
               fica FORA da descoberta, porque os compostos da tela secundaria
               foram escolhidos a partir dos acertos dela.
  REPLICACAO   PRISM Repurposing 20Q2 SECUNDARIA — outro experimento, outro ano,
               outro ensaio: curvas de 8 doses (10 uM, diluicao 4x), AUC por
               linhagem. Replicacao ENTRE ENSAIOS, nas linhagens em comum.
               Ressalva declarada: mesmas linhagens nos dois ensaios, entao um
               artefato proprio de uma linhagem poderia replicar junto; o nulo
               estratificado controla tecido e carga mutacional, nao tudo.
  GENOTIPO     DepMap 24Q2: perda (mutacao danosa OU delecao profunda) e, novo,
               mutacao pontual ATIVADORA (hotspot), que e onde estao os controles
               limpos de remedio.

OS TRES CONSERTOS DE `novidade.py`, JA APLICADOS
---------------------------------------------------
  SEGMENTO   genotipos que co-ocorrem (Jaccard >= 0,5) sao UM evento (9p21 inteira)
  MODULO     remedios com perfil de sensibilidade co-variante (|corr| >= 0,4) sao
             UM modulo — "todos os inibidores de MEK" nao contam como 7 achados
  EVENTO     replicado no nivel do evento, nao do par, para nao premiar carona
  NOVIDADE   evento cujo remedio tem o PROPRIO gene como alvo anotado (BRAF mutado
             -> inibidor de BRAF) e classe conhecida; e o portao final continua
             sendo busca de literatura no nivel do evento

PRE-REGISTRO — ANTES DE BAIXAR QUALQUER ARQUIVO DO PRISM
----------------------------------------------------------
Controles positivos (genotipo -> sensibilidade estabelecida):
    BRAF hotspot   -> inibidor de BRAF (dabrafenib, vemurafenib, PLX-4720,
                      encorafenib)
    NRAS hotspot   -> inibidor de MEK (trametinib, selumetinib, cobimetinib,
                      binimetinib)
    KRAS hotspot   -> inibidor de MEK
    EGFR hotspot   -> inibidor de EGFR (erlotinib, gefitinib, osimertinib, afatinib)
    PIK3CA hotspot -> inibidor de PI3K-alfa (alpelisib, taselisib)
    PTEN perda     -> inibidor de AKT/PI3K (capivasertib, ipatasertib, MK-2206)
    BRCA1/2 perda  -> inibidor de PARP (olaparib, talazoparib, niraparib)
Controle sem >= 20 linhagens com o genotipo na descoberta, ou sem nenhum remedio
correspondente no dado, e NAO TESTAVEL e sai do denominador.

  P1  CONTROLES: >= 60% dos testaveis aceitos, com >= 4 testaveis
  P2  NULO: falsas descobertas esperadas <= 1 (10 nulos estratificados)
  P3  REPLICA ENTRE ENSAIOS: >= 60% dos eventos aceitos testaveis na secundaria
      replicam (mais sensivel, |efeito| >= 0,1 em AUC padronizada)
  P4  INEDITO (mecanico): >= 1 evento aceito, replicado, que nao e controle, nao
      e classe on-target e cujo remedio NAO e oncologico
  P5  INEDITO (final): o evento de P4 sobrevive a busca de literatura no nivel do
      evento (remedio/mecanismo x genotipo) sem artigo que o reporte
PREVISAO: P1 raspa ou falha — BRAF e NRAS->MEK passam forte, o resto e fraco
nesta dose unica; aposto 3 a 4 de 6-7 testaveis. P2 passa. P3 aposto 55-75%
(dose unica e ruidosa). P4 aposto 1 a 5 eventos. P5 aposto ~20%: remedio nao
oncologico com efeito forte e seletivo tende a ter mecanismo toxico inespecifico
que alguem ja descreveu.

O QUE UM PASSE EM P5 SERIA E NAO SERIA
----------------------------------------
SERIA: uma hipotese de reposicionamento — um remedio ja aprovado para outra
doenca, seletivamente toxico para um genotipo de cancer, achado sem eu apontar,
replicado num segundo ensaio independente, fora da literatura buscada.
NAO SERIA: cura. 2,5 uM em placa nao e dose tolerada em paciente, linhagem nao e
tumor, e a hipotese precisa de validacao experimental e farmacologica.
"""

from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import pandas as pd

import novaphys.letalidade_sintetica as S
from novaphys.novidade import agrupa, segmentos_globais

PASTA = "D:/dados_cancer/prism"
SEMENTE = 20261031
JAC, COR = 0.5, 0.4

CONTROLES = {
    "BRAF_hot": ("dabrafenib", "vemurafenib", "plx-4720", "plx4720", "encorafenib"),
    "NRAS_hot": ("trametinib", "selumetinib", "cobimetinib", "binimetinib"),
    "KRAS_hot": ("trametinib", "selumetinib", "cobimetinib", "binimetinib"),
    "EGFR_hot": ("erlotinib", "gefitinib", "osimertinib", "afatinib"),
    "PIK3CA_hot": ("alpelisib", "taselisib", "byl-719", "byl719"),
    "PTEN": ("capivasertib", "ipatasertib", "mk-2206", "mk2206"),
    "BRCA2": ("olaparib", "talazoparib", "niraparib", "rucaparib"),
    "BRCA1": ("olaparib", "talazoparib", "niraparib", "rucaparib"),
}


def nucleo(bid):
    """BRD-K12345678-001-01-2 -> BRD-K12345678 (colapsa lotes do mesmo composto)."""
    m = re.match(r"(BRD-[A-Z]\d{8})", str(bid))
    return m.group(1) if m else str(bid)


def _col(cols, *palavras):
    for c in cols:
        lc = str(c).lower()
        if all(p in lc for p in palavras):
            return c
    return None


# ─── leitura ─────────────────────────────────────────────────────────────────

def descoberta_24q2(pasta=PASTA):
    """LFC colapsado das triagens NOVAS (REP1M/REP300), em matriz linhagem x
    composto. Linhagem por depmap_id; composto pelo nucleo do broad_id."""
    t = pd.read_csv(os.path.join(pasta, "Repurposing_Public_24Q2_LFC_COLLAPSED.csv"),
                    low_memory=False)
    cols = list(t.columns)
    c_lin = _col(cols, "depmap")
    if c_lin is None:
        meta = pd.read_csv(os.path.join(
            pasta, "Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv"), low_memory=False)
        rid = _col(cols, "row_id")
        mapa = dict(zip(meta["row_id"], meta["depmap_id"]))
        t["depmap_id"] = t[rid].map(mapa)
        c_lin = "depmap_id"
    c_bid = _col(cols, "broad")
    c_val = next((c for c in cols if str(c).lower() in ("lfc_cb", "lfc.cb")), None) \
        or next((c for c in cols if str(c).lower() == "lfc"), None) \
        or _col(cols, "lfc")
    t = t[t[c_lin].notna() & t[c_bid].notna()]
    # DEFEITO CORRIGIDO (visto no cabecalho real): a falha de controle de
    # qualidade vem EMBUTIDA no identificador ("BRD-A09155491-001-03-9 - QC
    # Failure"). Cortar pelo nucleo do broad_id misturaria essas medidas falhas
    # com as boas, em silencio.
    qc = t[c_bid].astype(str).str.contains("QC Failure", case=False)
    print(f"  descoberta: {int(qc.sum()):,} medidas com falha de QC descartadas "
          f"de {len(t):,}")
    t = t[~qc]
    t["composto"] = t[c_bid].map(nucleo)
    X = t.pivot_table(index=c_lin, columns="composto", values=c_val, aggfunc="median")
    X.index = X.index.astype(str)
    trat = pd.read_csv(os.path.join(pasta, "Repurposing_Public_24Q2_Treatment_Meta_Data.csv"),
                       low_memory=False)
    nomes = {}
    if "broad_id" in trat.columns and "name" in trat.columns:
        for b, n in zip(trat["broad_id"], trat["name"]):
            if isinstance(b, str) and isinstance(n, str):
                nomes.setdefault(nucleo(b), n)
    return X, nomes


def replicacao_20q2(pasta=PASTA):
    """AUC da tela secundaria (curvas de 8 doses), linhagem x composto, com a
    anotacao do Repurposing Hub (moa, alvo, area de doenca, fase)."""
    t = pd.read_csv(os.path.join(
        pasta, "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv"),
        low_memory=False)
    t = t[t["depmap_id"].notna() & t["broad_id"].notna() & t["auc"].notna()]
    t["composto"] = t["broad_id"].map(nucleo)
    # o README da secundaria recomenda MTS010 quando existe: mesmo composto numa
    # tela antiga e na MTS010 nao se tira media entre pipelines diferentes
    t["_pref"] = (t["screen_id"].astype(str) != "MTS010").astype(int)
    t = t.sort_values("_pref").drop_duplicates(["composto", "depmap_id"])
    A = t.pivot_table(index="depmap_id", columns="composto", values="auc",
                      aggfunc="mean")
    A.index = A.index.astype(str)
    ann = (t.drop_duplicates("composto").set_index("composto")
           [[c for c in ("name", "moa", "target", "disease.area", "phase")
             if c in t.columns]])
    return A, ann


def genotipo_24q2():
    """Perda (mutacao danosa | delecao profunda) + hotspot ativador."""
    E, L, lin, carga = S.carrega("24Q2")
    hot = os.path.join(S.PASTA, "24Q2", "OmicsSomaticMutationsMatrixHotspot.csv")
    if os.path.exists(hot):
        H = S._matriz(hot).reindex(L.index)
        Hb = pd.DataFrame(H.fillna(0).to_numpy() >= 1, index=L.index,
                          columns=[f"{g}_hot" for g in H.columns])
        Hb = Hb.loc[:, Hb.sum() > 0]
        L = pd.concat([L, Hb], axis=1)
    return L, lin, carga


# ─── eventos ─────────────────────────────────────────────────────────────────

def colapsa(aceitos, Lm, Xr, colsA, colsB):
    # segmento vem de `segmentos_globais`: carona e juntado ao gene
    # causal mesmo quando o causal nao foi aceito (defeito NEDD4L/VPS4B)
    As = list(dict.fromkeys(colsA[a] for _b, a, _d, _x in aceitos))
    Bs = list(dict.fromkeys(colsB[b] for _b, _a, b, _x in aceitos))
    ia = {g: i for i, g in enumerate(colsA)}
    ib = {g: i for i, g in enumerate(colsB)}

    def jac(u, v):
        x, y = Lm[:, ia[u]], Lm[:, ia[v]]
        return (x & y).sum() / max((x | y).sum(), 1) >= JAC

    def cor(u, v):
        return abs(np.corrcoef(Xr[:, ib[u]], Xr[:, ib[v]])[0, 1]) >= COR

    seg = segmentos_globais(Lm, colsA, As, JAC)
    mod = agrupa(Bs, cor)
    for g in As:
        seg.setdefault(g, -hash(g) % 10 ** 6)
    ev = {}
    for bits, a, b, d in aceitos:
        k = (seg[colsA[a]], mod[colsB[b]])
        e = ev.setdefault(k, dict(bits=bits, A=set(), B=set(), rep=(colsA[a], colsB[b]),
                                  delta=d))
        e["A"].add(colsA[a])
        e["B"].add(colsB[b])
        if bits > e["bits"]:
            e.update(bits=bits, rep=(colsA[a], colsB[b]), delta=d)
    return sorted(ev.values(), key=lambda e: -e["bits"])


def replica(e, Lg, Arep, lin_rep):
    """Mesmo sinal na secundaria: linhagens com QUALQUER genotipo do segmento x
    sem, na AUC padronizada media dos remedios do modulo (menor = mais sensivel)."""
    drogas = [c for c in e["B"] if c in Arep.columns]
    genes = [g for g in e["A"] if g in Lg.columns]
    linhas = [i for i in Arep.index if i in Lg.index]
    if not drogas or not genes or len(linhas) < 20:
        return None, 0
    Z = Arep.loc[linhas, drogas]
    Z = (Z - Z.mean()) / (Z.std() + 1e-6)
    z = Z.mean(axis=1).to_numpy().copy()  # pandas 3: to_numpy() e somente-leitura
    lr = lin_rep.reindex(linhas).fillna("?").to_numpy()
    for l in np.unique(lr):
        m = lr == l
        z[m] = z[m] - np.nanmean(z[m])
    g = Lg.loc[linhas, genes].to_numpy().any(1)
    ok = ~np.isnan(z)
    if (g & ok).sum() < 5 or (~g & ok).sum() < 5:
        return None, int((g & ok).sum())
    return float(np.nanmean(z[g & ok]) - np.nanmean(z[~g & ok])), int((g & ok).sum())


# ─── a corrida ───────────────────────────────────────────────────────────────

def main() -> None:
    # DESENHO INVALIDADO PELA ESTRUTURA DO DADO, ANTES DE QUALQUER RESULTADO.
    # Validacao dos leitores no arquivo real mostrou: so 13 compostos em comum
    # entre as triagens de 2024 (1.518) e a tela secundaria de 2020 (1.494), e
    # NENHUM remedio de controle (dabrafenib, trametinib, erlotinib, olaparib,
    # alpelisib...) nas triagens de 2024. Este desenho daria P1 sem controle
    # testavel e P3 com no maximo 13 compostos. Nenhuma associacao remedio x
    # genotipo foi olhada. Nao rodar.
    raise SystemExit("prism_letal: desenho invalidado pela estrutura do dado "
                     "(13 compostos em comum, 0 controles na descoberta). "
                     "Nao rodar este desenho.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulos", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("=" * 78)
    print("PRISM — remedio x genotipo: descoberta 2024 (dose unica), replicacao 2020 "
          "(8 doses)")
    print("=" * 78)

    X, nomes = descoberta_24q2()
    L, lin, carga = genotipo_24q2()
    linhas = [i for i in X.index if i in L.index]
    X = X.loc[linhas]
    ok = X.isna().mean() <= 0.2
    X = X.loc[:, ok]
    X = X.fillna(X.mean())
    Lg = L.loc[linhas]
    lin_a = lin.loc[linhas].to_numpy()
    n = len(linhas)
    print(f"  {n} linhagens com genotipo e triagem, {X.shape[1]:,} compostos, "
          f"{Lg.shape[1]:,} genotipos ({time.time()-t0:.0f}s)")

    tr = np.zeros(n, bool)
    tr[rng.permutation(n)[: n // 2]] = True
    te = ~tr
    estr = S.estratos2(lin_a, carga.loc[linhas], tr)
    Xr = S.residualiza(X.to_numpy(np.float32), estr, tr)
    Lm = Lg.to_numpy()
    real, _ = S.varre(Xr, Lm, tr, te)
    maximos, vals = [], []
    for k in range(a.nulos):
        v, mx = S.varre(Xr, S.permuta_em_estratos(Lm, estr, rng), tr, te)
        maximos.append(mx)
        vals.append(np.array([x[0] for x in v]))
        print(f"    nulo {k+1}/{a.nulos}: maximo {mx:8.1f} bits ({time.time()-t0:.0f}s)",
              flush=True)
    bruto = max(maximos)
    T = max(0.0, bruto)
    fd = float(np.mean([(v >= T).sum() for v in vals]))
    aceitos = [r for r in real if r[0] > T]
    colsA, colsB = list(Lg.columns), list(X.columns)
    print(f"\n  maximo bruto do nulo {bruto:.1f}; limiar {T:.1f}; falsas descobertas "
          f"esperadas {fd:.2f}; pares aceitos {len(aceitos):,}")

    Arep, ann = replicacao_20q2()
    nome_de = lambda c: str(ann["name"].get(c, nomes.get(c, c))) \
        if "name" in ann.columns else nomes.get(c, c)

    # controles
    ia = {g: i for i, g in enumerate(colsA)}
    acc_pares = {(colsA[x[1]], colsB[x[2]]) for x in aceitos}
    testaveis, passaram = [], []
    print("\n  CONTROLES")
    for gen, drogas in CONTROLES.items():
        alvo = [c for c in colsB if any(d in nome_de(c).lower() for d in drogas)]
        k = int(Lm[tr, ia[gen]].sum()) if gen in ia else 0
        if gen not in ia or k < S.MIN_ALT or not alvo:
            print(f"    {gen:11s} NAO TESTAVEL (genotipo {k}, remedios {len(alvo)})")
            continue
        testaveis.append(gen)
        hit = [nome_de(c) for c in alvo if (gen, c) in acc_pares]
        if hit:
            passaram.append(gen)
        print(f"    {gen:11s} {k:3d} linhagens  {'ACEITO: ' + ', '.join(hit) if hit else 'nao'}")

    ev = colapsa(aceitos, Lm, Xr, colsA, colsB)
    ctrl_gen = set(CONTROLES)
    testados = replicou = 0
    candidatos = []
    print(f"\n  {len(aceitos):,} pares -> {len(ev)} eventos")
    print(f"  {'bits':>7s} {'genotipo':12s} {'remedio':26s} {'d rep':>6s} {'n':>4s}  "
          f"area / alvo")
    for e in ev:
        d, nn = replica(e, Lg, Arep, lin)
        if d is not None:
            testados += 1
            # replica = mais sensivel (d < 0) com efeito nao trivial:
            # |d| >= 0,1 em unidades padronizadas de AUC
            replicou += int(d < 0 and abs(d) >= 0.1)
        ga, cb = e["rep"]
        nm = nome_de(cb)
        area = str(ann["disease.area"].get(cb, "?")) if "disease.area" in ann.columns else "?"
        alvo = str(ann["target"].get(cb, "")) if "target" in ann.columns else ""
        on_target = ga.replace("_hot", "") in [x.strip() for x in alvo.split(",")]
        # DEFEITO CORRIGIDO: anlotinib (inibidor de VEGFR, oncologico) tem area
        # de doenca VAZIA na anotacao. "Nao contem oncology" deixaria vazio passar
        # como nao oncologico. Vazio = desconhecido = fora dos candidatos.
        area_l = area.strip().lower()
        onc = (not area_l) or area_l in ("nan", "?") or "oncology" in area_l
        rep_ok = d is not None and d < 0
        eh_ctrl = bool(e["A"] & ctrl_gen)
        if rep_ok and not eh_ctrl and not on_target and not onc:
            candidatos.append((e, nm, area, alvo, d, nn))
        print(f"  {e['bits']:7.1f} {ga:12s} {nm[:26]:26s} "
              f"{('%+.2f' % d) if d is not None else '     -':>6s} {nn:4d}  "
              f"{area[:18]} / {alvo[:24]}{' [ON-TARGET]' if on_target else ''}")

    print("")
    print("=" * 78)
    fr = len(passaram) / max(len(testaveis), 1)
    print(f"P1 {'.' if len(testaveis) >= 4 and fr >= 0.6 else 'FALHOU .'} controles "
          f"{len(passaram)}/{len(testaveis)}")
    print(f"P2 {'.' if fd <= 1 else 'FALHOU .'} falsas descobertas esperadas {fd:.2f}")
    fq = replicou / max(testados, 1)
    print(f"P3 {'.' if testados and fq >= 0.6 else 'FALHOU .'} replica entre ensaios "
          f"{replicou}/{testados} ({100*fq:.0f}%)")
    print(f"P4 {'.' if candidatos else 'FALHOU .'} candidatos nao oncologicos, nao "
          f"on-target, replicados: {len(candidatos)}")
    for e, nm, area, alvo, d, nn in candidatos[:15]:
        print(f"      {','.join(sorted(e['A']))[:30]:30s} -> {nm} ({area}; alvo {alvo}) "
              f"d_rep {d:+.2f} n={nn}")
    print("P5 exige busca de literatura no nivel do evento — feita a mao depois")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

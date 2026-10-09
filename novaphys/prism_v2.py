"""
novaphys.prism_v2 — reposicionamento: descoberta em 8 doses, replicacao em dose unica
=======================================================================================

POR QUE V2 (declarado ANTES de olhar qualquer associacao remedio x genotipo)
------------------------------------------------------------------------------
Validar os leitores de `prism_letal.py` nos arquivos reais mostrou a ESTRUTURA do
dado: so 13 compostos em comum entre as triagens de 2024 e a secundaria de 2020,
e nenhum remedio de controle nas de 2024. A v2 inverte os papeis:
  DESCOBERTA  PRISM 20Q2 SECUNDARIA: 1.494 compostos x 737 linhagens, 8 doses
              (AUC), com todos os controles.
  REPLICACAO  PRISM 20Q2 PRIMARIA: os mesmos compostos em dose unica, outro
              experimento. Ressalva: a secundaria reuniu compostos ATIVOS na
              primaria (atividade geral), e as linhagens se sobrepoem.

HIPERMUTACAO: MARCAR, NAO REMOVER (medido em teste_consertos.py)
------------------------------------------------------------------
Regressao na carga deixou KMT2D->WRN (4,1 bits) e matou ARID1A->ARID1B. Estrato
tecido x hipermutacao matou o falso positivo E os tres controles (42 estratos,
76 hipermutadas em grupos de 1 a 5; nulo bruto -4,0 -> +6,6). Ajustar ate os
controles voltarem seria calibrar no gabarito. Entao: residuo por tecido, nulo
por tecido x tercil de carga, e evento MARCADO se o genotipo concentra >= 3x a
taxa de base de hipermutadas.

V2.1 (declarado ANTES de rodar, pela estrutura medida, nao por resultado)
--------------------------------------------------------------------------
A secundaria sao QUATRO telas; junta-las criou falta artificial e derrubou
1.325 dos 1.494 compostos. Descoberta agora so na HTS002 (480 x 1.319, 88,5%
densa). Replicacao na MTS010, que o README descreve como redo tecnico dos
mesmos compostos, em 734 linhagens — ja esta no disco. Limite declarado: so
~154 compostos existem nas duas, a maioria oncologica; candidato nao
oncologico fora da MTS010 fica 'nao replicavel' ate haver a primaria.
E o custo de estrutura passou a cobrar so genotipos TESTAVEIS (2.105, nao
26.169): 3,7 bits que eram cobrados de todo par por candidato nunca testado.

PRE-REGISTRO V2
-----------------
Controles: BRAF hot -> inib. BRAF; NRAS/KRAS hot -> MEK; EGFR hot -> inib. EGFR;
PIK3CA hot -> PI3K-alfa; PTEN perda -> MK-2206; BRCA1/2 perda -> PARP; TP53
INTACTO -> nutlin/idasanutlin.
  Q1  controles >= 60% dos testaveis, com >= 4 testaveis
  Q2  falsas descobertas esperadas <= 1
  Q3  >= 60% dos eventos testaveis replicam na MTS010 (mais sensivel e
      |efeito| >= 0,1 em AUC padronizada)
  Q4  >= 1 evento replicado, nao controle, nao on-target, nao marcado por
      hipermutacao, remedio nao oncologico (area preenchida, sem "oncology" nem
      "malignancy")
  Q5  sobrevive a busca de literatura no nivel do evento
PREVISAO: Q1 passa; Q2 passa; Q3 60-80%; Q4 1 a 6 eventos; Q5 ~20%.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import novaphys.letalidade_sintetica as S
import novaphys.prism_letal as P

PASTA = P.PASTA
SEMENTE = 20261101
CONTROLES = dict(P.CONTROLES)
CONTROLES["TP53_intacto"] = ("nutlin", "idasanutlin", "rg7112", "amg-232")


def primaria_20q2(pasta=PASTA):
    """LFC da primaria (dose unica): linhagem (depmap_id) x composto (nucleo)."""
    base = "prism-repurposing-20q2-primary-screen-"
    M = pd.read_csv(os.path.join(pasta, base + "replicate-collapsed-logfold-change.csv"),
                    index_col=0, low_memory=False)
    info = pd.read_csv(os.path.join(pasta, base + "replicate-collapsed-treatment-info.csv"),
                       low_memory=False)
    cel = pd.read_csv(os.path.join(pasta, base + "cell-line-info.csv"), low_memory=False)
    ml = dict(zip(cel["row_name"].astype(str), cel["depmap_id"].astype(str)))
    mc = dict(zip(info["column_name"].astype(str), info["broad_id"].map(P.nucleo)))
    M.index = [ml.get(str(i), str(i)) for i in M.index]
    M.columns = [mc.get(str(c), P.nucleo(c)) for c in M.columns]
    M = M.T.groupby(level=0).median().T
    M = M[pd.Index(M.index).astype(str).str.startswith("ACH-")]
    return M.groupby(level=0).median()


def secundaria_screen(screen="HTS002", pasta=PASTA):
    """AUC de UMA tela, em matriz densa, com a anotacao do Repurposing Hub.

    DEFEITO CORRIGIDO: juntar as quatro telas numa matriz so cria FALTA
    ARTIFICIAL — um composto de HTS002 nao tem valor nas linhagens que so existem
    em MTS010. Isso derrubou 1.325 dos 1.494 compostos no filtro de 30%. Por tela:
    HTS002 = 480 x 1.319; MTS010 = 734 x 154 (redo tecnico dos mesmos compostos,
    logo serve de replicacao sem baixar nada)."""
    t = pd.read_csv(os.path.join(
        pasta, "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv"),
        low_memory=False)
    t = t[t.depmap_id.notna() & t.broad_id.notna() & t.auc.notna()
          & (t.screen_id.astype(str) == screen)]
    t["composto"] = t.broad_id.map(P.nucleo)
    A = t.pivot_table(index="depmap_id", columns="composto", values="auc",
                      aggfunc="mean")
    A.index = A.index.astype(str)
    ann = (t.drop_duplicates("composto").set_index("composto")
           [[c for c in ("name", "moa", "target", "disease.area", "phase")
             if c in t.columns]])
    return A, ann


def genotipo():
    """Perda + hotspot (de prism_letal) + TP53 INTACTO para o controle de nutlin."""
    L, lin, carga = P.genotipo_24q2()
    tp = np.zeros(len(L), bool)
    for c in ("TP53", "TP53_hot"):
        if c in L.columns:
            tp |= L[c].to_numpy().astype(bool)
    L["TP53_intacto"] = ~tp
    return L, lin, carga


def replica_prim(e, Lg, Pm, lin):
    """Mesmo sinal em dose unica: linhagens com QUALQUER genotipo do segmento x
    sem, no LFC padronizado medio dos compostos do modulo, residuo por tecido."""
    drogas = [c for c in e["B"] if c in Pm.columns]
    genes = [g for g in e["A"] if g in Lg.columns]
    linhas = [i for i in Pm.index if i in Lg.index]
    if not drogas or not genes or len(linhas) < 20:
        return None, 0
    Z = Pm.loc[linhas, drogas]
    Z = (Z - Z.mean()) / (Z.std() + 1e-6)
    z = Z.mean(axis=1).to_numpy().copy()  # pandas 3: to_numpy() e somente-leitura
    lr = lin.reindex(linhas).fillna("?").to_numpy()
    for l in np.unique(lr):
        m = lr == l
        z[m] = z[m] - np.nanmean(z[m])
    g = Lg.loc[linhas, genes].to_numpy().any(1)
    ok = ~np.isnan(z)
    if (g & ok).sum() < 5 or (~g & ok).sum() < 5:
        return None, int((g & ok).sum())
    return float(np.nanmean(z[g & ok]) - np.nanmean(z[~g & ok])), int((g & ok).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulos", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("=" * 78)
    print("PRISM v2.1 — descoberta HTS002 (8 doses), replicacao MTS010 (redo tecnico)")
    print("=" * 78)
    A, ann = secundaria_screen("HTS002")
    L, lin, carga = genotipo()
    linhas = [i for i in A.index if i in L.index]
    A = A.loc[linhas]
    A = A.loc[:, A.isna().mean() <= 0.3]
    A = A.fillna(A.mean())
    Lg = L.loc[linhas]
    lin_a = lin.loc[linhas].to_numpy()
    n = len(linhas)
    print(f"  {n} linhagens, {A.shape[1]:,} compostos, {Lg.shape[1]:,} genotipos "
          f"({time.time()-t0:.0f}s)", flush=True)
    tr = np.zeros(n, bool)
    tr[rng.permutation(n)[: n // 2]] = True
    te = ~tr
    Xr = S.residualiza(A.to_numpy(np.float32), lin_a, tr)
    Lm = Lg.to_numpy()
    real, _ = S.varre(Xr, Lm, tr, te)
    estr = S.estratos(lin_a, carga.loc[linhas])
    maximos, vals = [], []
    for k in range(a.nulos):
        v, mx = S.varre(Xr, S.permuta_em_estratos(Lm, estr, rng), tr, te)
        maximos.append(mx)
        vals.append(np.array([x[0] for x in v]))
        print(f"    nulo {k+1}/{a.nulos}: maximo {mx:8.1f} bits "
              f"({time.time()-t0:.0f}s)", flush=True)
    bruto = max(maximos)
    T = max(0.0, bruto)
    fd = float(np.mean([(v >= T).sum() for v in vals]))
    aceitos = [r for r in real if r[0] > T]
    colsA, colsB = list(Lg.columns), list(A.columns)
    print(f"\n  nulo bruto {bruto:.1f}; limiar {T:.1f}; falsas descobertas "
          f"esperadas {fd:.2f}; pares aceitos {len(aceitos):,}")

    def nome(c):
        return str(ann["name"].get(c, c))

    ia = {g: i for i, g in enumerate(colsA)}
    acc = {(colsA[x[1]], colsB[x[2]]) for x in aceitos}
    testaveis, passaram = [], []
    print("\n  CONTROLES")
    for gen, ds in CONTROLES.items():
        alvo = [c for c in colsB if any(d in nome(c).lower() for d in ds)]
        k = int(Lm[tr, ia[gen]].sum()) if gen in ia else 0
        if gen not in ia or k < S.MIN_ALT or not alvo:
            print(f"    {gen:13s} NAO TESTAVEL (genotipo {k}, remedios {len(alvo)})")
            continue
        testaveis.append(gen)
        hit = [nome(c) for c in alvo if (gen, c) in acc]
        if hit:
            passaram.append(gen)
        print(f"    {gen:13s} {k:3d} linhagens  "
              f"{('ACEITO: ' + ', '.join(hit[:3])) if hit else 'nao'}")

    ev = P.colapsa(aceitos, Lm, Xr, colsA, colsB)
    Pm, _ann_rep = secundaria_screen("MTS010")
    h = S.hipermutada(carga.loc[linhas], np.ones(n, bool))
    base_h = max(float(h.mean()), 1e-9)
    ctrl = set(CONTROLES)
    testados = replicou = 0
    cand = []
    print(f"\n  {len(aceitos):,} pares -> {len(ev)} eventos")
    print(f"  {'bits':>7s} {'genotipo':13s} {'remedio':24s} {'d rep':>6s} {'n':>4s}"
          f"  area / alvo")
    for e in ev:
        d, nn = replica_prim(e, Lg, Pm, lin)
        rep = d is not None and d < 0 and abs(d) >= 0.1
        if d is not None:
            testados += 1
            replicou += int(rep)
        ga, cb = e["rep"]
        nm = nome(cb)
        area = str(ann["disease.area"].get(cb, "")).strip()
        al = area.lower()
        alvo = str(ann["target"].get(cb, ""))
        gene = ga.replace("_hot", "").replace("_intacto", "")
        on_t = gene in [x.strip() for x in alvo.split(",")]
        onc = (not al) or al == "nan" or "oncology" in al or "malignan" in al
        ge = [g for g in e["A"] if g in Lg.columns]
        pres = Lg[ge].to_numpy().astype(bool).any(1)
        hyper = bool(pres.sum()) and (float(h[pres].mean()) / base_h) >= 3
        eh_c = bool(set(e["A"]) & ctrl)
        if rep and not eh_c and not on_t and not onc and not hyper:
            cand.append((e, nm, area, alvo, d, nn))
        marcas = (" ON-TARGET" if on_t else "") + (" HIPERMUT" if hyper else "") \
            + (" CONTROLE" if eh_c else "")
        dtxt = ("%+.2f" % d) if d is not None else "     -"
        print(f"  {e['bits']:7.1f} {ga:13s} {nm[:24]:24s} {dtxt:>6s} {nn:4d}  "
              f"{area[:16]} / {alvo[:20]}{marcas}")

    fr = len(passaram) / max(len(testaveis), 1)
    fq = replicou / max(testados, 1)
    print("")
    print("=" * 78)
    print(f"Q1 {'.' if len(testaveis) >= 4 and fr >= 0.6 else 'FALHOU .'} "
          f"controles {len(passaram)}/{len(testaveis)}")
    print(f"Q2 {'.' if fd <= 1 else 'FALHOU .'} falsas descobertas esperadas {fd:.2f}")
    print(f"Q3 {'.' if testados and fq >= 0.6 else 'FALHOU .'} replica em dose unica "
          f"{replicou}/{testados} ({100*fq:.0f}%)")
    print(f"Q4 {'.' if cand else 'FALHOU .'} candidatos: {len(cand)}")
    for e, nm, area, alvo, d, nn in cand[:15]:
        print(f"      {','.join(sorted(e['A']))[:28]:28s} -> {nm} ({area}; alvo "
              f"{alvo}) d_rep {d:+.2f} n={nn}")
    print("Q5 exige busca de literatura no nivel do evento")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

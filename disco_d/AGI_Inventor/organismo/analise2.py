"""
organismo.analise2 — portoes da vida 2 (escopo aprendido + valor da criacao)
=============================================================================
Pareado por semente. Passa se a media atinge o limiar E o IC bootstrap 90%
fica inteiro do lado certo do zero (exceto onde o pre-registro diz outro).
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu

import analise as A

FIXOS = ("LOCAL2", "LARGURA_3", "GLOBAL")


def m_extra(r):
    L = [l for l in r["linhas"] if l["fonte"] == "percebido"]
    hset = [l for l in L if (l["tipo"] == "L" and l["fam"] == "spin_orbita")
            or (l["tipo"] == "Q" and l["molde"] == "sentido")]
    m = A.metricas(r)
    m["h"] = float(np.mean([l["ok"] for l in hset])) if hset else float("nan")
    return m


def main(pasta):
    vidas = A.carrega(pasta)
    S = sorted({s for _, s in vidas})
    B = sorted({b for b, _ in vidas})
    rng = np.random.default_rng(0)
    print(f"{len(vidas)} vidas, sementes {S}\n")
    cols = ["total", "rec", "inv", "h", "mag", "cic", "lac", "custo_total"]
    print(f"{'braco':12s}" + "".join(f"{c:>12s}" for c in cols))
    med = {}
    for b in B:
        ms = [m_extra(vidas[(b, s)]) for s in S if (b, s) in vidas]
        med[b] = {c: float(np.nanmean([m[c] for m in ms])) for c in cols}
        print(f"{b:12s}" + "".join(f"{med[b][c]:12.3f}" for c in cols))

    def dif(a, b, chave):
        d = []
        for s in S:
            if (a, s) in vidas and (b, s) in vidas:
                x, y = m_extra(vidas[(a, s)])[chave], m_extra(vidas[(b, s)])[chave]
                if np.isfinite(x) and np.isfinite(y):
                    d.append(x - y)
        return np.array(d)

    def gate(nome, a, b, chave, lim, exige_ic=True):
        d = dif(a, b, chave)
        if not len(d):
            print(f"{nome} sem dados")
            return
        lo, hi = A.boot(d, rng)
        m = d.mean() * 100
        ok = m >= lim and (lo > 0 or not exige_ic)
        print(f"{nome:3s} {a} - {b} [{chave}]: {m:+.1f} (IC90 {lo*100:+.1f} a {hi*100:+.1f}), "
              f"a favor {int((d > 0).sum())}/{len(d)}, limiar {lim:+} -> {'PASSA' if ok else 'falha'}")

    print("\nPORTOES (PRE_REGISTRO_vida2.md)")
    gate("E1", "ESCOPO", "LOCAL2", "total", 3)
    melhor_fixo = max(FIXOS, key=lambda b: med.get(b, {}).get("total", -1))
    print(f"    melhor largura fixa pela media: {melhor_fixo}")
    gate("E2", "ESCOPO", melhor_fixo, "total", 2)
    gate("E3", "ESCOPO", "PIPELINE", "total", 3)
    gate("E4", "ESCOPO", "LOCAL2", "h", 10)
    gate("E5", "ESCOPO", "LOCAL2", "mag", -5, exige_ic=False)
    # E6: a geometria aprendida e real
    uma, varias = [], []
    for s in S:
        r = vidas.get(("ESCOPO", s))
        if not r:
            continue
        for e in r.get("escopos", []):
            if e["tipo"] in ("conceito", "direcao") and e["familias_pagou"]:
                (varias if len(e["familias_pagou"]) >= 2 else uma).append(e["beta_medio"])
    if uma and varias:
        p = mannwhitneyu(varias, uma, alternative="less").pvalue
        print(f"E6  largura aprendida: pagou em >=2 familias mediana {np.median(varias):.2f} (n={len(varias)}) "
              f"x 1 familia {np.median(uma):.2f} (n={len(uma)}); Mann-Whitney p={p:.4f} -> "
              f"{'PASSA' if p < 0.05 else 'falha'}")
    else:
        print(f"E6  sem estruturas suficientes (1 familia: {len(uma)}, >=2: {len(varias)}) -> falha")
    gate("E7", "LOCAL2", "ORGANISMO", "inv", 5)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados/vida2")

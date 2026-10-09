"""
organismo.posthoc — leituras DECLARADAS COMO POS-HOC (nao sao portoes)
=======================================================================
1. G7 restrito ao unico molde lacrado onde conceito pode ajudar (varre@40),
   conforme o adendo do pre-registro escrito antes de ler resultados.
2. Por braco: acerto por familia/molde, eventos de ontogenia (reativacoes,
   coalizoes, sonhos que validaram, credito atrasado), estruturas no fim.
"""
import collections
import glob
import json
import os
import sys

import numpy as np


def main(pasta):
    vidas = {}
    for arq in glob.glob(os.path.join(pasta, "*.json")):
        r = json.load(open(arq, encoding="utf-8"))
        vidas[(r["braco"], r["semente"])] = r
    bracos = sorted({b for b, _ in vidas})
    sementes = sorted({s for _, s in vidas})

    print("1. LACRADAS so em q:varre@40 (pos-hoc)")
    d = []
    for s in sementes:
        r = vidas.get(("ORGANISMO", s))
        if not r:
            continue
        ls = [l for l in r["linhas"] if l.get("molde") == "q:varre@40"]
        if ls:
            d.append(np.mean([l["ok"] for l in ls]) - np.mean([l["ok_ablacao"] for l in ls]))
    if d:
        print(f"   ORGANISMO com conceitos - mesma mente sem: {np.mean(d)*100:+.1f} pts "
              f"(a favor {sum(x > 0 for x in d)}/{len(d)}, contra {sum(x < 0 for x in d)})")
    for b in bracos:
        ok = [l["ok"] for s in sementes if (b, s) in vidas for l in vidas[(b, s)]["linhas"]
              if l.get("molde") == "q:varre@40"]
        print(f"   {b:14s} acerto em varre@40: {np.mean(ok):.2f} (n={len(ok)})")

    print("\n2. ACERTO POR FAMILIA (L) / MOLDE (Q), percebidos")
    chaves = set()
    tab = collections.defaultdict(list)
    for (b, s), r in vidas.items():
        for l in r["linhas"]:
            if l["fonte"] != "percebido":
                continue
            k = (l["tipo"], l["fam"] if l["tipo"] == "L" else l["molde"])
            chaves.add(k)
            tab[(b, k)].append(l["ok"])
    print(f"   {'':24s}" + "".join(f"{b[:9]:>10s}" for b in bracos))
    for k in sorted(chaves):
        print(f"   {k[0]} {k[1]:22s}" + "".join(f"{np.mean(tab[(b, k)]):10.2f}" for b in bracos))

    print("\n3. ONTOGENIA POR BRACO (media por vida)")
    evs = ("reativacao", "coalizao", "sonho", "credito_atrasado", "nascimento")
    print(f"   {'':14s}" + "".join(f"{e[:11]:>12s}" for e in evs) + f"{'sonho_ok':>10s}{'prof2':>7s}")
    for b in bracos:
        cont = collections.Counter()
        sv = 0
        n = 0
        p2 = 0
        for s in sementes:
            arq = os.path.join(pasta, f"{b}_s{s}.jsonl")
            if not os.path.exists(arq):
                continue
            n += 1
            for linha in open(arq, encoding="utf-8"):
                e = json.loads(linha)
                cont[e["ev"]] += 1
                if e["ev"] == "sonho" and e.get("valida"):
                    sv += 1
            p2 += vidas[(b, s)].get("prof_max", 0) >= 2
        if n:
            print(f"   {b:14s}" + "".join(f"{cont[e] / n:12.1f}" for e in evs) + f"{sv / n:10.1f}{p2:5d}/{n}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados/final")

"""
organismo.auditor — reconta a partir do LOG de eventos, nao do resumo da vida
===============================================================================
1. Para cada vida: o acerto por problema no JSONL ('problema') bate com o JSON?
2. O fluxo de problemas (familia, tipo, fonte, molde) e identico entre bracos
   na mesma semente?
3. Gasto: quantos problemas estouraram o orcamento, e de quanto?
4. ORGANISMO e PIPELINE no G1 recontados so pelo log.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

ORC = {"L": 300.0, "Q": 120.0}


def problemas_do_log(arq):
    out = []
    for linha in open(arq, encoding="utf-8"):
        d = json.loads(linha)
        if d["ev"] == "problema":
            out.append(d)
    return out


def main(pasta):
    erros = 0
    fluxos = collections.defaultdict(dict)
    estouros = collections.Counter()
    excesso = collections.defaultdict(list)
    acerto_log = collections.defaultdict(dict)
    for arq in sorted(glob.glob(os.path.join(pasta, "*.json"))):
        r = json.load(open(arq, encoding="utf-8"))
        b, s = r["braco"], r["semente"]
        pl = problemas_do_log(arq[:-5] + ".jsonl")
        if len(pl) != len(r["linhas"]):
            print("ERRO tamanho", b, s, len(pl), len(r["linhas"]))
            erros += 1
            continue
        for p, l in zip(pl, r["linhas"]):
            if p["idx"] != l["idx"] or bool(p["ok"]) != bool(l["ok"]) or p["fam"] != l["fam"]:
                print("ERRO linha", b, s, p["idx"])
                erros += 1
            if p["gasto"] > ORC[p["tipo"]] + 1e-6:
                estouros[b] += 1
                excesso[b].append(p["gasto"] - ORC[p["tipo"]])
        fluxos[s][b] = [(l["fam"], l["tipo"], l["fonte"], l["molde"]) for l in r["linhas"]]
        ver = [p for p in pl if p["fonte"] == "percebido"]
        acerto_log[b][s] = float(np.mean([p["ok"] for p in ver]))
    for s, por_b in fluxos.items():
        ref = next(iter(por_b.values()))
        for b, f in por_b.items():
            if f != ref:
                print("ERRO fluxo diferente", s, b)
                erros += 1
    print(f"vidas auditadas: {sum(len(v) for v in acerto_log.values())}; erros: {erros}")
    print("estouros de orcamento (problemas; excesso medio):")
    for b in sorted(acerto_log):
        ex = excesso.get(b, [])
        print(f"  {b:14s} {estouros.get(b, 0):4d}  {np.mean(ex) if ex else 0:7.1f}")
    if "ORGANISMO" in acerto_log and "PIPELINE" in acerto_log:
        ss = sorted(set(acerto_log["ORGANISMO"]) & set(acerto_log["PIPELINE"]))
        d = [acerto_log["ORGANISMO"][s] - acerto_log["PIPELINE"][s] for s in ss]
        print(f"G1 recontado pelo log: ORGANISMO - PIPELINE = {np.mean(d) * 100:+.1f} pts "
              f"em {len(d)} sementes; a favor {sum(x > 0 for x in d)}/{len(d)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados/final")

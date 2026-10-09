import json, numpy as np, vida, analise, collections
DEVL = ("periodo_longo", "rmax_grande", "vmax_alta")
ms = []
for s in (1, 2, 3):
    r = vida.vive("ORGANISMO", s, DEVL, "resultados/dev2")
    m = analise.metricas(r); ms.append(m)
    print(s, {k: round(v, 2) for k, v in m.items() if k not in ("lac", "lac_abl")}, f"{r['segundos']:.0f}s", flush=True)
print("media", {k: round(float(np.mean([m[k] for m in ms])), 2) for k in ms[0] if k not in ("lac", "lac_abl")})
# por familia/molde (so ORGANISMO)
c = collections.defaultdict(list)
for s in (1, 2, 3):
    r = json.load(open(f"resultados/dev2/ORGANISMO_s{s}.json"))
    for l in r["linhas"]:
        if l["fonte"] == "percebido" and not l.get("lacrado"):
            c[(l["tipo"], l["fam"] if l["tipo"] == "L" else l["molde"])].append(l["ok"])
for k in sorted(c): print(k, f"{np.mean(c[k]):.2f}", len(c[k]))
ev = collections.Counter()
for s in (1, 2, 3):
    for l in open(f"resultados/dev2/ORGANISMO_s{s}.jsonl", encoding="utf-8"):
        d = json.loads(l); ev[d["ev"]] += 1
        if d["ev"] == "fonte" and s == 1: last = d
print(dict(ev)); print("fonte (ultimo evento s1):", last)

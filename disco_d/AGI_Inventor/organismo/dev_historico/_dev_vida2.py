import json, shutil, sys, time, collections
import numpy as np
sys.path.insert(0, ".")
import vida
shutil.rmtree("resultados/dev_v2", ignore_errors=True)
DEVL = ("periodo_longo", "rmax_grande", "vmax_alta")
for b in ("LOCAL2", "ESCOPO"):
    for s in (1, 2, 3):
        t0 = time.time()
        r = vida.vive(b, s, DEVL, "resultados/dev_v2")
        c = collections.Counter()
        for l in open(f"resultados/dev_v2/{b}_s{s}.jsonl", encoding="utf-8"):
            d = json.loads(l)
            if d["ev"] == "problema":
                for a in d["acoes"]: c[a[0]] += 1
        print(f"{b:7s} s{s} {time.time()-t0:5.1f}s inventar {c['inventar']:3d} derivar {c['derivar']:3d} juntar {c['juntar']:4d} conceitos {r['tipos'].get('conceito',0)}", flush=True)
        if b == "ESCOPO":
            es = [e for e in r["escopos"] if e["tipo"] in ("conceito", "direcao") and e["familias_pagou"]]
            for e in sorted(es, key=lambda e: e["beta_medio"])[:6]:
                print(f"      {e['id']:13s} n={e['n']:3d} largura {e['beta_medio']:5.2f} global {e['p_global']:.2f} pagou {e['familias_pagou']} {str(e['desc'])[:60]}")

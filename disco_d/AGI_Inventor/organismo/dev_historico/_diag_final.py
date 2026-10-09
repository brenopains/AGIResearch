import json, glob, collections, numpy as np, sys
sys.path.insert(0, ".")
import analise
P = "resultados/final"
V = analise.carrega(P); S = sorted({s for _, s in V})
rng = np.random.default_rng(1)
# 1) G4: IC bootstrap da razao de custos (por semente reamostrada)
co = np.array([analise.metricas(V[("ORGANISMO", s)])["custo_t3"] for s in S])
cc = np.array([analise.metricas(V[("SEM_COAL", s)])["custo_t3"] for s in S])
idx = rng.integers(0, len(S), (4000, len(S)))
rat = co[idx].mean(1) / cc[idx].mean(1)
print(f"G4 razao {co.mean()/cc.mean():.2f}  IC90 {np.quantile(rat,.05):.2f} a {np.quantile(rat,.95):.2f}; razao por semente: {np.round(co/cc,2)}")
# 2) G8 (medida do portao) em todos os bracos
for b in sorted({b for b, _ in V}):
    p2 = [analise.profundidade2(P, b, s) for s in S]
    print(f"G8 medida {b:13s} {sum(p2)}/10")
# 3) TETOS: acoes sobre estruturas ja dissolvidas? estouros por operacao
ev = collections.Counter(); mot = collections.Counter()
for s in S:
    for l in open(f"{P}/TETOS_s{s}.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["ev"] == "dissolucao": mot[(d["tipo"], d["motivo"])] += 1
        if d["ev"] == "problema" and d["gasto"] > (300 if d["tipo"] == "L" else 120):
            for a in d["acoes"][-1:]: ev[a[0]] += 1
print("TETOS dissolucoes:", dict(mot)); print("TETOS ultima acao nos estouros:", dict(ev))
# 4) h no vocabulario: nos 'sentido' do ORGANISMO, quantos acertos usaram conceito, e falhas
for b in ("ORGANISMO", "PIPELINE"):
    ok_c = ok_s = f = 0
    for s in S:
        for l in V[(b, s)]["linhas"]:
            if l["molde"] == "sentido":
                if l["ok"] and l["usados"]: ok_c += 1
                elif l["ok"]: ok_s += 1
                else: f += 1
    print(f"sentido {b}: acerto com conceito {ok_c}, acerto sem {ok_s}, falha {f}")

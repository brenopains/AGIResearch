"""
pure_neural_ms.py — TETO da CRIACAO AUTONOMA: NCA multi-seed (3 seeds + voto d8), SEM simbolo e SEM nenhuma
primitiva minha, em TODAS as 262 tarefas same-shape. Mata o ruido de seed unica e ensembla -> numero robusto
de quantas regras a maquina cria sozinha. Reusa unified_nca3.solve_nca3 (puro neural).
"""
import sys, json, time, pathlib, numpy as np
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified_nca3 as N
from solver import ARC_DIR

tasks = sorted(p.name[:-5] for p in ARC_DIR.glob("*.json"))
same = []
for tid in tasks:
    t = json.load(open(ARC_DIR / (tid + ".json")))
    d = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
    te = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
    if d and all(i.shape == o.shape for i, o in d + te): same.append(tid)
print(f"CRIACAO AUTONOMA multi-seed (sem simbolo/primitivas) em {len(same)} same-shape", flush=True)
sc = 0; got = []; t0 = time.time()
for i, tid in enumerate(same):
    t = json.load(open(ARC_DIR / (tid + ".json")))
    demos = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
    tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
    try: outs = N.solve_nca3(demos, tests, seeds=(0, 1, 2), T=16, steps=300)
    except Exception: outs = None
    if outs is not None and all(pr.shape == yo.shape and np.array_equal(pr, yo) for pr, (_, yo) in zip(outs, tests)):
        sc += 1; got.append(tid)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(same)}  criadas {sc} ({sc/(i+1):.1%})  ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"n": i + 1, "solved": sc, "got": got}, open(HERE / "pure_neural_ms_progress.json", "w"))
print(f"\n=== TETO DA CRIACAO AUTONOMA (multi-seed, sem simbolo): {sc}/{len(same)} same-shape = {sc/len(same):.1%} ===", flush=True)
print(f"  ({sc}/400 do total = {sc/400:.1%})  regras criadas: {got}", flush=True)
json.dump({"solved": sc, "same_total": len(same), "got": got}, open(HERE / "pure_neural_ms_result.json", "w"))

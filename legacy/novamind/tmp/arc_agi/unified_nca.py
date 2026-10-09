"""
unified_nca.py — UNIFICACAO: minha base simbolica forte (unified.solve_unified) + NCA per-puzzle (exp_h70_nca)
como FASE 1.5 nas tarefas same-shape que a base nao resolve. NCA capta regras LOCAIS ITERATIVAS (propagar/crescer/
preencher/contorno) que o simbolico nao pega. GPU, sem LLM. Mede o delta real da uniao base+NCA vs base so.
"""
import sys, json, time, pathlib, numpy as np
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))   # arc_agi + tmp
import unified as U
import exp_h70_nca as NCA
from solver import ARC_DIR

tasks = sorted(ARC_DIR.glob("*.json"))
sc = 0; by_sym = 0; by_nca = 0; n = 0; t0 = time.time(); nca_tasks = []
print(f"UNIFIED+NCA: base simbolica (FASE 1) + NCA per-puzzle (FASE 1.5) | {len(tasks)} tasks", flush=True)
for p in tasks:
    task = json.load(open(p)); task["_id"] = p.name; n += 1
    prog = None
    try: prog = U.solve_unified(task)
    except Exception: prog = None
    if prog:
        by_sym += 1; sc += 1
    else:
        train = [(np.array(x["input"]), np.array(x["output"])) for x in task["train"]]
        tests = [(np.array(x["input"]), np.array(x["output"])) for x in task["test"]]
        if train and all(i.shape == o.shape for i, o in train) and all(i.shape == o.shape for i, o in tests):
            try:
                outs = NCA.solve_nca(train, tests)
            except Exception:
                outs = None
            if outs is not None and all(pr.shape == yo.shape and np.array_equal(pr, yo) for pr, (_, yo) in zip(outs, tests)):
                by_nca += 1; sc += 1; nca_tasks.append(p.name[:-5])
    if n % 50 == 0:
        print(f"  {n}/{len(tasks)}  total {sc} (sym {by_sym} + NCA {by_nca})  {sc/n:.1%}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"n": n, "solved": sc, "by_sym": by_sym, "by_nca": by_nca, "nca_tasks": nca_tasks},
                  open(HERE / "unified_nca_progress.json", "w"))
print(f"\nUNIFIED+NCA: {sc}/{n} = {sc/n:.1%}  (simbolico {by_sym} + NCA {by_nca})", flush=True)
print(f"base simbolica so = {by_sym}/{n} = {by_sym/n:.1%}  ->  + NCA = {sc/n:.1%}  (NCA novas: {by_nca})", flush=True)
print(f"NCA resolveu: {nca_tasks[:20]}", flush=True)

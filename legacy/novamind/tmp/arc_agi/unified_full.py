"""
unified_full.py — UNIFICACAO DE VERDADE: minha base forte (unified.solve_unified, 8.5% sem LLM) como FASE 1,
e o LLM-minting do v3_bridge (llm_growth) como FASE 2 no fracasso. Mede o lift combinado real numa fatia.
(O 25% historico = base fraca 5.2% + LLM + NCA; aqui troco a base fraca pela forte.)
"""
import sys, json, time, pathlib, os
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified as U
import v3_bridge as V
from solver import ARC_DIR, test_correct

SLICE = int(sys.argv[1]) if len(sys.argv) > 1 else 80
GENS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
quiet = lambda s: None

lib = V.Library(); lib.reset()
try: lib.seed()
except Exception: pass
tasks = sorted(ARC_DIR.glob("*.json"))[:SLICE]
sc = 0; by_uni = 0; by_llm = 0; n = 0; t0 = time.time()
print(f"UNIFIED-FULL: base forte (FASE 1) + LLM-minting (FASE 2) | {len(tasks)} tasks, gens={GENS}", flush=True)
for p in tasks:
    task = json.load(open(p)); task["_id"] = p.name; n += 1
    prog = None
    try: prog = U.solve_unified(task)            # FASE 1 (sem LLM, ja faz gate no teste)
    except Exception: prog = None
    if prog:
        by_uni += 1; sc += 1
    else:
        try:
            preds, calls, prim = V.llm_growth(task, lib, GENS, 1, quiet)   # FASE 2 (LLM)
            if preds is not None and test_correct(preds, task):
                by_llm += 1; sc += 1
        except Exception:
            pass
        try: lib.save()
        except Exception: pass
    if n % 10 == 0:
        print(f"  {n}/{len(tasks)}  total {sc} (base {by_uni} + LLM {by_llm})  {sc/n:.1%}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"n":n,"total":len(tasks),"solved":sc,"by_base":by_uni,"by_llm":by_llm,"rate":sc/n},
                  open(HERE/"unified_full_progress.json","w"))
print(f"\nUNIFIED-FULL: {sc}/{n} = {sc/n:.1%}  (base sem-LLM {by_uni} + LLM-minting {by_llm})", flush=True)
print(f"comparacao: minha base so = {by_uni}/{n} = {by_uni/n:.1%}  ->  + LLM = {sc/n:.1%}", flush=True)

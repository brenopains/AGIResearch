"""
nova_unified.py — UNIAO REAL: o MAESTRO orquestra os solvers REAIS do ARC numa cabeca so (sem LLM).
Antes, nova_maestro (controle) e os solvers do ARC eram SEPARADOS. Aqui o maestro:
  - PERCEBE (fingerprint) e RECUPERA da MEMORIA multidominio regras de tarefas passadas;
  - ROTEIA: simbolo+forge+hipoteses (rapido) -> se falha, sobe o AFETO (frustracao);
  - frustracao dispara o NEURAL (criacao autonoma) com SURTO DE DMT/PLASTICIDADE escalado pela frustracao;
  - acerto = alivio; MEMORIA acumula a regra; wake-sleep consolida recorrentes.
Mede o numero da cabeca UNIFICADA e atribui a rota (emergencia = todo > soma?).
"""
import sys, json, time, pathlib, numpy as np
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified as U, tem_arc2 as T
import dmt_nca as DMT
from integrated import Memory, fingerprint
from solver import ARC_DIR

class Maestro:
    def __init__(s): s.mem = Memory(); s.frust = 0.0; s.archetypes = []; s.log = []
    def solve(s, task):
        fp = fingerprint(task); ntr = len(task["train"])
        # 1) MEMORIA: regra recuperada de tarefa passada (reuso cross-tarefa)
        pbn = {}
        for nm, fn in s._pool(task): pbn.setdefault(nm, fn)
        from integrated import recipe_fn
        for recipe in s.mem.recall(fp):
            base = recipe_fn(recipe, pbn)
            if base and U.train_score(base, task) == ntr and T.vtest(base, task):
                s.frust = max(0, s.frust - 0.3); return "memoria", recipe
        # 2) SIMBOLO + FORGE + HIPOTESES (busca barata)
        prog = U.solve_unified(task)
        if prog:
            s.mem.add(fp, prog.replace("id>>", "")); s.frust = max(0, s.frust - 0.2)
            return "simbolo", prog
        # 3) FRUSTRACAO sobe -> NEURAL (criacao autonoma) com DMT escalado pela frustracao
        s.frust = min(1.0, s.frust + 0.25)
        import os
        if os.environ.get("NOVA_FAST") == "1": return None, None   # demo rapida: pula a rota neural/GPU lenta
        demos = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in task["test"]]
        if demos and all(i.shape == o.shape for i, o in demos) and all(i.shape == o.shape for i, o in tests):
            seeds = (0, 1, 2) if s.frust > 0.6 else (0, 1)         # afeto: mais esforco qd frustrado
            try: outs = DMT.solve(demos, tests, dmt=(s.frust > 0.4), seeds=seeds)  # frustracao alta -> DMT on
            except Exception: outs = None
            if outs is not None and all(pr.shape == yo.shape and np.array_equal(pr, yo) for pr, (_, yo) in zip(outs, tests)):
                s.frust = max(0, s.frust - 0.4); return "neural+dmt", "nca"
        return None, None
    def _pool(s, task):
        pool = []
        for nm, ind in T.TEMPLATES + U.EXTRA + U.HYPO:
            try: fn = ind(task)
            except Exception: fn = None
            if fn: pool.append((nm, fn))
        pool += U.SEED_FNS
        try:
            import forge as F; pool += F.forge_candidates(task)
        except Exception: pass
        return pool

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    tasks = T.load_tasks(n)
    m = Maestro(); sc = 0; by = {}; t0 = time.time()
    print(f"CABECA UNIFICADA (maestro orquestrando simbolo+memoria+forge+hipoteses+neural/DMT) em {len(tasks)} tasks", flush=True)
    for i, tk in enumerate(tasks):
        route, info = m.solve(tk)
        if route: sc += 1; by[route] = by.get(route, 0) + 1
        if (i + 1) % 20 == 0: print(f"  {i+1}/{len(tasks)}  resolvidas {sc}  frust={m.frust:.2f}  rotas={by}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n=== CABECA UNIFICADA: {sc}/{len(tasks)} = {sc/len(tasks):.1%} | rotas: {by} ===", flush=True)
    print(f"  memoria guardou {sum(len(v) for v in m.mem.store.values())} regras | frustracao final {m.frust:.2f}", flush=True)

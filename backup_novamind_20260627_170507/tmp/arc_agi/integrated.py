"""
integrated.py — SOLVER INTEGRADO (sem LLM): junta TUDO que provamos numa cabeca so.
  PERCEPCAO -> MEMORIA (recupera regras de tarefas passadas por similaridade) -> SIMBOLO+FORGE+HIPOTESES (busca)
  -> e a memoria ACUMULA cada vitoria. A novidade real: a MEMORIA MULTIDOMINIO — regra descoberta na tarefa A e
  RECUPERADA e tentada (ate composta com +1 op = mais profundo) na tarefa B similar. Generalizacao entre tarefas.
Mede: total integrado, quantas vieram da MEMORIA, e se a memoria resolve algo que a busca fria NAO resolve
(=ganho real de integracao). NCA/world-model (GPU) entram pela uniao (TTT-NCA rodando em paralelo).
"""
import sys, json, time, pathlib
from collections import Counter
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified as U, tem_arc2 as T

def fingerprint(task):
    """assinatura perceptiva da tarefa (p/ recuperar tarefas similares da memoria)."""
    p = task["train"][0]; gi, go = p["input"], p["output"]
    Hi, Wi = T.dims(gi); Ho, Wo = T.dims(go)
    same = (Hi, Wi) == (Ho, Wo)
    ncol = len(set(v for r in gi for v in r))
    bg = T.bg_color(gi); nobj = len(T.components(gi, bg))
    rr = round((Ho / Hi) if Hi else 0, 1); rc = round((Wo / Wi) if Wi else 0, 1)
    return (same, ncol, min(nobj, 6), rr, rc)

class Memory:
    """MEMORIA MULTIDOMINIO: fingerprint -> receitas vitoriosas. Recupera por similaridade entre tarefas."""
    def __init__(self): self.store = {}
    def add(self, fp, recipe):
        self.store.setdefault(fp, [])
        if recipe not in self.store[fp]: self.store[fp].append(recipe)
    def recall(self, fp):
        out = list(self.store.get(fp, []))
        for k, recs in self.store.items():                 # similares: mesma forma+ratio (cor/objetos podem diferir)
            if k != fp and k[0] == fp[0] and k[3] == fp[3] and k[4] == fp[4]:
                out += recs
        seen = set(); uniq = []
        for r in out:
            if r not in seen: seen.add(r); uniq.append(r)
        return uniq[:24]

def recipe_fn(recipe, pool_by_name):
    """reconstroi a fn de uma receita (ex 'flip_v>>mirror_down') a partir do pool DESTA tarefa."""
    fns = []
    for nm in recipe.split(">>"):
        if nm not in pool_by_name: return None
        fns.append(pool_by_name[nm])
    def fn(g):
        for f in fns: g = f(g)
        return g
    return fn

def solve_integrated(task, mem, use_memory=True):
    ntr = len(task["train"]); fp = fingerprint(task)
    pool = U.build_pool_named(task) if hasattr(U, "build_pool_named") else None
    # pool nomeado: reusa o pool do unified
    pool = []
    for nm, ind in T.TEMPLATES + U.EXTRA + U.HYPO:
        try: fn = ind(task)
        except Exception: fn = None
        if fn: pool.append((nm, fn))
    pool += U.SEED_FNS
    try:
        import forge as FORGE; pool += FORGE.forge_candidates(task)
    except Exception: pass
    pbn = {}
    for nm, fn in pool:
        pbn.setdefault(nm, fn)
    # 1) MEMORIA: tenta receitas recuperadas (e compoe com +1 op = mais profundo que a busca alcanca)
    if use_memory:
        for recipe in mem.recall(fp):
            base = recipe_fn(recipe, pbn)
            if base is None: continue
            if U.train_score(base, task) == ntr and T.vtest(base, task):
                return ("memoria", recipe)
            for nm, fn in pool:                            # receita_lembrada >> op  (profundidade via memoria)
                comp = (lambda g, b=base, f=fn: f(b(g)))
                if U.train_score(comp, task) == ntr and T.vtest(comp, task):
                    return ("memoria+", recipe + ">>" + nm)
    # 2) BUSCA (simbolo+forge+hipoteses, beam) — reusa solve_unified
    prog = U.solve_unified(task)
    if prog:
        mem.add(fp, prog.replace("id>>", ""))
        return ("busca", prog)
    return (None, None)

if __name__ == "__main__":
    tasks = U.T.load_tasks()
    print(f"INTEGRADO (memoria multidominio + simbolo+forge+hipoteses): {len(tasks)} tasks", flush=True)
    # passada COM memoria (acumula e recupera entre tarefas)
    mem = Memory(); sc = 0; by = Counter(); t0 = time.time(); solved = []
    for i, tk in enumerate(tasks):
        route, prog = solve_integrated(tk, mem, use_memory=True)
        if route: sc += 1; by[route] += 1; solved.append(tk["_id"])
        if (i + 1) % 100 == 0: print(f"  {i+1}/400  resolvidas {sc}  ({time.time()-t0:.0f}s)", flush=True)
    # passada SEM memoria (busca fria) p/ medir o GANHO da memoria
    sc_cold = 0; cold = set()
    for tk in tasks:
        route, prog = solve_integrated(tk, Memory(), use_memory=False)
        if route: sc_cold += 1; cold.add(tk["_id"])
    extra = [s for s in solved if s not in cold]
    print(f"\nINTEGRADO (com memoria): {sc}/400 = {sc/400:.1%}", flush=True)
    print(f"  rotas: {dict(by)}", flush=True)
    print(f"BUSCA FRIA (sem memoria): {sc_cold}/400 = {sc_cold/400:.1%}", flush=True)
    print(f"GANHO DA MEMORIA (resolvido SO via memoria, busca fria falhou): {len(extra)} -> {extra[:12]}", flush=True)
    json.dump({"solved": solved, "by": dict(by), "cold": sorted(cold), "memory_extra": extra},
              open(HERE / "integrated_result.json", "w"))

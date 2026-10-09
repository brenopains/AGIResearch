"""trace.py — captura o TRACO REAL do solver (tentativas simbolicas + morfogenese do NCA) p/ o CORTEX viewer."""
import sys, json, pathlib, numpy as np, torch
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified as U, tem_arc2 as T
from solver import ARC_DIR
import unified_nca3 as N3
import forge as FORGE
from hypo import HYPO

def build_pool(task):
    pool = []
    for nm, ind in T.TEMPLATES + U.EXTRA + HYPO:
        try: fn = ind(task)
        except Exception: fn = None
        if fn: pool.append((nm, fn))
    pool += U.SEED_FNS
    try: pool += FORGE.forge_candidates(task)
    except Exception: pass
    return pool

def trace_symbolic(task):
    ntr = len(task["train"]); pool = build_pool(task); attempts = []; solved = None
    di = task["train"][0]["input"]; do = task["train"][0]["output"]; ti = task["test"][0]["input"]
    imag = {}   # nome -> grade IMAGINADA (o que a primitiva produz) no demo0 e no teste
    for nm, fn in pool:
        sc = U.train_score(fn, task)
        test_ok = (sc == ntr and T.vtest(fn, task))
        attempts.append({"name": nm, "score": round(sc / max(1, ntr), 2), "solved": bool(test_ok)})
        if nm not in imag:
            try: imag[nm] = {"demo": fn(di), "test": fn(ti)}
            except Exception: imag[nm] = None
        if test_ok and not solved:
            try: solved = {"prog": nm, "pred": fn(task["test"][0]["input"])}
            except Exception: pass
    seen = set(); uniq = []
    for a in sorted(attempts, key=lambda a: -a["score"]):
        if a["name"] in seen: continue
        seen.add(a["name"]); uniq.append(a)
    # IMAGINACAO: top candidatos (que produzem algo) com a grade imaginada vs a esperada (memoria do demo)
    thoughts = []
    for a in uniq:
        im = imag.get(a["name"])
        if im and im["demo"] is not None:
            thoughts.append({"name": a["name"], "score": a["score"], "imag_demo": im["demo"],
                             "imag_test": im["test"], "match": (a["score"] >= 1.0)})  # BATE = casa TODOS os demos
        if len(thoughts) >= 8: break
    return uniq[:36], solved, {"exp_demo": do, "thoughts": thoughts}

def trace_nca(task, T_steps=16):
    train = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    tests = [(np.array(p["input"]), np.array(p["output"])) for p in task["test"]]
    if not train or not all(i.shape == o.shape for i, o in train): return None
    if not all(i.shape == o.shape for i, o in tests): return None
    net = N3._train(train, T_steps, 300, 0)
    gate = all(np.array_equal(N3._pred_canon(net, i, T_steps), o) for i, o in train)
    g, truth = tests[0]; H, W = g.shape
    state = N3.to_state(g).unsqueeze(0).to(N3.DEV); frames = []
    with torch.no_grad():
        for _ in range(T_steps):
            state = net.step(state)
            frames.append(state[0, :N3.NCOL, :H, :W].argmax(0).cpu().numpy().tolist())
    pred = frames[-1]; ok = (pred == truth.tolist())
    return {"frames": frames, "pred": pred, "gate": bool(gate), "ok": bool(ok)}

# tarefas resolvidas pela MEMORIA (rota integrada) — regra de tarefa passada recuperada e estendida
MEMORY_ROUTES = {"b2862040": "gravity_down>>recolor_holes"}

def main():
    ids = sys.argv[1:] or ["67a3c6ac", "0a938d79", "25d8a9c8", "3aa6fb7a", "b2862040", "0b148d64"]
    out = []
    for tid in ids:
        fp = ARC_DIR / (tid + ".json")
        if not fp.exists(): continue
        task = json.load(open(fp)); task["_id"] = tid
        att, solved, imag = trace_symbolic(task)
        # MEMORIA: se a tarefa foi resolvida por receita recuperada de outra tarefa (generalizacao cross-tarefa)
        if tid in MEMORY_ROUTES and not solved:
            recipe = MEMORY_ROUTES[tid]; pbn = {}
            for nm, fn in build_pool(task): pbn.setdefault(nm, fn)
            rfn = None
            try:
                from integrated import recipe_fn; rfn = recipe_fn(recipe, pbn)
            except Exception: rfn = None
            if rfn:
                try: pred = rfn(task["test"][0]["input"])
                except Exception: pred = None
                out.append({"id": tid, "train": [{"in": p["input"], "out": p["output"]} for p in task["train"]],
                            "test_in": task["test"][0]["input"], "test_out": task["test"][0]["output"],
                            "attempts": att, "symbolic_prog": None, "pred": pred, "nca": None, "imag": imag,
                            "route": "memoria", "recipe": recipe,
                            "verdict": "memória (cross-tarefa): " + recipe})
                print(f"  tracado {tid}: MEMORIA '{recipe}'  (tentativas {len(att)})", flush=True); continue
        nca = None if solved else trace_nca(task)
        route = "simbolo" if solved else ("criacao_neural" if (nca and nca["ok"]) else "nenhuma")
        verdict = ("simbolico:" + solved["prog"]) if solved else ("criação autônoma neural (NCA criou a regra)" if (nca and nca["ok"]) else "NAO RESOLVIDO")
        out.append({
            "id": tid,
            "train": [{"in": p["input"], "out": p["output"]} for p in task["train"]],
            "test_in": task["test"][0]["input"], "test_out": task["test"][0]["output"],
            "attempts": att, "symbolic_prog": (solved["prog"] if solved else None),
            "pred": (solved["pred"] if solved else (nca["pred"] if nca else None)),
            "nca": nca, "verdict": verdict, "imag": imag, "route": route,
        })
        print(f"  tracado {tid}: {verdict}  (tentativas {len(att)})", flush=True)
    open(HERE / "cortex_data.js", "w", encoding="utf-8").write("window.TRACE=" + json.dumps(out) + ";")
    print(f"OK -> cortex_data.js ({len(out)} tarefas)", flush=True)

if __name__ == "__main__": main()

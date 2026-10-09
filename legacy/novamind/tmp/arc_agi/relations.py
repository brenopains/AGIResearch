"""
relations.py — A RELACAO-GRAFO DA ADA: a maquina deixa de operar sobre pixels e passa a operar sobre RELACOES
entre objetos (a Maquina Analitica "tece padroes algebricos"). Constroi um grafo: contem / esta-dentro /
vizinhanca / forma-unica / alinhamento, e induz a regra como FUNCAO de uma RELACAO (nao so de uma propriedade).
Ex.: "recolore o objeto que CONTEM outro", "apaga os que sao ADJACENTES ao maior", "mantem a forma UNICA".
Princip. MDL: prefere a relacao mais simples que explica TODOS os demos. Sem LLM, sem primitiva minha p/ tarefa.
"""
import tem_arc2 as T
from collections import Counter

def _objs(g):
    bg = T.bg_color(g); os = T.components(g, bg)
    for o in os: o["holes"] = T.holes(g, o, bg)
    return bg, os

def _bbox_inside(a, b):   # bbox de a dentro do bbox de b (a != b)
    return a["top"] >= b["top"] and a["left"] >= b["left"] and a["bot"] <= b["bot"] and a["right"] <= b["right"] and a["size"] < b["size"]

def _touch(a, b):
    sa = set((y, x) for y, x in a["cells"])
    for y, x in b["cells"]:
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            if (y+dy, x+dx) in sa: return True
    return False

def _rel_keys(objs):
    """propriedades RELACIONAIS de cada objeto (o grafo)."""
    n = len(objs)
    mx = max((o["size"] for o in objs), default=0)
    largest = [o for o in objs if o["size"] == mx]
    shapes = Counter(T.shape_sig(o) for o in objs)
    for a in objs:
        a["_contem"] = sum(1 for b in objs if b is not a and _bbox_inside(b, a))
        a["_dentro"] = sum(1 for b in objs if b is not a and _bbox_inside(a, b))
        a["_vizinhos"] = sum(1 for b in objs if b is not a and _touch(a, b))
        a["_toca_maior"] = 1 if any(b is not a and _touch(a, b) for b in largest) else 0
        a["_forma_unica"] = 1 if shapes[T.shape_sig(a)] == 1 else 0
    return {
        "contem_outro":   lambda o: 1 if o["_contem"] > 0 else 0,
        "esta_dentro":    lambda o: 1 if o["_dentro"] > 0 else 0,
        "n_vizinhos":     lambda o: o["_vizinhos"],
        "isolado":        lambda o: 1 if o["_vizinhos"] == 0 else 0,
        "toca_o_maior":   lambda o: o["_toca_maior"],
        "forma_unica":    lambda o: o["_forma_unica"],
    }

def _out_color(o, go):
    cols = set(go[y][x] for (y, x) in o["cells"])
    return next(iter(cols)) if len(cols) == 1 else None

def induce_recolor_rel(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    keynames = None; learned = {}
    for p in task["train"]:
        gi, go = p["input"], p["output"]; bg, objs = _objs(gi)
        if len(objs) < 2: return None
        kf = _rel_keys(objs)
        if keynames is None: keynames = list(kf.keys()); learned = {k: {} for k in keynames}
        for o in objs:
            oc = _out_color(o, go)
            if oc is None: return None
            for kn in keynames:
                kv = kf[kn](o); m = learned[kn]
                if kv in m and m[kv] != oc: m["__bad__"] = True
                else: m[kv] = oc
    for kn in keynames:
        m = learned[kn]
        if m.get("__bad__"): continue
        mp = {k: v for k, v in m.items() if k != "__bad__"}
        if len(set(mp.values())) < 2: continue
        def fn(g, kn=kn, mp=mp):
            bg, objs = _objs(g); out = T.gcopy(g)
            if len(objs) < 1: return out
            kf = _rel_keys(objs)
            for o in objs:
                kv = kf[kn](o)
                if kv in mp:
                    for y, x in o["cells"]: out[y][x] = mp[kv]
            return out
        return ("recolor_rel[" + kn + "]", fn)
    return None

def induce_select_rel(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    keynames = None; survive = {}
    for p in task["train"]:
        gi, go = p["input"], p["output"]; bg, objs = _objs(gi)
        if len(objs) < 2: return None
        kf = _rel_keys(objs)
        if keynames is None: keynames = list(kf.keys()); survive = {k: {} for k in keynames}
        for o in objs:
            outcols = set(go[y][x] for (y, x) in o["cells"])
            kept = (outcols == {o["color"]}); gone = (outcols == {bg})
            if not (kept or gone): return None
            for kn in keynames:
                kv = kf[kn](o); m = survive[kn]
                if kv in m and m[kv] != kept: m["__bad__"] = True
                else: m[kv] = kept
    for kn in keynames:
        m = survive[kn]
        if m.get("__bad__"): continue
        vals = {k: v for k, v in m.items() if k != "__bad__"}
        if len(set(vals.values())) < 2: continue
        def fn(g, kn=kn, vals=vals):
            bg, objs = _objs(g); H, W = T.dims(g); out = [[bg] * W for _ in range(H)]
            kf = _rel_keys(objs)
            for o in objs:
                if vals.get(kf[kn](o), False):
                    for y, x in o["cells"]: out[y][x] = o["color"]
            return out
        return ("manter_rel[" + kn + "]", fn)
    return None

INDUCERS = [induce_recolor_rel, induce_select_rel]

def relation_candidates(task):
    out = []
    for ind in INDUCERS:
        try: r = ind(task)
        except Exception: r = None
        if r: out.append(r)
    return out

if __name__ == "__main__":
    import json, pathlib, time
    from solver import ARC_DIR
    base = set(json.load(open("unified_solved.json"))) if pathlib.Path("unified_solved.json").exists() else set()
    ids = sorted(p.name[:-5] for p in ARC_DIR.glob("*.json"))
    sc = 0; novas = []; t0 = time.time()
    for tid in ids:
        t = json.load(open(ARC_DIR / (tid + ".json"))); t.setdefault("_id", tid)
        ok = any(T.vtrain(fn, t) and T.vtest(fn, t) for nm, fn in relation_candidates(t))
        if ok:
            sc += 1
            if tid not in base: novas.append(tid)
    print(f"RELACAO-GRAFO (Ada): {sc}/400 | NOVAS (alem do simbolico): {len(novas)} {novas}  ({time.time()-t0:.0f}s)")

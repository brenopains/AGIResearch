"""
relational.py — RACIOCINIO RELACIONAL / structure-mapping (Gentner; LISA) — abordagem NAO-OBVIA nunca testada.
Raciocina sobre OBJETOS + RELACOES + PAPEIS, nao pixels: seleciona o objeto pelo seu PAPEL relacional
(o unico/odd-one-out, o que contem buracos, a ordem por posicao) e age. A regra transfere entre objetos por
construcao (binding de papel). Mede ortogonalidade ao simbolico (acende tarefas object-centric que ele nao pega).
"""
import tem_arc2 as T
from collections import Counter

def _objs(g):
    bg = T.bg_color(g); os = T.components(g, bg)
    for o in os: o["sig"] = T.shape_sig(o); o["holes"] = T.holes(g, o, bg)
    return bg, os

def t_odd_one_out(task, mode):
    """seleciona o objeto cujo PAPEL e unico (shape OU cor unica) e: keep (so ele) / remove (some)."""
    def fn(g):
        bg, os = _objs(g); H, W = T.dims(g)
        if len(os) < 3: return T.gcopy(g)
        bys = Counter(o["sig"] for o in os); byc = Counter(o["color"] for o in os)
        odd = None
        uniq_s = [o for o in os if bys[o["sig"]] == 1]
        uniq_c = [o for o in os if byc[o["color"]] == 1]
        if len(uniq_s) == 1: odd = uniq_s[0]
        elif len(uniq_c) == 1: odd = uniq_c[0]
        if odd is None: return T.gcopy(g)
        if mode == "keep":
            out = [[bg] * W for _ in range(H)]
            for y, x in odd["cells"]: out[y][x] = odd["color"]
            return out
        out = T.gcopy(g)
        for y, x in odd["cells"]: out[y][x] = bg
        return out
    return fn

def t_recolor_by_pos_rank(task):
    """ordena objetos por posicao (cima->baixo, esq->dir) e recolore cada um por seu RANK (mapa aprendido)."""
    mp = {}
    for p in task["train"]:
        gi, go = p["input"], p["output"]
        if T.dims(gi) != T.dims(go): return None
        bg, oi = _objs(gi)
        order = sorted(range(len(oi)), key=lambda k: (oi[k]["top"], oi[k]["left"]))
        for rk, idx in enumerate(order):
            cols = {go[y][x] for y, x in oi[idx]["cells"]}
            if len(cols) != 1: return None
            oc = next(iter(cols))
            if rk in mp and mp[rk] != oc: return None
            mp[rk] = oc
    if not mp: return None
    def fn(g):
        bg, o = _objs(g); out = T.gcopy(g)
        order = sorted(range(len(o)), key=lambda k: (o[k]["top"], o[k]["left"]))
        for rk, idx in enumerate(order):
            if mp.get(rk) is not None:
                for y, x in o[idx]["cells"]: out[y][x] = mp[rk]
        return out
    return fn

def t_whole_object_gravity(task, d):
    """move cada OBJETO inteiro (nao celula) ate a borda d (down/up/left/right), empilhando — relacao espacial."""
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    def fn(g):
        bg, os = _objs(g); H, W = T.dims(g); out = [[bg] * W for _ in range(H)]
        key = {"down": lambda o: -o["bot"], "up": lambda o: o["top"],
               "left": lambda o: o["left"], "right": lambda o: -o["right"]}[d]
        for o in sorted(os, key=key):
            dy = dx = 0
            if d in ("down", "up"):
                while True:
                    ny = [(y + dy + (1 if d == "down" else -1)) for y, x in o["cells"]]
                    if any(v < 0 or v >= H for v in ny) or any(out[v][x] != bg for v, (y, x) in zip(ny, o["cells"])): break
                    dy += 1 if d == "down" else -1
            else:
                while True:
                    nx = [(x + dx + (1 if d == "right" else -1)) for y, x in o["cells"]]
                    if any(v < 0 or v >= W for v in nx) or any(out[y][v] != bg for v, (y, x) in zip(nx, o["cells"])): break
                    dx += 1 if d == "right" else -1
            for y, x in o["cells"]: out[y + dy][x + dx] = o["color"]
        return out
    return fn

REL = [("odd_keep", lambda t: t_odd_one_out(t, "keep")), ("odd_remove", lambda t: t_odd_one_out(t, "remove")),
       ("recolor_pos_rank", t_recolor_by_pos_rank),
       ("obj_gravity_down", lambda t: t_whole_object_gravity(t, "down")),
       ("obj_gravity_left", lambda t: t_whole_object_gravity(t, "left"))]

if __name__ == "__main__":
    import sys, pathlib, json
    HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
    import unified as U
    tasks = T.load_tasks()
    base = set(json.load(open(HERE / "unified_solved.json"))) if (HERE / "unified_solved.json").exists() else set()
    print(f"RELACIONAL: testando {len(REL)} inducers relacionais em {len(tasks)} tasks (ortogonalidade vs simbolico)", flush=True)
    solved = []; novas = []
    for tk in tasks:
        for nm, ind in REL:
            try: fn = ind(tk)
            except Exception: fn = None
            if fn and T.vtrain(fn, tk) and T.vtest(fn, tk):
                solved.append((tk["_id"], nm))
                if tk["_id"] not in base: novas.append((tk["_id"], nm))
                break
    print(f"\nRELACIONAL resolveu: {len(solved)} | NOVAS (que o simbolico nao pega): {len(novas)}", flush=True)
    print(f"  novas: {novas}", flush=True)
    print(f"  todas: {solved[:20]}", flush=True)

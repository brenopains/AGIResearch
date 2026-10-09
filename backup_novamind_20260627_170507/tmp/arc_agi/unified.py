"""
unified.py — UNIFICACAO das tecnologias num solver ARC unico (passo 1: largura combinada + busca dirigida).
Une: (a) os 10 TEMPLATES parametricos object-level do tem_arc2 (recolor por rank/forma/buracos, mover, simetria,
preencher) — fortes mas SO same-size; (b) os 30 OPS geometricos/espaciais do seed_primitives (rotacao, espelho,
ladrilho, escala, gravidade, crop, contorno...) que mudam tamanho e pegam as tarefas que os templates nao alcancam.
Busca em FEIXE (beam) com composicao profunda + CREDITO PARCIAL (nº de pares de treino exatos = guia, ideia do
loop do lab) + DEDUP por estado (poda becos/equivalentes). Gate EXATO no teste held-out. Sem LLM.
Mede o DELTA honesto vs o baseline tem_arc2 (11/400 = 2.8%).
"""
import sys, os, json, glob, pathlib, time
from collections import Counter
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import tem_arc2 as T                                   # reusa templates, helpers, gate, loader
from seed_primitives import SEED
import forge as FORGE                                  # criacao aberta de primitivas (seletor x acao sintetizado)
from hypo import HYPO                                  # hipoteses estilo humano (raciocinio do usuario codado)

SAFE = {"__builtins__": {"range": range, "len": len, "min": min, "max": max, "abs": abs, "sorted": sorted,
        "list": list, "set": set, "dict": dict, "sum": sum, "enumerate": enumerate, "zip": zip, "map": map,
        "int": int, "float": float, "bool": bool, "tuple": tuple, "str": str, "round": round,
        "all": all, "any": any, "reversed": reversed, "filter": filter}}
def compile_op(src):
    import numpy as np
    ns = dict(SAFE); ns["__builtins__"] = dict(SAFE["__builtins__"]); ns["__builtins__"]["__import__"] = __import__
    try:
        exec(src, ns); fn = ns["transform"]
        return fn
    except Exception:
        return None
SEED_FNS = [(nm, compile_op(src)) for nm, src in SEED]
SEED_FNS = [(nm, f) for nm, f in SEED_FNS if f]

# ====== INDUTORES PARAMETRICOS EXTRA (preenchem buracos obvios dos 10 templates) ======
def t_global_recolor(task):
    """mapa de cor celula-a-celula consistente em todo o treino (same-size)."""
    mp = {}
    for p in task["train"]:
        gi, go = p["input"], p["output"]
        if T.dims(gi) != T.dims(go): return None
        for r1, r2 in zip(gi, go):
            for a, b in zip(r1, r2):
                if a in mp and mp[a] != b: return None
                mp[a] = b
    if all(k == v for k, v in mp.items()): return None
    def fn(g): return [[mp.get(v, v) for v in r] for r in g]
    return fn
def t_scale_inferred(task):
    """out = cada celula repetida kxk; infere k do treino."""
    ks = set()
    for p in task["train"]:
        hi, wi = T.dims(p["input"]); ho, wo = T.dims(p["output"])
        if hi == 0 or wi == 0 or ho % hi or wo % wi or ho // hi != wo // wi: return None
        ks.add(ho // hi)
    if len(ks) != 1: return None
    k = ks.pop()
    if not (2 <= k <= 6): return None
    def fn(g):
        out = []
        for r in g:
            row = []
            for v in r: row += [v] * k
            for _ in range(k): out.append(list(row))
        return out
    return fn
def t_tile_inferred(task):
    """out = grade inteira repetida a x b; infere (a,b)."""
    ab = set()
    for p in task["train"]:
        hi, wi = T.dims(p["input"]); ho, wo = T.dims(p["output"])
        if hi == 0 or wi == 0 or ho % hi or wo % wi: return None
        ab.add((ho // hi, wo // wi))
    if len(ab) != 1: return None
    a, b = ab.pop()
    if a * b < 2 or a > 5 or b > 5: return None
    def fn(g):
        out = []
        for _ in range(a):
            for r in g: out.append(list(r) * b)
        return out
    return fn
def t_downscale_inferred(task):
    """out = subamostra a cada k celulas; infere k (inverso de escala)."""
    ks = set()
    for p in task["train"]:
        hi, wi = T.dims(p["input"]); ho, wo = T.dims(p["output"])
        if ho == 0 or wo == 0 or hi % ho or wi % wo or hi // ho != wi // wo: return None
        ks.add(hi // ho)
    if len(ks) != 1: return None
    k = ks.pop()
    if not (2 <= k <= 6): return None
    def fn(g):
        return [[g[i * k][j * k] for j in range(len(g[0]) // k)] for i in range(len(g) // k)]
    return fn
def t_crop_largest(task):
    """extrai o maior objeto (sua bounding-box, com interior). Comum no ARC."""
    def fn(g):
        bg = T.bg_color(g); objs = T.components(g, bg)
        if not objs: return g
        o = max(objs, key=lambda z: z["size"])
        return [row[o["left"]:o["right"] + 1] for row in g[o["top"]:o["bot"] + 1]]
    return fn
def _symfill(g, kind):
    H, W = T.dims(g); bg = T.bg_color(g); out = T.gcopy(g)
    for i in range(H):
        for j in range(W):
            if out[i][j] != bg: continue
            if kind == "rot180": si, sj = H - 1 - i, W - 1 - j
            elif kind == "d" and H == W: si, sj = j, i
            elif kind == "ad" and H == W: si, sj = W - 1 - j, H - 1 - i
            else: continue
            if 0 <= si < H and 0 <= sj < W and g[si][sj] != bg: out[i][j] = g[si][sj]
    return out
def t_sym_rot180(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    return lambda g: _symfill(g, "rot180")
def t_sym_diag(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    return lambda g: _symfill(g, "d")
def t_mirror_tile(task):
    """out = input ladrilhado a x b COM reflexoes; infere (a,b) e o padrao de flip de cada bloco."""
    ab = set()
    for p in task["train"]:
        hi, wi = T.dims(p["input"]); ho, wo = T.dims(p["output"])
        if hi == 0 or wi == 0 or ho % hi or wo % wi: return None
        ab.add((ho // hi, wo // wi))
    if len(ab) != 1: return None
    a, b = ab.pop()
    if a * b < 2 or a > 4 or b > 4: return None
    gi, go = task["train"][0]["input"], task["train"][0]["output"]; hi, wi = T.dims(gi)
    pat = {}
    for r in range(a):
        for c in range(b):
            blk = [row[c * wi:(c + 1) * wi] for row in go[r * hi:(r + 1) * hi]]
            if blk == gi: pat[(r, c)] = (0, 0)
            elif [row[::-1] for row in blk] == gi: pat[(r, c)] = (0, 1)
            elif blk[::-1] == gi: pat[(r, c)] = (1, 0)
            elif [row[::-1] for row in blk[::-1]] == gi: pat[(r, c)] = (1, 1)
            else: return None
    def fn(g):
        h2, w2 = T.dims(g); out = [[0] * (w2 * b) for _ in range(h2 * a)]
        for r in range(a):
            for c in range(b):
                fv, fh = pat[(r, c)]
                blk = [list(row) for row in g]
                if fv: blk = blk[::-1]
                if fh: blk = [row[::-1] for row in blk]
                for i in range(h2):
                    for j in range(w2): out[r * h2 + i][c * w2 + j] = blk[i][j]
        return out
    return fn
def _connect(g):
    H, W = T.dims(g); bg = T.bg_color(g); out = T.gcopy(g)
    for i in range(H):
        cells = [(j, g[i][j]) for j in range(W) if g[i][j] != bg]
        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                j1, c1 = cells[a]; j2, c2 = cells[b]
                if c1 == c2 and all(g[i][k] == bg for k in range(j1 + 1, j2)) and j2 > j1 + 1:
                    for k in range(j1 + 1, j2): out[i][k] = c1
    for j in range(W):
        cells = [(i, g[i][j]) for i in range(H) if g[i][j] != bg]
        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                i1, c1 = cells[a]; i2, c2 = cells[b]
                if c1 == c2 and all(g[k][j] == bg for k in range(i1 + 1, i2)) and i2 > i1 + 1:
                    for k in range(i1 + 1, i2): out[k][j] = c1
    return out
def t_connect_same_color(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    return lambda g: _connect(g)
def _fillbbox(g):
    bg = T.bg_color(g); out = T.gcopy(g)
    for o in T.components(g, bg):
        for i in range(o["top"], o["bot"] + 1):
            for j in range(o["left"], o["right"] + 1): out[i][j] = o["color"]
    return out
def t_fill_object_bbox(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    return lambda g: _fillbbox(g)
def _smallest_tile(g):
    H, W = T.dims(g)
    for ph in range(1, H + 1):
        if H % ph: continue
        for pw in range(1, W + 1):
            if W % pw: continue
            if ph == H and pw == W: continue
            if all(g[i][j] == g[i % ph][j % pw] for i in range(H) for j in range(W)):
                return [row[:pw] for row in g[:ph]]
    return None
def t_dedup_periodic(task):
    for p in task["train"]:
        if _smallest_tile(p["input"]) != p["output"]: return None
    return lambda g: (_smallest_tile(g) or g)
EXTRA = [("global_recolor", t_global_recolor), ("scale_inferred", t_scale_inferred),
         ("tile_inferred", t_tile_inferred), ("downscale_inferred", t_downscale_inferred),
         ("crop_largest", t_crop_largest), ("sym_rot180", t_sym_rot180), ("sym_diag", t_sym_diag),
         ("mirror_tile", t_mirror_tile), ("connect_same_color", t_connect_same_color),
         ("fill_object_bbox", t_fill_object_bbox), ("dedup_periodic", t_dedup_periodic)]

MAXCELL = 1600                                          # guarda: poda grades absurdas (40x40)
def ok_grid(g):
    return isinstance(g, list) and g and isinstance(g[0], list) and len(g) * len(g[0]) <= MAXCELL

def state_sig(fn, task):
    try:
        outs = []
        for p in task["train"]:
            o = fn(p["input"])
            if not ok_grid(o): return None
            outs.append(tuple(tuple(r) for r in o))
        return hash(tuple(outs))
    except Exception:
        return None
def train_score(fn, task):
    s = 0
    for p in task["train"]:
        try:
            if T.geq(fn(p["input"]), p["output"]): s += 1
        except Exception:
            pass
    return s
def compose(pf, f):
    def c(g):
        return f(pf(g))
    return c

def solve_unified(task, beamk=8, maxdepth=3):
    ntrain = len(task["train"])
    if ntrain == 0: return None
    # pool combinado: templates parametricos (fitados na task) + ops geometricos fixos
    pool = []
    for nm, ind in T.TEMPLATES + EXTRA + HYPO:           # templates + indutores extra + hipoteses humanas
        try:
            fn = ind(task)
        except Exception:
            fn = None
        if fn: pool.append((nm, fn))
    pool += SEED_FNS
    try: pool += FORGE.forge_candidates(task)         # FORGE: primitivas sintetizadas pelos dados
    except Exception: pass
    # beam: comeca da identidade; depth-1 = ops sozinhos, depth-2/3 = composicoes
    ident = lambda g: [list(r) for r in g]
    beam = [("id", ident)]
    seen = set([state_sig(ident, task)])
    for _depth in range(maxdepth):
        scored = []
        for pname, pf in beam:
            for nm, f in pool:
                comp = compose(pf, f)
                sg = state_sig(comp, task)
                if sg is None or sg in seen: continue
                seen.add(sg)
                sc = train_score(comp, task)
                if sc == ntrain:                         # casa TODO o treino -> gate no teste held-out
                    if T.vtest(comp, task):
                        return (pname + ">>" + nm).replace("id>>", "")
                scored.append((pname + ">>" + nm, comp, sc))
        if not scored: break
        scored.sort(key=lambda x: -x[2])                 # CREDITO PARCIAL guia o feixe
        beam = [(n, f) for n, f, s in scored[:beamk]]
    return None

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    tasks = T.load_tasks(limit)
    print(f"UNIFIED (templates+ops, beam composicao): {len(tasks)} tasks", flush=True)
    t0 = time.time(); solved = []; bywhat = Counter()
    for i, tk in enumerate(tasks):
        prog = solve_unified(tk)
        if prog:
            solved.append((tk["_id"], prog)); depth = prog.count(">>") + 1; bywhat[depth] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tasks)}  resolvidas {len(solved)}  ({time.time()-t0:.0f}s)", flush=True)
    n = len(tasks)
    print(f"\nRESOLVIDAS (gate exato held-out): {len(solved)}/{n} = {len(solved)/n:.1%}", flush=True)
    print(f"baseline tem_arc2 = 11/400 = 2.8%  ->  DELTA: {len(solved)-11:+d} tarefas", flush=True)
    print(f"por profundidade de composicao: {dict(bywhat)}", flush=True)
    print(f"exemplos: {solved[:18]}", flush=True)
    json.dump([s[0] for s in solved], open(HERE / "unified_solved.json", "w"))

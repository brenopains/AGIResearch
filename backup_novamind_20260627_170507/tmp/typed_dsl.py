"""
DSL TIPADA SOBRE OBJETOS (convergencia do conselho: Stark/Ada/Banner/Turing/Wayne).
Programas tipados: Grid -> Objects -> Objects -> Grid. O beam compoe SO operacoes cujo tipo de saida casa o
proximo tipo de entrada -> poda massiva, composicao profunda (depth 4) sem explodir. Crédito parcial guia a busca;
gate EXATO no teste. Sem LLM. Mede NOVAS vs teto. (Operar sobre OBJETOS+RELACOES, nao pixels — a tese da Ada.)
"""
import os, json, glob, time, numpy as np
from collections import Counter
try: from scipy import ndimage
except Exception: ndimage = None

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "arc_data", "data", "training")

def bg(g): v, c = np.unique(g, return_counts=True); return int(v[np.argmax(c)])

class Obj:
    __slots__ = ("cells", "color", "size", "t", "l", "b", "r", "holes")
    def __init__(s, cells, color):
        s.cells = cells; s.color = color; s.size = len(cells)
        ys = [y for y, x in cells]; xs = [x for y, x in cells]
        s.t, s.l, s.b, s.r = min(ys), min(xs), max(ys), max(xs); s.holes = 0

def segment(g, conn8=False):
    if ndimage is None: return []
    b = bg(g); objs = []
    st = np.ones((3, 3)) if conn8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for c in range(10):
        if c == b: continue
        lbl, n = ndimage.label(g == c, structure=st)
        for k in range(1, n + 1):
            cells = list(map(tuple, np.argwhere(lbl == k)))
            if cells: objs.append(Obj(cells, c))
    return objs

# ---- operacoes TIPADAS. cada uma: (nome, tipo_in, tipo_out, fn) ----
def _ranks(objs, key, rev=True):
    vals = sorted(set(key(o) for o in objs), reverse=rev); return {v: i for i, v in enumerate(vals)}

def OPS_grid_to_objs():
    return [("seg4", "G", "O", lambda g: segment(g, False)),
            ("seg8", "O8", "O", lambda g: segment(g, True))]

def OPS_objs_filters(objs):
    if not objs: return []
    mx = max(o.size for o in objs); mn = min(o.size for o in objs)
    cc = Counter(o.color for o in objs); shp = Counter((o.b - o.t, o.r - o.l, o.size) for o in objs)
    F = [("largest", lambda L: [o for o in L if o.size == mx]),
         ("smallest", lambda L: [o for o in L if o.size == mn]),
         ("uniq_color", lambda L: [o for o in L if cc[o.color] == 1]),
         ("uniq_shape", lambda L: [o for o in L if shp[(o.b - o.t, o.r - o.l, o.size)] == 1]),
         ("border", lambda L: [o for o in L if o.t == 0 or o.l == 0]),
         ("not_border", lambda L: [o for o in L if o.t != 0 and o.l != 0])]
    return [(nm, "O", "O", fn) for nm, fn in F]

def OPS_objs_transform(g_shape):
    out = [("recolor_rank", "O", "O", _recolor_rank)]
    for c in range(10):
        out.append(("recolor_%d" % c, "O", "O", (lambda L, c=c: [_clone(o, c) for o in L])))
    return out

def _clone(o, color):
    n = Obj(o.cells, color); n.holes = o.holes; return n
def _recolor_rank(L):
    if not L: return L
    rk = _ranks(L, lambda o: o.size, True); pal = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    return [_clone(o, pal[rk[o.size] % len(pal)]) for o in L]

def render_on_bg(shape, b):
    def f(L):
        out = np.full(shape, b, int)
        for o in L:
            for y, x in o.cells: out[y, x] = o.color
        return out
    return f
def render_over(g):
    def f(L):
        out = g.copy()
        for o in L:
            for y, x in o.cells: out[y, x] = o.color
        return out
    return f

# busca em feixe tipada
def solve(train, tests, depth=4, beam=40):
    gi0 = train[0][0]; shp = gi0.shape; b = bg(gi0)
    # estados: (tipo, valor_por_demo_inputs, programa)
    demos_in = [i for i, o in train]; demos_out = [o for i, o in train]
    def score(grids):  # crédito parcial: nº de demos exatos
        s = 0
        for gg, oo in zip(grids, demos_out):
            if isinstance(gg, np.ndarray) and gg.shape == oo.shape and np.array_equal(gg, oo): s += 1
        return s
    # operacoes
    g2o = OPS_grid_to_objs()
    o2g = [("on_bg", "O", "G", lambda L, b=b, shp=shp: render_on_bg(shp, b)(L))]
    # estado inicial: Grid
    States = [("G", [g.copy() for g in demos_in], [])]
    best = None; seen = set()
    for d in range(depth):
        nxt = []
        for typ, vals, prog in States:
            if typ == "G":
                # tenta fechar (Grid ja e candidato)
                sc = score(vals)
                if sc == len(train):
                    if all(np.array_equal(v, o) for v, o in zip(vals, demos_out)):
                        return prog, _runner(prog, b, shp)
                if best is None or sc > best[0]: best = (sc, prog)
                # Grid -> Objects
                for nm, ti, to, fn in g2o:
                    try: nv = [fn(v) for v in vals]
                    except Exception: continue
                    nxt.append(("O", nv, prog + [(nm, fn)]))
            elif typ == "O":
                objs0 = vals[0]
                for nm, ti, to, fn in OPS_objs_filters(objs0) + OPS_objs_transform(shp):
                    try: nv = [fn(v) for v in vals]
                    except Exception: continue
                    nxt.append(("O", nv, prog + [(nm, fn)]))
                # Objects -> Grid (render)
                for nm, ti, to, fn in o2g:
                    try: nv = [fn(v) for v in vals]
                    except Exception: continue
                    nxt.append(("G", nv, prog + [(nm, fn)]))
                nxt.append(("G", [render_over(gi)(v) for gi, v in zip(demos_in, vals)], prog + [("over", "OVER")]))
        # poda por crédito parcial nos que ja sao Grid; mantem diversidade de Objects
        def st_score(st):
            typ, vals, prog = st
            if typ == "G": return score(vals)
            return 0
        nxt.sort(key=st_score, reverse=True)
        States = nxt[:beam]
        if not States: break
    return None, None

def _runner(prog, b, shp):
    def run(g):
        cur = g; typ = "G"
        for step in prog:
            nm, fn = step
            if fn == "OVER":
                tmp = g.copy()
                for o in cur:
                    for y, x in o.cells: tmp[y, x] = o.color
                cur = tmp; typ = "G"
            else:
                cur = fn(cur)
                typ = "O" if nm.startswith("seg") or typ == "O" and not nm == "on_bg" else "G"
        return cur if isinstance(cur, np.ndarray) else g
    return run

def prog_name(prog): return ">>".join(nm for nm, _ in prog)

def main():
    t0 = time.time()
    ceil = set()
    for f in ["arc_ceiling_v4.json", "arc_ceiling_v3.json", "arc_unified_final.json", "h111_solved.json"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            try: ceil |= set(json.load(open(p)))
            except Exception: pass
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    solved = {}; novas = {}
    for fp in files:
        name = os.path.basename(fp); t = json.load(open(fp))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        try: prog, run = solve(train, tests)
        except Exception: prog = None
        if prog is None: continue
        try:
            if all(np.array_equal(run(i), o) for i, o in train):
                outs = [run(xi) for xi, _ in tests]
                if all(isinstance(r, np.ndarray) and r.shape == yo.shape and np.array_equal(r, yo) for r, (_, yo) in zip(outs, tests)):
                    solved[name] = prog_name(prog)
                    if name not in ceil: novas[name] = prog_name(prog)
        except Exception: pass
    print("=" * 74)
    print(f"DSL TIPADA SOBRE OBJETOS | teto={len(ceil)} | resolveu(total)={len(solved)} | NOVAS={len(novas)}: {sorted(novas)}")
    for n in sorted(novas)[:14]: print(f"    NOVA {n}: {novas[n]}")
    print(f"  novo TETO: {len(ceil | set(solved))}/400 = {100*len(ceil | set(solved))/400:.1f}%  ({time.time()-t0:.0f}s)")
    json.dump(sorted(ceil | set(solved)), open(os.path.join(HERE, "arc_ceiling_v5.json"), "w"))

if __name__ == "__main__":
    main()

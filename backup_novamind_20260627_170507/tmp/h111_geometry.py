"""
PASSO 1-A do conselho: GEOMETRIA-DE-SAIDA. O motor H111 descarta 138 tarefas que mudam de forma (linha 96).
Aqui abduzimos o FATOR inteiro (Hout/Hin, Wout/Win) dos demos e testamos uma familia de regras de geometria:
  tile, scale(kron), self-tile/fractal (bloco = copia da entrada onde a celula e nao-fundo), mirror-tile,
  downscale (block-reduce), crop (bbox do nao-fundo / sub-bloco). Gate EXATO nos demos + teste. Mede NOVAS vs teto.
Honesto: vocabulario de geometria fixo; mede se a parede das different-shape era falta DESSA familia.
"""
import os, json, glob, time, numpy as np

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "arc_data", "data", "training")

def bg(g):
    v, c = np.unique(g, return_counts=True); return int(v[np.argmax(c)])

def _factor(train):
    facs = set()
    for i, o in train:
        ih, iw = i.shape; oh, ow = o.shape
        if not ih or not iw: return None
        if oh % ih or ow % iw: return None
        facs.add((oh // ih, ow // iw))
    return facs.pop() if len(facs) == 1 else None

def _divfactor(train):
    facs = set()
    for i, o in train:
        ih, iw = i.shape; oh, ow = o.shape
        if not oh or not ow: return None
        if ih % oh or iw % ow: return None
        facs.add((ih // oh, iw // ow))
    return facs.pop() if len(facs) == 1 else None

# ---- candidatos de geometria (cada um e fn(grid)->grid) ----
def g_tile(rh, rw): return lambda g: np.tile(g, (rh, rw))
def g_scale(rh, rw): return lambda g: np.kron(g, np.ones((rh, rw), g.dtype))
def g_mirrortile(rh, rw):
    def f(g):
        rows = []
        for bi in range(rh):
            row = []
            for bj in range(rw):
                t = g
                if bi % 2: t = t[::-1, :]
                if bj % 2: t = t[:, ::-1]
                row.append(t)
            rows.append(np.concatenate(row, 1))
        return np.concatenate(rows, 0)
    return f
def g_selftile(pred):
    """fractal: out[bloco(i,j)] = copia da entrada se pred(cell) senao fundo. (007bbfb7-style)"""
    def f(g):
        H, W = g.shape; b = bg(g); out = np.full((H * H, W * W), b, g.dtype)
        for i in range(H):
            for j in range(W):
                if pred(int(g[i, j]), b):
                    out[i * H:(i + 1) * H, j * W:(j + 1) * W] = g
        return out
    return f
def g_downscale(rh, rw):
    def f(g):
        H, W = g.shape
        return g[::rh, ::rw][:H // rh, :W // rw]
    return f
def g_crop_bbox(g):
    b = bg(g); ys, xs = np.where(g != b)
    if len(ys) == 0: return g
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def candidates(train):
    cands = []
    f = _factor(train)
    if f:
        rh, rw = f
        if (rh, rw) != (1, 1):
            cands += [("tile_%dx%d" % (rh, rw), g_tile(rh, rw)),
                      ("scale_%dx%d" % (rh, rw), g_scale(rh, rw)),
                      ("mirrortile_%dx%d" % (rh, rw), g_mirrortile(rh, rw))]
        # self-tile (fractal) quando o fator == dimensao da entrada
        ih, iw = train[0][0].shape
        if rh == ih and rw == iw:
            cands += [("selftile_nz", g_selftile(lambda c, b: c != b)),
                      ("selftile_eqbg", g_selftile(lambda c, b: c == b))]
            for col in range(10):
                cands.append(("selftile_c%d" % col, g_selftile(lambda c, b, col=col: c == col)))
    df = _divfactor(train)
    if df and df != (1, 1):
        cands.append(("downscale_%dx%d" % df, g_downscale(*df)))
    cands.append(("crop_bbox", g_crop_bbox))
    return cands

def main():
    t0 = time.time()
    ceil = set()
    for f in ["arc_unified_final.json", "h111_solved.json", "h111_connect_novas.json"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            try: ceil |= set(json.load(open(p)))
            except Exception: pass
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    solved = {}; novas = {}; att = 0
    for fp in files:
        name = os.path.basename(fp); t = json.load(open(fp))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if all(i.shape == o.shape for i, o in train): continue   # so DIFFERENT-shape (o que o H111 descarta)
        att += 1
        for nm, fn in candidates(train):
            try:
                if all(fn(i).shape == o.shape and np.array_equal(fn(i), o) for i, o in train):
                    outs = [fn(xi) for xi, _ in tests]
                    if all(r.shape == yo.shape and np.array_equal(r, yo) for r, (_, yo) in zip(outs, tests)):
                        solved[name] = nm
                        if name not in ceil: novas[name] = nm
                        break
            except Exception: pass
    print("=" * 74)
    print(f"PASSO 1-A (GEOMETRIA-DE-SAIDA) | different-shape testadas={att} | teto={len(ceil)}")
    print(f"  resolveu (total): {len(solved)} | NOVAS vs teto: {len(novas)}: {sorted(novas)}")
    for n in sorted(novas): print(f"    NOVA {n}: regra = {novas[n]}")
    print(f"  novo TETO: {len(ceil | set(solved))}/400 = {100*len(ceil | set(solved))/400:.1f}%")
    print(f"  ({time.time()-t0:.0f}s)")
    json.dump(sorted(novas), open(os.path.join(HERE, "h111_geometry_novas.json"), "w"))

if __name__ == "__main__":
    main()

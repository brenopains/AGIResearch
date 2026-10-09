"""
pure_neural2.py — ESCALA a criacao autonoma (H154): priors sinteticos MAIS RICOS (composicoes + nivel-objeto +
erosao/contorno/manter-maior/simetrizar) + 800 meta-steps + avaliacao maior. Continua SEM nenhuma primitiva minha
no teste: o NCA neural CRIA a regra sozinho (TTT). Mede SEM prior vs COM prior (rico).
"""
import sys, json, time, pathlib, random, numpy as np, torch, torch.nn.functional as F
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified_nca3 as N
from solver import ARC_DIR
random.seed(1); torch.manual_seed(1)
NC = N.NCOL

def rand_grid():
    H, W = random.randint(5, 14), random.randint(5, 14); g = np.zeros((H, W), dtype=np.int64)
    for _ in range(random.randint(2, 5)):                       # objetos como pequenos BLOCOS (mais estrutura)
        h, w = random.choice([(1, 1), (1, 2), (2, 1), (2, 2)]); c = random.randint(1, NC - 1)
        y, x = random.randrange(H - h + 1), random.randrange(W - w + 1); g[y:y + h, x:x + w] = c
    return g
def _components(g):
    H, W = g.shape; seen = np.zeros_like(g); objs = []
    for i in range(H):
        for j in range(W):
            if g[i, j] == 0 or seen[i, j]: continue
            col = g[i, j]; st = [(i, j)]; cells = []; seen[i, j] = 1
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = 1; st.append((ny, nx))
            objs.append((cells, col))
    return objs
def base_transform():
    t = random.choice(["recolor", "fliph", "flipv", "rot180", "dilate", "erode", "outline",
                       "gravD", "gravU", "gravL", "gravR", "translate", "keep_largest", "recolor_largest", "symH", "symV"])
    if t == "recolor":
        perm = list(range(1, NC)); random.shuffle(perm); lut = np.array([0] + perm)
        return lambda g: lut[g]
    if t == "fliph": return lambda g: g[:, ::-1].copy()
    if t == "flipv": return lambda g: g[::-1, :].copy()
    if t == "rot180": return lambda g: np.rot90(g, 2).copy()
    if t == "dilate":
        def f(g):
            o = g.copy()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)): o = np.where(o == 0, np.roll(g, (dy, dx), (0, 1)), o)
            return o
        return f
    if t == "erode":
        def f(g):
            H, W = g.shape; o = g.copy()
            for i in range(H):
                for j in range(W):
                    if g[i, j] != 0:
                        nb = [g[i + dy, j + dx] for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + dy < H and 0 <= j + dx < W]
                        if any(v == 0 for v in nb): o[i, j] = 0
            return o
        return f
    if t == "outline":
        def f(g):
            H, W = g.shape; o = g.copy()
            for i in range(H):
                for j in range(W):
                    if g[i, j] != 0:
                        nb = [g[i + dy, j + dx] for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + dy < H and 0 <= j + dx < W]
                        if nb and all(v != 0 for v in nb): o[i, j] = 0
            return o
        return f
    if t in ("gravD", "gravU", "gravL", "gravR"):
        def f(g):
            o = np.zeros_like(g)
            if t in ("gravD", "gravU"):
                for j in range(g.shape[1]):
                    col = [v for v in g[:, j] if v != 0]
                    if t == "gravD": o[g.shape[0] - len(col):, j] = col
                    else: o[:len(col), j] = col
            else:
                for i in range(g.shape[0]):
                    row = [v for v in g[i, :] if v != 0]
                    if t == "gravR": o[i, g.shape[1] - len(row):] = row
                    else: o[i, :len(row)] = row
            return o
        return f
    if t == "translate":
        dy, dx = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        return lambda g: np.roll(g, (dy, dx), (0, 1))
    if t == "keep_largest":
        def f(g):
            objs = _components(g)
            if not objs: return g
            big = max(objs, key=lambda o: len(o[0])); o = np.zeros_like(g)
            for y, x in big[0]: o[y, x] = big[1]
            return o
        return f
    if t == "recolor_largest":
        c = random.randint(1, NC - 1)
        def f(g):
            objs = _components(g)
            if not objs: return g
            big = max(objs, key=lambda o: len(o[0])); o = g.copy()
            for y, x in big[0]: o[y, x] = c
            return o
        return f
    if t in ("symH", "symV"):
        def f(g):
            o = g.copy(); H, W = g.shape
            for i in range(H):
                for j in range(W):
                    si, sj = (i, W - 1 - j) if t == "symH" else (H - 1 - i, j)
                    if o[i, j] == 0 and g[si, sj] != 0: o[i, j] = g[si, sj]
            return o
        return f
def synth_transform():
    f1 = base_transform()
    if random.random() < 0.35:                                   # COMPOSICAO de 2 (regras mais profundas)
        f2 = base_transform(); return lambda g: f2(f1(g))
    return f1
def synth_task(K=3):
    f = synth_transform(); return [(g, f(g)) for g in (rand_grid() for _ in range(K))]

def ttt(init_sd, demos, steps=280, T=16):
    net = N.NCA3().to(N.DEV)
    if init_sd is not None: net.load_state_dict(init_sd)
    opt = torch.optim.AdamW(net.parameters(), 2e-3); X, Y = [], []
    for i, o in demos:
        for k in range(8):
            X.append(N.to_state(np.ascontiguousarray(N.dih(i, k)))); Y.append(N.to_target(np.ascontiguousarray(N.dih(o, k))))
    X = torch.stack(X).to(N.DEV); Y = torch.stack(Y).to(N.DEV); net.train()
    for _ in range(steps):
        outs = net.rollout(X.clone(), T, collect_last=3)
        loss = sum(F.cross_entropy(s[:, :NC], Y, ignore_index=-100) for s in outs) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net

def pretrain(metasteps=800):
    meta = N.NCA3().to(N.DEV); print(f"[{N.DEV}] meta-pre-treino RICO ({metasteps})...", flush=True)
    for ms in range(metasteps):
        net = ttt(meta.state_dict(), synth_task(), steps=16)
        with torch.no_grad():
            for pm, pn in zip(meta.parameters(), net.parameters()): pm += 0.1 * (pn - pm)
        if ms % 100 == 0: print(f"  meta {ms}/{metasteps}", flush=True)
    return meta.state_dict()

def eval_solves(init_sd, tids, tag):
    n = 0; got = []
    for tid in tids:
        t = json.load(open(ARC_DIR / (tid + ".json")))
        demos = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if not demos or not all(i.shape == o.shape for i, o in demos + tests): continue
        net = ttt(init_sd, demos)
        if not all(np.array_equal(N._pred_canon(net, i, 16), o) for i, o in demos): continue
        if all(np.array_equal(N._pred_vote(net, xi, 16), yo) for xi, yo in tests): n += 1; got.append(tid)
    print(f"  {tag}: {n} -> {got}", flush=True); return n, got

if __name__ == "__main__":
    allt = sorted(p.name[:-5] for p in ARC_DIR.glob("*.json")); same = []
    for tid in allt:
        t = json.load(open(ARC_DIR / (tid + ".json")))
        d = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        te = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if d and all(i.shape == o.shape for i, o in d + te): same.append(tid)
    sl = same[:110]; print(f"same-shape: {len(same)} (avaliando {len(sl)})", flush=True); t0 = time.time()
    meta = pretrain(800)
    n1, g1 = eval_solves(meta, sl, "COM prior RICO")
    n0, g0 = eval_solves(None, sl, "SEM prior (do zero)")
    novas = [x for x in g1 if x not in g0]
    print(f"\n=== CRIACAO AUTONOMA NEURAL v2 (priors ricos), {len(sl)} same-shape ===", flush=True)
    print(f"  SEM prior: {n0}  |  COM prior RICO: {n1}  |  novas pelo prior: {len(novas)} {novas}", flush=True)
    print(f"  vs v1 (prior simples, 60 tarefas): 3->5. tempo {time.time()-t0:.0f}s", flush=True)
    json.dump({"sem": n0, "com": n1, "g0": g0, "g1": g1, "novas": novas}, open(HERE / "pure_neural2_result.json", "w"))

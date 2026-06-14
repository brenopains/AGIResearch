"""
pure_neural.py — CRIACAO AUTONOMA (ideia do usuario): SEM nenhuma primitiva minha no teste. O NCA neural CRIA a
regra sozinho a partir dos demos (test-time training). Testa se um PRIOR (meta-pre-treino reptile em transformacoes
sinteticas = "instinto de dinamica de grade") ajuda a maquina a criar melhor — vs SEM prior (do zero).
Priors sao permitidos (humanos tem); o proibido e eu dar a solucao pronta. Mede solves PUROS (neural, sem simbolo).
"""
import sys, json, time, pathlib, random, copy, numpy as np, torch, torch.nn.functional as F
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified_nca3 as N
from solver import ARC_DIR
random.seed(0); torch.manual_seed(0)

# ---- transformacoes SINTETICAS (priors p/ pre-treino; NAO sao solvers do teste) ----
def rand_grid():
    H, W = random.randint(4, 12), random.randint(4, 12); g = np.zeros((H, W), dtype=np.int64)
    for _ in range(random.randint(3, 9)): g[random.randrange(H), random.randrange(W)] = random.randint(1, N.NCOL - 1)
    return g
def synth_transform():
    t = random.choice(["recolor", "fliph", "flipv", "rot180", "dilate", "gravity", "translate"])
    if t == "recolor":
        perm = list(range(1, N.NCOL)); random.shuffle(perm); lut = [0] + perm
        return lambda g: np.array([[lut[v] for v in r] for r in g])
    if t == "fliph": return lambda g: g[:, ::-1].copy()
    if t == "flipv": return lambda g: g[::-1, :].copy()
    if t == "rot180": return lambda g: np.rot90(g, 2).copy()
    if t == "dilate":
        def f(g):
            o = g.copy()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)): o = np.where(o == 0, np.roll(g, (dy, dx), (0, 1)), o)
            return o
        return f
    if t == "gravity":
        def f(g):
            o = np.zeros_like(g)
            for j in range(g.shape[1]):
                col = [v for v in g[:, j] if v != 0]; o[g.shape[0] - len(col):, j] = col
            return o
        return f
    dy, dx = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
    return lambda g: np.roll(g, (dy, dx), (0, 1))
def synth_task(K=3):
    f = synth_transform(); return [(g, f(g)) for g in (rand_grid() for _ in range(K))]

def ttt(init_sd, demos, steps=220, T=16):
    net = N.NCA3().to(N.DEV)
    if init_sd is not None: net.load_state_dict(init_sd)
    opt = torch.optim.AdamW(net.parameters(), 2e-3)
    X, Y = [], []
    for i, o in demos:
        for k in range(8):
            X.append(N.to_state(np.ascontiguousarray(N.dih(i, k)))); Y.append(N.to_target(np.ascontiguousarray(N.dih(o, k))))
    X = torch.stack(X).to(N.DEV); Y = torch.stack(Y).to(N.DEV); net.train()
    for _ in range(steps):
        outs = net.rollout(X.clone(), T, collect_last=3)
        loss = sum(F.cross_entropy(s[:, :N.NCOL], Y, ignore_index=-100) for s in outs) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net

def pretrain(metasteps=200):
    meta = N.NCA3().to(N.DEV)
    print(f"[{N.DEV}] meta-pre-treino reptile (prior de dinamica de grade)...", flush=True)
    for ms in range(metasteps):
        net = ttt(meta.state_dict(), synth_task(), steps=18)
        with torch.no_grad():
            for pm, pn in zip(meta.parameters(), net.parameters()): pm += 0.12 * (pn - pm)
        if ms % 50 == 0: print(f"  meta {ms}/{metasteps}", flush=True)
    return meta.state_dict()

def eval_solves(init_sd, tids, tag, steps=250):
    n = 0; got = []
    for tid in tids:
        t = json.load(open(ARC_DIR / (tid + ".json")))
        demos = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if not demos or not all(i.shape == o.shape for i, o in demos + tests): continue
        net = ttt(init_sd, demos, steps)
        if not all(np.array_equal(N._pred_canon(net, i, 16), o) for i, o in demos): continue
        if all(np.array_equal(N._pred_vote(net, xi, 16), yo) for xi, yo in tests): n += 1; got.append(tid)
    print(f"  {tag}: {n} solves -> {got}", flush=True)
    return n, got

if __name__ == "__main__":
    allt = sorted(p.name[:-5] for p in ARC_DIR.glob("*.json"))
    # filtra same-shape (onde o NCA puro opera) e pega uma fatia
    same = []
    for tid in allt:
        t = json.load(open(ARC_DIR / (tid + ".json")))
        d = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        te = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if d and all(i.shape == o.shape for i, o in d + te): same.append(tid)
    slice_ = same[:60]
    print(f"tarefas same-shape: {len(same)} (avaliando {len(slice_)})", flush=True)
    t0 = time.time()
    n0, g0 = eval_solves(None, slice_, "SEM prior (do zero)")
    meta = pretrain(200)
    n1, g1 = eval_solves(meta, slice_, "COM prior (meta-init)")
    novas = [x for x in g1 if x not in g0]
    print(f"\n=== CRIACAO AUTONOMA NEURAL (sem minhas primitivas), {len(slice_)} same-shape ===", flush=True)
    print(f"  SEM prior: {n0}  |  COM prior: {n1}  |  novas pelo prior: {len(novas)} {novas}", flush=True)
    print(f"  prior {'AJUDA a maquina criar' if n1 > n0 else 'nao ajudou aqui'} | tempo {time.time()-t0:.0f}s", flush=True)
    json.dump({"sem_prior": n0, "com_prior": n1, "g0": g0, "g1": g1, "novas": novas},
              open(HERE / "pure_neural_result.json", "w"))

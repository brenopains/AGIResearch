"""
PASSO 2 do conselho: COMPOR vocabularios. Motor por-celula = base H111 + CONECTIVIDADE (1-B) + RAIOS + SIMETRIA.
Raios: cor nao-fundo mais proxima em cada uma das 4 direcoes (capta "projetar linha", "preencher ate bater").
Simetria: cor do parceiro espelhado (h/v) — capta completar simetria / pintar pelo reflexo.
Re-roda a inducao gulosa MDL (gate exato) e mede NOVAS vs teto 106. Honesto: vocabulario fixo expandido; testa
se a parede dos 111 "recolor-local/outro" era falta DESTAS features ou se e a saturacao que o Turing previu.
"""
import os, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p = os.path.join(os.path.dirname(__file__), f); s = importlib.util.spec_from_file_location(f[:-3], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
CN = _load("h111_connect.py")           # base + conectividade (feats2, FEATS2)
H111 = CN.H111
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "arc_data", "data", "training")

def bg(g):
    v, c = np.unique(g, return_counts=True); return int(v[np.argmax(c)])

def ray_features(g):
    H, W = g.shape; b = bg(g)
    up = np.zeros((H, W), np.int64); dn = np.zeros((H, W), np.int64)
    lf = np.zeros((H, W), np.int64); rt = np.zeros((H, W), np.int64)
    for x in range(W):
        last = 0
        for y in range(H): up[y, x] = last;  last = g[y, x] if g[y, x] != b else last
        last = 0
        for y in range(H - 1, -1, -1): dn[y, x] = last; last = g[y, x] if g[y, x] != b else last
    for y in range(H):
        last = 0
        for x in range(W): lf[y, x] = last; last = g[y, x] if g[y, x] != b else last
        last = 0
        for x in range(W - 1, -1, -1): rt[y, x] = last; last = g[y, x] if g[y, x] != b else last
    return {"ray_up": up, "ray_dn": dn, "ray_lf": lf, "ray_rt": rt}

def sym_features(g):
    return {"sym_h": g[:, ::-1].astype(np.int64), "sym_v": g[::-1, :].astype(np.int64)}

RFEATS = ["ray_up", "ray_dn", "ray_lf", "ray_rt", "sym_h", "sym_v"]
FEATS3 = CN.FEATS2 + RFEATS

def feats3(g):
    F = CN.feats2(g); F.update(ray_features(g)); F.update(sym_features(g)); return F

def induce3(demos, max_rules=18):
    if not all(i.shape == o.shape for i, o in demos): return None
    fa = {k: [] for k in FEATS3}; Y = []; IN = []
    for i, o in demos:
        F = feats3(i)
        for k in FEATS3: fa[k].append(F[k].ravel())
        Y.append(o.ravel()); IN.append(i.ravel())
    fa = {k: np.concatenate(v) for k, v in fa.items()}; Y = np.concatenate(Y); IN = np.concatenate(IN)
    covered = np.zeros(len(Y), bool); rules = []
    for _ in range(max_rules):
        if covered.all(): break
        unc = ~covered; best = None
        for fk in FEATS3:
            arr = fa[fk]
            for v in np.unique(arr[unc]):
                m = unc & (arr == v)
                if not m.any(): continue
                cnt = int(m.sum())
                if np.all(Y[m] == IN[m]):
                    if best is None or cnt > best[0]: best = (cnt, fk, int(v), "copy", 0)
                u = np.unique(Y[m])
                if len(u) == 1:
                    if best is None or cnt > best[0]: best = (cnt, fk, int(v), "const", int(u[0]))
        if best is None or best[0] == 0: return None
        cnt, fk, v, act, col = best
        m = (~covered) & (fa[fk] == v); rules.append((fk, v, act, col)); covered |= m
    return rules if covered.all() else None

def apply3(rules, g):
    F = feats3(g); flat = {k: F[k].ravel() for k in FEATS3}
    out = g.ravel().copy(); assigned = np.zeros(g.size, bool)
    for fk, v, act, col in rules:
        m = (~assigned) & (flat[fk] == v)
        if act == "const": out[m] = col
        assigned |= m
    return out.reshape(g.shape)

def main():
    t0 = time.time()
    ceil = set()
    for f in ["arc_ceiling_v2.json", "arc_unified_final.json", "h111_solved.json", "h111_connect_novas.json"]:
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
        if not all(i.shape == o.shape for i, o in train): continue
        att += 1
        try: rules = induce3(train)
        except Exception: rules = None
        if rules is None: continue
        if all(np.array_equal(apply3(rules, i), o) for i, o in train):
            outs = [apply3(rules, xi) for xi, _ in tests]
            if all(r.shape == yo.shape and np.array_equal(r, yo) for r, (_, yo) in zip(outs, tests)):
                solved[name] = rules
                if name not in ceil: novas[name] = rules
    print("=" * 74)
    print(f"PASSO 2 (base+conectividade+RAIOS+SIMETRIA) | mesmo-shape={att}/400 | teto={len(ceil)}")
    print(f"  resolveu (total): {len(solved)} | NOVAS vs teto v2: {len(novas)}: {sorted(novas)}")
    print(f"  novo TETO: {len(ceil | set(solved))}/400 = {100*len(ceil | set(solved))/400:.1f}%")
    for n in sorted(novas)[:14]:
        print(f"    NOVA {n}: [{H111.readable(novas[n])}]")
    print(f"  ({time.time()-t0:.0f}s)")
    json.dump(sorted(ceil | set(solved)), open(os.path.join(HERE, "arc_ceiling_v3.json"), "w"))

if __name__ == "__main__":
    main()

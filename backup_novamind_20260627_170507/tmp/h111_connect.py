"""
PASSO 1-B do conselho: adiciona UM vocabulario novo ao motor por-celula H111 — CONECTIVIDADE (componente conexo).
A unica adicao de VOCABULARIO (nao de busca) que o registro nunca testou. Predicados novos por celula:
  comp_rank (rank do tamanho do componente same-cor, capado), in_largest, in_smallest, enclosed (fundo encerrado),
  comp_size_bucket. Re-roda a inducao gulosa MDL (gate exato) nas tarefas mesmo-shape e mede NOVAS vs teto 101.
Honesto: continua um fecho (vocabulario fixo, agora + conectividade); mede se a parede era falta DESSE vocabulario.
"""
import os, sys, json, glob, time, numpy as np
from scipy import ndimage
import importlib.util
def _load(f):
    p = os.path.join(os.path.dirname(__file__), f); s = importlib.util.spec_from_file_location(f[:-3], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
H111 = _load("exp_h111_grammarforge.py")
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "arc_data", "data", "training")

def bg(g):
    v, c = np.unique(g, return_counts=True); return int(v[np.argmax(c)])

def connect_features(g):
    """features de CONECTIVIDADE por celula (componente conexo same-cor, 4-conn)."""
    H, W = g.shape; b = bg(g)
    comp_size = np.zeros((H, W), np.int64); comp_rank = np.zeros((H, W), np.int64)
    in_largest = np.zeros((H, W), np.int64); in_smallest = np.zeros((H, W), np.int64)
    enclosed = np.zeros((H, W), np.int64)
    sizes = []  # (size, mask) p/ rank dos componentes de FRENTE
    for c in range(0, 10):
        if c == b: continue
        lbl, n = ndimage.label(g == c)
        for k in range(1, n + 1):
            m = (lbl == k); sz = int(m.sum()); sizes.append((sz, m))
    if sizes:
        uniq = sorted(set(s for s, _ in sizes), reverse=True)
        rankmap = {s: min(i + 1, 6) for i, s in enumerate(uniq)}   # rank capado em 6
        mx = max(s for s, _ in sizes); mn = min(s for s, _ in sizes)
        for sz, m in sizes:
            comp_size[m] = min(sz, 9); comp_rank[m] = rankmap[sz]
            if sz == mx: in_largest[m] = 1
            if sz == mn: in_smallest[m] = 1
    # encerrado: componente de FUNDO que nao toca a borda (buraco)
    lblb, nb = ndimage.label(g == b)
    border = set(np.unique(np.concatenate([lblb[0, :], lblb[-1, :], lblb[:, 0], lblb[:, -1]])))
    for k in range(1, nb + 1):
        if k not in border: enclosed[lblb == k] = 1
    return {"comp_rank": comp_rank, "in_largest": in_largest, "in_smallest": in_smallest,
            "enclosed": enclosed, "comp_size": comp_size}

CFEATS = ["comp_rank", "in_largest", "in_smallest", "enclosed", "comp_size"]
FEATS2 = H111.FEATS + CFEATS

def feats2(g):
    F = H111.cell_features(g); F.update(connect_features(g)); return F

def induce2(demos, max_rules=16):
    if not all(i.shape == o.shape for i, o in demos): return None
    fa = {k: [] for k in FEATS2}; Y = []; IN = []
    for i, o in demos:
        F = feats2(i)
        for k in FEATS2: fa[k].append(F[k].ravel())
        Y.append(o.ravel()); IN.append(i.ravel())
    fa = {k: np.concatenate(v) for k, v in fa.items()}; Y = np.concatenate(Y); IN = np.concatenate(IN)
    covered = np.zeros(len(Y), bool); rules = []
    for _ in range(max_rules):
        if covered.all(): break
        unc = ~covered; best = None
        for fk in FEATS2:
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

def apply2(rules, g):
    F = feats2(g); flat = {k: F[k].ravel() for k in FEATS2}
    out = g.ravel().copy(); assigned = np.zeros(g.size, bool)
    for fk, v, act, col in rules:
        m = (~assigned) & (flat[fk] == v)
        if act == "const": out[m] = col
        assigned |= m
    return out.reshape(g.shape)

def main():
    t0 = time.time()
    ceil = set()
    for f in ["arc_unified_final.json", "h111_solved.json"]:
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
        try: rules = induce2(train)
        except Exception: rules = None
        if rules is None: continue
        if all(np.array_equal(apply2(rules, i), o) for i, o in train):
            outs = [apply2(rules, xi) for xi, _ in tests]
            if all(r.shape == yo.shape and np.array_equal(r, yo) for r, (_, yo) in zip(outs, tests)):
                solved[name] = rules
                if name not in ceil: novas[name] = rules
    print("=" * 74)
    print(f"PASSO 1-B (H111 + CONECTIVIDADE) | mesmo-shape={att}/400 | teto={len(ceil)}")
    print(f"  resolveu (total): {len(solved)} | NOVAS vs teto: {len(novas)}: {sorted(novas)}")
    print(f"  novo TETO: {len(ceil | set(solved))}/400 = {100*len(ceil | set(solved))/400:.1f}%")
    for n in sorted(novas)[:12]:
        print(f"    NOVA {n}: [{H111.readable(novas[n])}]")
    print(f"  ({time.time()-t0:.0f}s)")
    json.dump(sorted(novas), open(os.path.join(HERE, "h111_connect_novas.json"), "w"))

if __name__ == "__main__":
    main()

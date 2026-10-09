"""
RETESTE de ideia do conselho dismissada cedo demais: CONJUNCAO de predicados, mas sobre o VOCABULARIO NOVO.
O Rick disse 'conjuncao saturou (H112=0)' — mas H112 rodou nas 19 features ANTIGAS, ANTES da conectividade/raios.
Aqui a inducao gulosa MDL pode usar predicados de 1 termo OU conjuncoes (own==cor AND feat==v) sobre
base+conectividade+raios+simetria. Capta regras cor-especificas-condicionadas-a-estrutura que 1 termo nao expressa.
Gate exato. Mede NOVAS vs teto v3 (107). Honesto: ainda fecho fixo; testa se a fusao tecnica-refutada x vocab-novo destrava.
"""
import os, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p = os.path.join(os.path.dirname(__file__), f); s = importlib.util.spec_from_file_location(f[:-3], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
CP = _load("h111_compose.py")           # feats3 = base+conectividade+raios+simetria
H111 = CP.CN.H111
FEATS3 = CP.FEATS3
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "arc_data", "data", "training")

def induce_conj(demos, max_rules=20):
    if not all(i.shape == o.shape for i, o in demos): return None
    fa = {k: [] for k in FEATS3}; Y = []; IN = []
    for i, o in demos:
        F = CP.feats3(i)
        for k in FEATS3: fa[k].append(F[k].ravel())
        Y.append(o.ravel()); IN.append(i.ravel())
    fa = {k: np.concatenate(v) for k, v in fa.items()}; Y = np.concatenate(Y); IN = np.concatenate(IN)
    own = fa["own"]; owncols = np.unique(own)
    covered = np.zeros(len(Y), bool); rules = []
    for _ in range(max_rules):
        if covered.all(): break
        unc = ~covered; best = None  # best=(cnt, kind, payload)
        # 1 termo
        for fk in FEATS3:
            arr = fa[fk]
            for v in np.unique(arr[unc]):
                m = unc & (arr == v)
                if not m.any(): continue
                cnt = int(m.sum())
                if np.all(Y[m] == IN[m]) and (best is None or cnt > best[0]): best = (cnt, "1", (fk, int(v), "copy", 0))
                u = np.unique(Y[m])
                if len(u) == 1 and (best is None or cnt > best[0]): best = (cnt, "1", (fk, int(v), "const", int(u[0])))
        # 2 termos: own==c AND feat==v  (cor-especifico x estrutura) — o que H112 nao tinha (sem conectividade/raios)
        for c in owncols:
            mc = unc & (own == c)
            if not mc.any(): continue
            for fk in FEATS3:
                if fk == "own": continue
                arr = fa[fk]
                for v in np.unique(arr[mc]):
                    m = mc & (arr == v)
                    if not m.any(): continue
                    cnt = int(m.sum())
                    if np.all(Y[m] == IN[m]) and (best is None or cnt > best[0]): best = (cnt, "2", (int(c), fk, int(v), "copy", 0))
                    u = np.unique(Y[m])
                    if len(u) == 1 and (best is None or cnt > best[0]): best = (cnt, "2", (int(c), fk, int(v), "const", int(u[0])))
        if best is None or best[0] == 0: return None
        cnt, kind, pl = best
        if kind == "1":
            fk, v, act, col = pl; m = (~covered) & (fa[fk] == v)
        else:
            c, fk, v, act, col = pl; m = (~covered) & (own == c) & (fa[fk] == v)
        rules.append((kind, pl)); covered |= m
    return rules if covered.all() else None

def apply_conj(rules, g):
    F = CP.feats3(g); flat = {k: F[k].ravel() for k in FEATS3}
    out = g.ravel().copy(); assigned = np.zeros(g.size, bool)
    for kind, pl in rules:
        if kind == "1":
            fk, v, act, col = pl; m = (~assigned) & (flat[fk] == v)
        else:
            c, fk, v, act, col = pl; m = (~assigned) & (flat["own"] == c) & (flat[fk] == v)
        if act == "const": out[m] = col
        assigned |= m
    return out.reshape(g.shape)

def readable(rules):
    parts = []
    for kind, pl in rules:
        if kind == "1":
            fk, v, act, col = pl; parts.append(f"{fk}=={v}->{'cor%d'%col if act=='const' else 'copia'}")
        else:
            c, fk, v, act, col = pl; parts.append(f"(own=={c}&{fk}=={v})->{'cor%d'%col if act=='const' else 'copia'}")
    return " ; ".join(parts)

def main():
    t0 = time.time()
    ceil = set()
    for f in ["arc_ceiling_v3.json", "arc_ceiling_v2.json", "arc_unified_final.json", "h111_solved.json"]:
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
        try: rules = induce_conj(train)
        except Exception: rules = None
        if rules is None: continue
        if all(np.array_equal(apply_conj(rules, i), o) for i, o in train):
            outs = [apply_conj(rules, xi) for xi, _ in tests]
            if all(r.shape == yo.shape and np.array_equal(r, yo) for r, (_, yo) in zip(outs, tests)):
                solved[name] = rules
                if name not in ceil: novas[name] = rules
    print("=" * 74)
    print(f"RETESTE CONJUNCAO x VOCAB-NOVO | mesmo-shape={att}/400 | teto v3={len(ceil)}")
    print(f"  resolveu (total): {len(solved)} | NOVAS vs teto: {len(novas)}: {sorted(novas)}")
    print(f"  novo TETO: {len(ceil | set(solved))}/400 = {100*len(ceil | set(solved))/400:.1f}%")
    for n in sorted(novas)[:14]:
        print(f"    NOVA {n}: [{readable(novas[n])}]")
    print(f"  ({time.time()-t0:.0f}s)")
    json.dump(sorted(ceil | set(solved)), open(os.path.join(HERE, "arc_ceiling_v4.json"), "w"))

if __name__ == "__main__":
    main()

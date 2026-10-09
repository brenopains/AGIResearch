"""
CONEXAO COM O ARC REAL: quantas das ~293 tarefas residuais sao COMPOSICOES das primitivas de core-knowledge?
Busca simbolica (rapida, gate EXATO) sobre composicoes (profundidade 1-3) das 10 primitivas core. Mede:
quantas resolve no total e quantas sao NOVAS vs o teto 107. Diz quanto do ARC REAL a receita (core-knowledge +
composicao) ja toca — e onde para. Honesto: e o conjunto PEQUENO de primitivas; ARC completo exige mais (cor
parametrica, mudanca de forma, contagem relacional). Mas mede o sinal real em dados reais.
"""
import os, json, glob, itertools, time, numpy as np
HERE = os.path.dirname(__file__); ARC = os.path.join(HERE, "arc_data", "data", "training")

def rot180(g): return np.rot90(g, 2)
def fliph(g): return g[:, ::-1]
def flipv(g): return g[::-1, :]
def transp(g): return g.T if g.shape[0] == g.shape[1] else g
def shiftU(g): return np.roll(g, -1, 0)
def shiftD(g): return np.roll(g, 1, 0)
def shiftL(g): return np.roll(g, -1, 1)
def shiftR(g): return np.roll(g, 1, 1)
def bg(g): v, c = np.unique(g, return_counts=True); return int(v[np.argmax(c)])
def color_cycle(g):
    b = bg(g); cols = sorted(set(int(v) for v in np.unique(g)) - {b})
    if not cols: return g
    nxt = {c: cols[(i + 1) % len(cols)] for i, c in enumerate(cols)}
    o = g.copy()
    for c in cols: o[g == c] = nxt[c]
    return o
def gravityD(g):
    b = bg(g); o = np.full_like(g, b)
    for j in range(g.shape[1]):
        col = g[:, j]; nz = col[col != b]
        if len(nz): o[g.shape[0] - len(nz):, j] = nz
    return o
PRIMS = [("rot180", rot180), ("fliph", fliph), ("flipv", flipv), ("transp", transp),
         ("shiftU", shiftU), ("shiftD", shiftD), ("shiftL", shiftL), ("shiftR", shiftR),
         ("color_cycle", color_cycle), ("gravityD", gravityD)]

def apply_prog(prog, g):
    for _, f in prog: g = np.ascontiguousarray(f(g))
    return g

def solves(prog, train, tests):
    try:
        if not all(np.array_equal(apply_prog(prog, i), o) for i, o in train): return False
        return all(apply_prog(prog, i).shape == o.shape and np.array_equal(apply_prog(prog, i), o) for i, o in tests)
    except Exception:
        return False

def main():
    ceil = set()
    for f in ["arc_ceiling_v4.json", "h111_solved.json"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            try: ceil |= set(x.replace(".json", "") for x in json.load(open(p)))
            except: pass
    ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ARC, "*.json")))
    solved = []; novas = []; t0 = time.time()
    # programas: profundidade 1, 2, 3 (composicoes das 10 primitivas)
    progs = [[p] for p in PRIMS]
    progs += [[a, b] for a in PRIMS for b in PRIMS]
    progs += [[a, b, c] for a in PRIMS for b in PRIMS for c in PRIMS[:6]]   # depth-3 limitado p/ velocidade
    for tid in ids:
        t = json.load(open(os.path.join(ARC, tid + ".json")))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if not all(i.shape == o.shape for i, o in train): continue        # so same-shape
        for prog in progs:
            if solves(prog, train, tests):
                solved.append(tid)
                if tid not in ceil: novas.append((tid, ">>".join(n for n, _ in prog)))
                break
    print("=" * 64)
    print(f"CORE-KNOWLEDGE x ARC REAL | {len(progs)} programas | teto={len(ceil)}")
    print(f"  resolve (total same-shape): {len(solved)}/400")
    print(f"  NOVAS vs teto 107: {len(novas)}")
    for tid, prog in novas[:20]: print(f"    NOVA {tid}: {prog}")
    print(f"  cobertura do ARC real por estas 10 primitivas+composicao: {len(solved)/400:.1%}  ({time.time()-t0:.0f}s)")
    print("  LEITURA: mede quanto do ARC REAL a receita ja toca com um conjunto PEQUENO de primitivas core.")
    print("           Expandir o conjunto (cor parametrica, shape-change, contagem) sobe a cobertura.")

if __name__ == "__main__":
    main()

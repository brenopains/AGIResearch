"""
CURRICULO AUTO-GERADO (validacao em miniatura da "fabricar a vida de biblioteca"): sem NENHUM dado externo, o
sistema gera as proprias tarefas (composicoes das primitivas atuais, profundidade crescente), e CRESCE a biblioteca
comprimindo sub-composicoes recorrentes em primitivas nomeadas (DreamCoder wake-sleep). A cada "epoca da vida",
mede a COBERTURA do ARC REAL (single ou composicao-2 da biblioteca atual). Pergunta honesta:
  a biblioteca cresce E a cobertura do ARC real SOBE sozinha? -> a auto-geracao adquire biblioteca util.
  biblioteca cresce mas cobertura ESTAGNA? -> composicao fica no mesmo fecho; falta DESCOBRIR operador novo (honesto).
Simbolico, rapido, sem LLM.
"""
import os, json, glob, random, itertools, time, numpy as np
from collections import Counter
HERE = os.path.dirname(__file__); ARC = os.path.join(HERE, "arc_data", "data", "training")
random.seed(0)

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
    o = g.copy()
    for i, c in enumerate(cols): o[g == c] = cols[(i + 1) % len(cols)]
    return o
def gravityD(g):
    b = bg(g); o = np.full_like(g, b)
    for j in range(g.shape[1]):
        col = g[:, j]; nz = col[col != b]
        if len(nz): o[g.shape[0] - len(nz):, j] = nz
    return o
BASE = {"rot180": rot180, "fliph": fliph, "flipv": flipv, "transp": transp, "shiftU": shiftU,
        "shiftD": shiftD, "shiftL": shiftL, "shiftR": shiftR, "color_cycle": color_cycle, "gravityD": gravityD}

def apply_chain(names, lib, g):
    for nm in names:
        for f in lib[nm]: g = np.ascontiguousarray(f(g))
    return g

def load_real():
    ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ARC, "*.json")))
    tasks = []
    for tid in ids:
        t = json.load(open(os.path.join(ARC, tid + ".json")))
        tr = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        te = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if all(i.shape == o.shape for i, o in tr): tasks.append((tid, tr, te))
    return tasks

def coverage(lib, tasks):
    names = list(lib.keys()); solved = 0
    progs = [[n] for n in names] + [[a, b] for a in names for b in names]
    for tid, tr, te in tasks:
        for prog in progs:
            try:
                if all(np.array_equal(apply_chain(prog, lib, i), o) for i, o in tr) and \
                   all(apply_chain(prog, lib, i).shape == o.shape and np.array_equal(apply_chain(prog, lib, i), o) for i, o in te):
                    solved += 1; break
            except Exception: pass
    return solved

def main():
    real = load_real(); t0 = time.time()
    lib = {nm: [f] for nm, f in BASE.items()}                 # genetica = as 10 base (cada = cadeia de 1)
    base_names = list(BASE.keys())
    print(f"CURRICULO AUTO-GERADO | tarefas reais same-shape={len(real)} | base={len(lib)} primitivas")
    print(f"  epoca 0 (so genetica): cobertura ARC real = {coverage(lib, real)}/400", flush=True)
    for epoca in range(1, 5):
        # GERAR: tarefas sinteticas = composicoes (profundidade cresce com a vida)
        depth = 2 + epoca
        bag = Counter()
        for _ in range(4000):
            chain = [random.choice(base_names) for _ in range(depth)]
            # SLEEP/ABSTRACAO: conta sub-cadeias de 2 que recorrem (candidatas a virar primitiva)
            for i in range(len(chain) - 1): bag[(chain[i], chain[i + 1])] += 1
        # CRESCER biblioteca: top sub-composicoes viram primitivas nomeadas (reutilizaveis)
        for (a, b), _ in bag.most_common(6):
            nm = f"{a}>>{b}"
            if nm not in lib: lib[nm] = lib[a] + lib[b]
        cov = coverage(lib, real)
        print(f"  epoca {epoca}: biblioteca={len(lib)} primitivas | cobertura ARC real = {cov}/400  ({time.time()-t0:.0f}s)", flush=True)
    print("=" * 64)
    print("  LEITURA: cobertura SOBE com a biblioteca -> auto-geracao adquire biblioteca util (validado).")
    print("           cobertura ESTAGNA -> composicao fica no mesmo fecho; falta DESCOBRIR operador NOVO (honesto).")

if __name__ == "__main__":
    main()

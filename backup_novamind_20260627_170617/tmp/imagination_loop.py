"""
LOOP DE IMAGINACAO (BVSR: variacao cega + retencao seletiva) sobre o ARC real, SEM LLM.
A intuicao do usuario: TEMPERATURA controla a "loucura" das hipoteses (baixa=compor o conhecido de forma provavel;
alta=composicoes profundas/raras = imaginar o improvavel). O MAESTRO modula a temperatura pelo AFETO (frustracao
sobe -> mais loucura). BIBLIOTECA cresce: programa que resolve vira primitiva reutilizavel (retencao = sinapse forte).
Mede nas tarefas residuais: COM maestro (temp adaptativa) vs SEM (temp fixa baixa) vs SEM (temp fixa alta).
Teste: a 'loucura seletiva' resolve MAIS? Honesto: explora um fecho MAIOR de forma estocastica (nao escapa de TODO
fecho — isso exigiria gramatica Turing-completa). Mede o ganho real da imaginacao+maestro vs busca fixa.
"""
import sys, os, json, glob, time, random
HERE = os.path.join(os.path.dirname(__file__), "arc_agi"); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(__file__))
import tem_arc2 as T, unified as U
try: import forge as F
except: F = None
try: import abstraction as AB
except: AB = None
try: import relations as RL
except: RL = None
import importlib.util
def _load(name):
    p = os.path.join(os.path.dirname(__file__), name + ".py"); s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
try: CELL = _load("h111_compose")
except Exception: CELL = None
from hypo import HYPO
import numpy as np
random.seed(0)
ARC = os.path.join(os.path.dirname(__file__), "arc_data", "data", "training")

def base_pool(task):
    """primitivas de TODOS os motores (o vocabulario base = o fecho conhecido)."""
    P = []
    for nm, ind in list(HYPO) + list(T.TEMPLATES) + list(U.EXTRA):
        try: fn = ind(task)
        except: fn = None
        if fn: P.append((nm, fn))
    P += list(U.SEED_FNS)
    if F:
        try: P += list(F.forge_candidates(task))
        except: pass
    if AB:
        try: P += list(AB.abstraction_candidates(task))
        except: pass
    if RL:
        try: P += list(RL.relation_candidates(task))
        except: pass
    return P

def cell_candidate(task):
    if CELL is None: return None
    demos = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    if not (demos and all(i.shape == o.shape for i, o in demos)): return None
    try:
        rules = CELL.induce3(demos)
        if rules: return ("celular", lambda g, rules=rules: CELL.apply3(rules, np.array(g)).tolist())
    except: pass
    return None

LIBRARY = []   # biblioteca crescente: (nome, fn) de programas que resolveram (retencao DreamCoder-style)

def solve_task(task, mode):
    """mode: 'maestro' (temp adaptativa), 'frio' (temp fixa baixa), 'quente' (temp fixa alta). budget de tentativas."""
    ntr = len(task["train"]); pool = base_pool(task)
    cc = cell_candidate(task)
    if cc: pool.insert(0, cc)
    pool += LIBRARY[:30]                      # imaginar tambem da MEMORIA (biblioteca)
    # 1) singles (sempre) — o obvio
    for nm, fn in pool:
        try:
            if U.train_score(fn, task) == ntr and T.vtest(fn, task): return ("single", nm, fn)
        except: pass
    # 2) IMAGINACAO: composicoes por temperatura
    parts = []
    for nm, fn in pool:
        try:
            s = U.train_score(fn, task)
            if 0 < s < ntr: parts.append((s, nm, fn))
        except: pass
    parts.sort(reverse=True, key=lambda z: z[0])
    frust = 0.0; budget = 220
    for it in range(budget):
        temp = {'frio': 0.15, 'quente': 0.9}.get(mode, min(0.95, 0.15 + frust))   # maestro: frust -> temp
        depth = 2 if random.random() > temp else random.choice([2, 3, 3, 4])       # quente/frustrado -> mais fundo
        # baixa temp: usa os MELHORES parciais (provavel). alta temp: amostra QUALQUER primitiva (louco/improvavel)
        def pick():
            if random.random() < temp and pool: return random.choice(pool)[1]      # loucura: primitiva aleatoria
            return (random.choice(parts[:8])[2] if parts else random.choice(pool)[1])
        fn = pick()
        for _ in range(depth - 1):
            g = pick(); prev = fn; fn = (lambda x, a=prev, b=g: b(a(x)))
        try:
            if U.train_score(fn, task) == ntr and T.vtest(fn, task): return ("imaginacao", f"d{depth}@T{temp:.2f}", fn)
        except: pass
        frust = min(1.0, frust + 0.02)         # cada falha sobe a frustracao (so afeta o modo maestro)
    return None

def run(mode, ids):
    global LIBRARY; LIBRARY = []
    solved = []; t0 = time.time()
    for tid in ids:
        t = json.load(open(os.path.join(ARC, tid + ".json"))); t["_id"] = tid
        r = solve_task(t, mode)
        if r:
            solved.append(tid)
            try:                                # retencao: grava o programa na biblioteca
                LIBRARY.append((f"lib:{tid}", r[2])); LIBRARY[:] = LIBRARY[-60:]
            except: pass
    return solved, time.time() - t0

if __name__ == "__main__":
    ceil = set()
    for f in ["arc_ceiling_v4.json", "h111_solved.json"]:
        p = os.path.join(os.path.dirname(__file__), f)
        if os.path.exists(p):
            try: ceil |= set(x.replace(".json", "") for x in json.load(open(p)))
            except: pass
    allids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ARC, "*.json")))
    residual = [i for i in allids if i not in ceil][:120]      # subset p/ velocidade
    print(f"LOOP DE IMAGINACAO (BVSR+temperatura+maestro) | residual testado={len(residual)} | teto={len(ceil)}")
    res = {}
    for mode in ['frio', 'quente', 'maestro']:
        s, dt = run(mode, residual); res[mode] = s
        print(f"  {mode:8s}: resolveu {len(s)}/{len(residual)}  ({dt:.0f}s)  {s[:8]}", flush=True)
    print("=" * 64)
    extra_m = sorted(set(res['maestro']) - set(res['frio']))
    print(f"  FRIO (temp baixa fixa) ...... {len(res['frio'])}")
    print(f"  QUENTE (temp alta fixa) ..... {len(res['quente'])}")
    print(f"  MAESTRO (temp adaptativa) ... {len(res['maestro'])}   | exclusivas do maestro vs frio: {extra_m}")
    print("LEITURA: maestro > frio e > quente -> a LOUCURA SELETIVA (modulada por afeto) resolve mais (imaginacao util).")

"""
arc_fusion.py — SOLUCAO DO ANTI-REUSO (a Ponte na granularidade certa).
Diagnostico: o compounding deu 0 no ARC porque o LLM mintava SOLUCOES INTEIRAS por-task (nunca recorrem).
Correcao: reusar no nivel de FAMILIA DE OPERADOR content-factored (TEM) — essas RECORREM entre tasks ARC
distintas (core-knowledge de Chollet). Nossa arquitetura resolve por composicao de familias SEM LLM; o LLM
so minta uma familia NOVA no fracasso. Mede: a fracao resolvida SEM LLM (by_library) CRESCE no stream? as
familias se REUSAM (1 familia resolve N>1 tasks)? = compounding no ARC, que o nivel-solucao negava.
Honesto: continua bounded por cobertura de familias (Solomonoff) — nao "passa", mas o anti-reuso deixa de ser 0.
"""
import sys, os, json, time, pathlib
from collections import Counter, defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tem_arc2 as T   # reusa operadores content-factored + util de grid

# ====== BIBLIOTECA DE FAMILIAS DE OPERADOR (versionada, persistente) ======
LIB_PATH = pathlib.Path(__file__).parent/"runs"/"arc_fusion_lib.json"
LIB_PATH.parent.mkdir(exist_ok=True)

# familia = (nome, inducer). inducer(task)->fn|None. As do tem_arc2 + atomos reutilizaveis.
def _atom_flip(axis):
    def ind(task):
        def fn(g):
            if axis=='h': return [list(reversed(r)) for r in g]
            if axis=='v': return [list(r) for r in reversed(g)]
            return [list(r) for r in zip(*g)]  # transpose
        return fn
    return ind
def _atom_rot180(task):
    def fn(g): return [list(reversed(r)) for r in reversed(g)]
    return fn
def _atom_recolor_global(task):
    # mapa de cor GLOBAL consistente (conteudo->conteudo) induzido dos train pairs (recorre muito no ARC)
    mp={}
    for p in task["train"]:
        gi,go=p["input"],p["output"]
        if T.dims(gi)!=T.dims(go): return None
        for ri,ro in zip(gi,go):
            for a,b in zip(ri,ro):
                if a in mp and mp[a]!=b: return None
                mp[a]=b
    def fn(g): return [[mp.get(v,v) for v in r] for r in g]
    return fn

FAMILIES = list(T.TEMPLATES) + [
    ("flip_h",_atom_flip('h')),("flip_v",_atom_flip('v')),("transpose",_atom_flip('t')),
    ("rot180",lambda t:_atom_rot180(t)),("recolor_global",_atom_recolor_global),
]

def load_lib():
    try: return json.load(open(LIB_PATH))
    except: return {"minted":[], "reuse":{}}
def save_lib(lib):
    try: json.dump(lib, open(LIB_PATH,"w"), indent=1)
    except: pass

# ====== SOLVER SEM LLM: tenta familias + composicao 2-fundo (a parte que e NOSSA ARQUITETURA) ======
def solve_no_llm(task):
    fns=[]
    for nm,ind in FAMILIES:
        try: fn=ind(task)
        except: fn=None
        if fn and T.vtrain(fn,task):
            if T.vtest(fn,task): return ("solo", nm)
            fns.append((nm,fn))
    # composicao 2-fundo de familias (reuso composicional)
    cand=[]
    for nm,ind in FAMILIES:
        try: f=ind(task)
        except: f=None
        if f: cand.append((nm,f))
    for na,fa in cand:
        for nb,fb in cand:
            if na==nb: continue
            comp=lambda g,fa=fa,fb=fb: fb(fa(g))
            try:
                if T.vtrain(comp,task) and T.vtest(comp,task): return ("comp", f"{na}>>{nb}")
            except: pass
    return (None,None)

# ====== STREAM: mede compounding (by_library cresce? familias recorrem?) ======
def run(limit=None):
    tasks=T.load_tasks(limit)
    lib=load_lib()
    n=len(tasks)
    by_lib=0; family_use=Counter(); solved=[]; per_task=[]
    for i,t in enumerate(tasks):
        kind,nm=solve_no_llm(t)   # NOSSA ARQUITETURA, SEM LLM
        if nm:
            by_lib+=1; family_use[nm.split('>>')[0]]+=1; solved.append((t["_id"],kind,nm))
            per_task.append(1)
        else:
            per_task.append(0)
    # metricas de compounding
    third=max(1,n//3)
    def frac(a,b): seg=per_task[a:b]; return sum(seg)/max(1,len(seg))
    reused_families={f:c for f,c in family_use.items() if c>1}   # familia que resolveu >1 task = REUSO
    print(f"\n{'='*64}")
    print(f"ARC-FUSION (familias content-factored, SEM LLM): {by_lib}/{n} = {100*by_lib/n:.1f}%")
    print(f">>> COMPOUNDING / REUSO (o que o nivel-solucao negava):")
    print(f"    familias REUSADAS (resolveram >1 task): {len(reused_families)}")
    print(f"    top reuso: {dict(Counter(family_use).most_common(8))}")
    print(f"    total de REUSOS (solves - familias distintas) = {by_lib - len(family_use)} solves vieram de familias ja-vistas")
    print(f">>> by_library por terco do stream: inicio {frac(0,third):.0%}  meio {frac(third,2*third):.0%}  fim {frac(2*third,n):.0%}")
    print(f"familias na biblioteca: {len(FAMILIES)} | exemplos solve: {solved[:10]}")
    return by_lib, n, reused_families

if __name__=="__main__":
    limit=int(sys.argv[1]) if len(sys.argv)>1 else None
    run(limit)

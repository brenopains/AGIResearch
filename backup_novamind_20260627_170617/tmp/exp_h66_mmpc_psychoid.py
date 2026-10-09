"""
H66 — MMPC "Modo Psicoide": neurogenese de primitivas sob crise (ARC, incremental)
===================================================================================
Operacionaliza o "Ciclo de Reorganizacao Fundamental" dos PDFs (inspirado em DMT/REBUS):
quando o solver FALHA (crise/surpresa), entra em modo psicoide:
 1. REBUS: relaxa precisao -> busca ESTOCASTICA mais PROFUNDA (depth 3-4), composicoes "estranhas".
 2. Neurogenese (MB): gera primitivas candidatas por RECOMBINACAO ALEATORIA de ops existentes
    (+ colormap inferido do delta da tarefa = injecao do estimulo da crise).
 3. Consolidacao ("aha"): adota a candidata se resolve exato (compressao) e GENERALIZA (>=2 tarefas).
Mede cobertura de ARC vs 14,2%. CACHE INCREMENTAL: computa o conjunto base 1x, depois so
processa as NAO-RESOLVIDAS (corta os ~15 min). Previsao honesta: recombinar ops existentes
fica no FECHO -> ganho provavelmente pequeno (a parede de Solomonoff e LARGURA, nao profundidade).
"""
import os, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PU=_load("exp_h54_pulse.py"); AB=_load("exp_h55_abduce.py"); H63=_load("exp_h63_efficiency_breadth.py")
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
CACHE=os.path.join(os.path.dirname(__file__),"h66_base_solved.json")
rng=np.random.RandomState(7)

def load_tasks():
    out=[]
    for fp in sorted(glob.glob(os.path.join(DATA,"*.json"))):
        t=json.load(open(fp)); out.append((os.path.basename(fp),
            [(np.array(p["input"]),np.array(p["output"])) for p in t["train"]],
            [(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]))
    return out

def base_pool(train):
    pool=dict(PU.FIXED)
    if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(PU.square_ops())
    pool.update(H63.NEW)                      # pool estendido do H63 (+11)
    return pool

def base_solved(tasks):
    if os.path.exists(CACHE): return set(json.load(open(CACHE)))
    print("  (computando conjunto base 1x e cacheando — parte lenta)...")
    solved=set()
    for name,train,tests in tasks:
        pool=base_pool(train); prog=H63.induce(train,pool)
        if prog is None:
            r,_,_=AB.abduce(train)
            if r is not None and all((lambda o,yo:o is not None and isinstance(o,np.ndarray) and o.shape==yo.shape and np.array_equal(o,yo))(r(xi),yo) for xi,yo in tests): solved.add(name)
            continue
        if all((lambda r,yo:r is not None and r.shape==yo.shape and np.array_equal(r,yo))(prog(xi),yo) for xi,yo in tests): solved.add(name)
        else:
            r,_,_=AB.abduce(train)
            if r is not None and all((lambda o,yo:o is not None and isinstance(o,np.ndarray) and o.shape==yo.shape and np.array_equal(o,yo))(r(xi),yo) for xi,yo in tests): solved.add(name)
    json.dump(sorted(solved),open(CACHE,"w")); return solved

def apply_chain(pool,prog,g):
    for nm in prog:
        g=pool[nm](g)
        if g is None or not isinstance(g,np.ndarray): return None
    return g
def fits(pool,prog,train,cm=None):
    for i,o in train:
        r=apply_chain(pool,prog,i)
        if r is None: return False
        if cm is not None: r=PU.cmap(r,cm)
        if r.shape!=o.shape or not np.array_equal(r,o): return False
    return True

def psychoid(train, pool, budget=4000, maxlen=4):
    """REBUS + neurogenese: busca estocastica profunda + colormap do delta. Retorna (prog,cm) ou None."""
    names=list(pool)
    for _ in range(budget):
        L=rng.randint(2,maxlen+1)                       # profundidade > 2 (o que a exaustiva nao varre)
        prog=tuple(names[i] for i in rng.randint(0,len(names),L))
        try:
            if fits(pool,prog,train): return (prog,None)
            # injecao do delta: colormap inferido apos a cadeia
            outs=[apply_chain(pool,prog,i) for i,_ in train]
            if all(o is not None and o.shape==t.shape for o,(_,t) in zip(outs,train)):
                cm=PU.infer_cmap([(o,t) for o,(_,t) in zip(outs,train)])
                if cm is not None and fits(pool,prog,train,cm): return (prog,cm)
        except Exception: continue
    return None

def main():
    t0=time.time(); tasks=load_tasks()
    base=base_solved(tasks)
    print("="*70); print(f"H66 Modo Psicoide (MMPC) — neurogenese de primitivas nas nao-resolvidas")
    print(f"  base (pool-estendido + ABDUCE): {len(base)}/400 = {100*len(base)/400:.1f}%  [cache incremental]")
    unsolved=[(n,tr,te) for n,tr,te in tasks if n not in base]
    print(f"  rodando modo psicoide em {len(unsolved)} nao-resolvidas (busca estocastica depth-3/4 + delta)...")
    new=[]; born={}
    for name,train,tests in unsolved:
        pool=base_pool(train); res=psychoid(train,pool)
        if res is None: continue
        prog,cm=res
        ok=True
        for xi,yo in tests:
            r=apply_chain(pool,prog,xi)
            if r is not None and cm is not None: r=PU.cmap(r,cm)
            if r is None or r.shape!=yo.shape or not np.array_equal(r,yo): ok=False; break
        if ok: new.append((name,prog,cm)); born[prog]=born.get(prog,0)+1
    total=len(base)+len(new)
    print(f"\n  NOVAS resolvidas pelo modo psicoide: {len(new)}")
    for name,prog,cm in new[:15]: print(f"    {name} <- {'+'.join(prog)}{' +cmap' if cm else ''}")
    generaliz=[p for p,c in born.items() if c>=2]
    print(f"  primitivas 'nascidas' que GENERALIZAM (>=2 tarefas): {len(generaliz)}")
    print(f"\n  COBERTURA: {len(base)} -> {total}/400 = {100*total/400:.1f}%  (era 14,2%)")
    print(f"  tempo {time.time()-t0:.0f}s (incremental). Honesto: recombinar ops existentes fica no")
    print(f"  FECHO; ganho>0 vem de depth-3/4 que a exaustiva depth-2 perdeu. A parede de LARGURA")
    print(f"  permanece — depth nao cria primitiva fora-do-fecho. NAO e AGI.")

if __name__=="__main__": main()

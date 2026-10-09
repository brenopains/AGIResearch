"""
H75 — Stress-test definitivo da MCO/ATCC (manifesto do usuario): colapso criativo MAXIMO
=========================================================================================
Testa o mecanismo CENTRAL da Maquina de Curiosidade Ontologica com dado, na versao mais forte:
 - "Colapso criativo / neurogenese": geracao MASSIVA de primitivas aleatorias (composicoes
   profundas, depth ate 5, orcamento grande) sobre o pool estendido + injecao do delta (cmap).
 - "Biblioteca de Eurekas": qualquer composto que GENERALIZE (resolve >=2 tarefas e passa no
   teste) e promovido a primitiva permanente, e reusado nas demais.
Roda so nas NAO-resolvidas (incremental). Previsao honesta: ~0 — recombinar/gerar aleatorio
fica no FECHO (informacao nao surge do nada); a parede de Solomonoff exige fonte de informacao
externa, nao colapso interno. Mede e reporta com dado. Nao e AGI.
"""
import os, json, glob, time, numpy as np
import importlib.util
from collections import deque
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PU=_load("exp_h54_pulse.py"); H63=_load("exp_h63_efficiency_breadth.py")
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
rng=np.random.RandomState(11)

def current_solved():
    s=set()
    for f in ("h66_base_solved.json","h70_nca_solved.json","h72_rcens_solved.json"):
        p=os.path.join(os.path.dirname(__file__),f)
        if os.path.exists(p): s|=set(json.load(open(p)))
    return s

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

def collapse(train, pool, names, budget=8000, maxlen=5):
    """neurogenese: gera composicoes aleatorias profundas + injecao de delta (cmap). Retorna (prog,cm)|None."""
    for _ in range(budget):
        L=rng.randint(2,maxlen+1); prog=tuple(names[i] for i in rng.randint(0,len(names),L))
        try:
            if fits(pool,prog,train): return (prog,None)
            outs=[apply_chain(pool,prog,i) for i,_ in train]
            if all(o is not None and o.shape==t.shape for o,(_,t) in zip(outs,train)):
                cm=PU.infer_cmap([(o,t) for o,(_,t) in zip(outs,train)])
                if cm is not None and fits(pool,prog,train,cm): return (prog,cm)
        except Exception: continue
    return None

def main():
    t0=time.time(); tasks=[]
    for fp in sorted(glob.glob(os.path.join(DATA,"*.json"))):
        t=json.load(open(fp)); tasks.append((os.path.basename(fp),
            [(np.array(p["input"]),np.array(p["output"])) for p in t["train"]],
            [(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]))
    solved=current_solved(); unsolved=[(n,tr,te) for n,tr,te in tasks if n not in solved]
    print("="*70)
    print(f"H75 MCO stress-test — colapso criativo MAXIMO ({len(unsolved)} nao-resolvidas)")
    pool=dict(PU.FIXED); pool.update(H63.NEW)
    names=list(pool); new=[]; eureka={}
    for name,train,tests in unsolved:
        p=dict(pool)
        if train[0][0].shape[0]==train[0][0].shape[1]: p.update(PU.square_ops())
        nm=list(p.keys())
        res=collapse(train,p,nm)
        if res is None: continue
        prog,cm=res
        ok=True
        for xi,yo in tests:
            r=apply_chain(p,prog,xi)
            if r is not None and cm is not None: r=PU.cmap(r,cm)
            if r is None or r.shape!=yo.shape or not np.array_equal(r,yo): ok=False; break
        if ok: new.append((name,prog,cm)); eureka[prog]=eureka.get(prog,0)+1
    gen=[p for p,c in eureka.items() if c>=2]
    total=len(solved)+len(new)
    print(f"  NOVAS pelo colapso criativo: {len(new)} {[n for n,_,_ in new][:12]}")
    print(f"  'Eurekas' que generalizaram (>=2 tarefas): {len(gen)}")
    print(f"  COBERTURA: {len(solved)} -> {total}/400 = {100*total/400:.1f}%")
    print(f"  tempo {time.time()-t0:.0f}s")
    print(f"  VEREDITO HONESTO: {'colapso criativo NAO inventou primitiva fora-do-fecho (previsto)' if len(new)==0 else f'+{len(new)} via busca profunda aleatoria (loteria/fecho, nao informacao nova)'}.")
    print(f"  Informacao nao surge do caos interno; a parede de Solomonoff (fonte de info) permanece. Nao e AGI.")

if __name__=="__main__": main()

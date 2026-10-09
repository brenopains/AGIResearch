"""
H98 — A MAQUINA CRESCE A PROPRIA LINGUAGEM DE PROGRAMACAO PRO ARC (DreamCoder honesto)
======================================================================================
Desafio ao meu proprio pessimismo: o H92 (library learning) falhou, mas eu testei a versao
ERRADA — abstrai sobre tarefas AUTO-GERADAS (presas ao fecho) com criterio ingenuo. O metodo
real (DreamCoder, Ellis 2021) faz WAKE-SLEEP sobre as tarefas REAIS e COMPOSTA: cada primitiva
abstraida destrava tarefas antes impossiveis. Nunca rodamos isso no ARC real.

Combina TUDO que deu certo + conserta o que falhou:
  - WAKE (resolver): busca evolutiva (H89, +2) sobre o DSL atual; gate exato.
  - SEMENTE: FIXED+NEW + as familias do H93 (+5: periodic/symmetrize/enclosed/tile) como primitivas.
  - SLEEP (abstrair): comprime as SOLUCOES REAIS em primitivas novas, pelo criterio de REUSO
    ENTRE TAREFAS DISTINTAS (nao frequencia ingenua) — a pressao de generalizacao que o gargalo
    de transmissao (H95) provou favorecer primitivas GERAIS que transferem.
  - COMPOSTA: o DSL crescido baratea/aprofunda a busca -> resolve mais -> mais p/ abstrair.
Mede por iteracao: total resolvido, NOVAS vs teto (91 do H93), tamanho do DSL, macros formados,
e SE composta (iteracao N+1 > N por causa das abstracoes). So-CPU. Gate exato (honestidade).
A aposta: se UMA abstracao destravar cascata, o DSL cresce sozinho e fura a barreira SEM codar a
mao. Honesto: provavelmente satura (residuo precisa de familia nova, nao recombinacao) — mas se
compostar, e o comeco da chama. Medido com dado, 360 graus.
"""
import os, sys, json, glob, time, random, numpy as np
from collections import Counter, defaultdict
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H63=_load("exp_h63_efficiency_breadth.py"); PU=H63.PU
Z=_load("exp_h93_zealot.py")     # familias H93: periodic_complete, symmetrize, enclosed_fill, tile_fn
random.seed(0); np.random.seed(0); LMAX=8

def seed_dsl():
    fns=dict(PU.FIXED); fns.update(PU.square_ops()); fns.update(H63.NEW)
    fns['periodic']=Z.periodic_complete
    for k in ('h','v','both','t','at'): fns[f'sym_{k}']=Z.symmetrize(k)
    fns['sym_rot4']=Z.symmetrize_rot4
    for c in range(1,7): fns[f'fill_{c}']=Z.enclosed_fill(c)
    for ky,kx in [(2,2),(1,2),(2,1),(3,1),(1,3),(3,3),(2,3),(3,2)]: fns[f'tile_{ky}x{kx}']=Z.tile_fn(ky,kx,False)
    return fns

def apply_prog(prog, fns, g):
    cur=g
    try:
        for nm in prog:
            cur=fns[nm](cur)
            if cur is None or not isinstance(cur,np.ndarray) or cur.size==0 or cur.ndim!=2: return None
        return cur
    except Exception: return None

def make_macro(seq, fns):
    flist=[fns[n] for n in seq]
    def m(g, flist=flist):
        cur=g
        for f in flist:
            cur=f(cur)
            if cur is None or not isinstance(cur,np.ndarray) or cur.size==0 or cur.ndim!=2: return None
        return cur
    return m

def score(prog, fns, demos):
    tot=0.0
    for i,o in demos:
        r=apply_prog(prog,fns,i)
        if r is None: return 0.0
        if r.shape==o.shape: tot+=float((r==o).mean())
        else:
            dh=abs(r.shape[0]-o.shape[0]); dw=abs(r.shape[1]-o.shape[1]); tot+=0.2*max(0.0,1-(dh+dw)/max(1,o.shape[0]+o.shape[1]))
    return tot/len(demos)

def evolve_solve(train, names, fns, weights, budget):
    grid={}; best=(0.0,None)
    def consider(p):
        if not p: return 0.0
        f=score(p,fns,train); bd=(min(len(p),LMAX),)+ (tuple() )
        i,o=train[0]; r=apply_prog(p,fns,i)
        b2=(0 if r is None else min(len(np.unique(r)),9), 0 if r is None else int(np.sign(r.size-i.size))+1, min(len(p),LMAX))
        nonlocal best
        if b2 not in grid or grid[b2][0]<f: grid[b2]=(f,p)
        if f>best[0]: best=(f,p)
        return f
    for _ in range(48):
        consider([random.choices(names,weights=[weights.get(n,1.0) for n in names])[0] for _ in range(random.randint(1,3))])
    t0=time.time()
    while time.time()-t0<budget and best[0]<1.0:
        vals=list(grid.values()); _,par=random.choice(vals); p=list(par); k=random.random()
        def pk(): return random.choices(names,weights=[weights.get(n,1.0) for n in names])[0]
        if k<0.3 and len(p)<LMAX: p.insert(random.randint(0,len(p)),pk())
        elif k<0.55 and len(p)>1: del p[random.randrange(len(p))]
        elif k<0.85 and p: p[random.randrange(len(p))]=pk()
        else: p.append(pk())
        if random.random()<0.25 and len(vals)>1:
            _,oth=random.choice(vals); cut=random.randint(1,len(p)); p=(p[:cut]+oth)[:LMAX]
        consider(p[:LMAX])
    return best

def wake(files, fns, names, weights, budget):
    solved={}; sols={}
    for fp in files:
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        local=dict(fns); lnames=list(names); cm=PU.infer_cmap(train)
        if cm is not None: local['cmap']=lambda g,cm=cm:PU.cmap(g,cm); lnames=lnames+['cmap']
        f,prog=evolve_solve(train,lnames,local,weights,budget)
        if prog is not None and f>=1.0-1e-9:
            if all((lambda r,o:r is not None and r.shape==o.shape and np.array_equal(r,o))(apply_prog(prog,local,i),o) for i,o in train):
                outs=[apply_prog(prog,local,xi) for xi,_ in tests]
                if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
                    solved[name]=prog; sols[name]=[x for x in prog if x!='cmap']
                    for nm in prog: weights[nm]+=1.0
    return solved, sols

def sleep_abstract(sols, fns, names, kmax=6):
    # criterio DreamCoder-like: subsequencia reusada por MAIS TAREFAS DISTINTAS (generalizacao), nao freq bruta
    bytask=defaultdict(set)
    for task,prog in sols.items():
        for L in (2,3,4):
            for i in range(len(prog)-L+1):
                bytask[tuple(prog[i:i+L])].add(task)
    cand=sorted(bytask.items(), key=lambda kv:(-len(kv[1]),-len(kv[0])))
    added=[]
    for seq,tasks in cand:
        if len(tasks)>=2 and len(added)<kmax and all(s in fns for s in seq):
            nm=f"M{len(names)+len(added)}"; added.append((nm,seq,len(tasks)))
    for nm,seq,_ in added:
        fns[nm]=make_macro(list(seq),fns); names.append(nm)
    return added

def main():
    t0=time.time()
    budget=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--budget=')),'1.0'))
    iters=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--iters=')),'3'))
    h93=set(json.load(open(os.path.join(HERE,"h93_zealot_solved.json")))) if os.path.exists(os.path.join(HERE,"h93_zealot_solved.json")) else set()
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    h89=set(json.load(open(os.path.join(HERE,"h89_evo_solved.json")))) if os.path.exists(os.path.join(HERE,"h89_evo_solved.json")) else set()
    ceiling=union|h89|h93   # teto atual ~91
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    fns=seed_dsl(); names=list(fns); weights=Counter()
    print(f"H98 DreamCoder no ARC real | DSL semente={len(names)} ops | teto atual={len(ceiling)} | iters={iters} budget={budget}s")
    all_solved=set(); hist=[]
    for it in range(iters):
        solved,sols=wake(files,fns,names,weights,budget)
        all_solved|=set(solved)
        added=sleep_abstract(sols,fns,names)
        novas=all_solved-ceiling
        hist.append((it,len(solved),len(all_solved),len(novas),len(names),len(added)))
        print(f"  iter {it}: wake resolveu {len(solved)} | acumulado {len(all_solved)} | NOVAS vs teto {len(novas)} | DSL={len(names)} (+{len(added)} macros)")
        if added: print(f"           macros novos (reuso entre tarefas): {[(nm,'+'.join(seq),n) for nm,seq,n in added]}")
        sys.stdout.flush()
    novas=all_solved-ceiling
    json.dump(sorted(all_solved),open(os.path.join(HERE,"h98_dreamcoder_solved.json"),"w"))
    print("="*74)
    print(f"H98 — {time.time()-t0:.0f}s | DSL cresceu {len(seed_dsl())} -> {len(names)} ops")
    print(f"  total resolvido: {len(all_solved)} | NOVAS vs teto(91): {len(novas)} {sorted(novas)[:14]}")
    print(f"  novo TETO: {len(ceiling|all_solved)}/400 = {100*len(ceiling|all_solved)/400:.1f}%")
    comp = len(hist)>=2 and hist[-1][2]>hist[0][2]
    print(f"  COMPOSTOU? {'SIM — DSL crescido destravou novas (chama!)' if comp and len(novas)>0 else 'saturou (recombinacao nao gera familia nova)'}")
    print("  Honesto: se NOVAS>0 via macro auto-abstraido de tarefas REAIS, a maquina cresceu o proprio")
    print("  vocabulario e furou a barreira SEM codar a mao. Se 0, confirma: a familia nova precisa vir de fora.")

if __name__=="__main__": main()

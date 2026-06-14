"""
H89 — "Deus darwiniano": sintese evolutiva de programas com DIVERSIDADE (MAP-Elites)
=====================================================================================
A ideia do usuario, destilada para algo REAL e que roda numa 3050 (busca simbolica =
CPU; AGI nao "escala parametros", escala IDEIAS evolutivas):
  - genoma = um PROGRAMA (sequencia de primitivas do nosso DSL de ARC);
  - selecao natural artificial: fitness GRADUADO (acuracia por celula nos demos) da gradiente
    onde "resolveu/nao resolveu" e esparso demais;
  - DIVERSIDADE preservada (MAP-Elites): nichos por descritor de comportamento -> mutacao
    inutil hoje vira breakthrough depois (preserva diversidade cognitiva, nao "o melhor");
  - FRACTALIZACAO: clona elites e muta em varias direcoes (add/del/swap/append/crossover);
  - "a evolucao tambem evolui": PRIORES DE OPERADOR auto-adaptativos — o sistema APRENDE,
    entre tarefas, quais primitivas tendem a compor solucoes e enviesa a mutacao por elas
    (inteligencia guiando a propria evolucao; o "deus" que destina a selecao).

TESTE HONESTO: a busca FIXA ja existente (pulse depth-4 + induce depth-2) cobre composicoes
rasas. A evolucao alcanca composicoes MAIS PROFUNDAS e parametricas. Mede: resolve algo ALEM
da uniao (84) que a busca fixa NAO alcanca? Se sim, profundidade/diversidade era gargalo real
(eixo novo, nao a reordenacao nula do H62). Se nao, o teto e largura de primitiva (Solomonoff),
nao busca. Gate exato; CPU; orcamento de tempo por tarefa. Nao e AGI — e o motor que a natureza usou.
"""
import os, sys, json, glob, time, random, numpy as np
from collections import Counter
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H63=_load("exp_h63_efficiency_breadth.py"); PU=H63.PU
random.seed(0); np.random.seed(0)

def build_pool(train):
    pool=dict(PU.FIXED)
    if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(PU.square_ops())
    pool.update(H63.NEW)
    cm=PU.infer_cmap(train)
    if cm is not None: pool['cmap']=lambda g,cm=cm: PU.cmap(g,cm)
    return pool

def apply_prog(prog, fns, g):
    cur=g
    try:
        for nm in prog:
            cur=fns[nm](cur)
            if cur is None or not isinstance(cur,np.ndarray) or cur.size==0: return None
        return cur
    except Exception: return None

def score_prog(prog, fns, demos):
    tot=0.0
    for i,o in demos:
        r=apply_prog(prog,fns,i)
        if r is None: return 0.0
        if r.shape==o.shape: tot+=float((r==o).mean())
        else:
            dh=abs(r.shape[0]-o.shape[0]); dw=abs(r.shape[1]-o.shape[1])
            tot+=0.2*max(0.0,1-(dh+dw)/max(1,o.shape[0]+o.shape[1]))
    return tot/len(demos)

def behavior(prog, fns, demos, L):
    i,o=demos[0]; r=apply_prog(prog,fns,i)
    if r is None: return (9,9,min(len(prog),L))
    ds=int(np.sign(r.size-i.size))+1                  # 0/1/2: encolhe/igual/cresce
    ncol=min(int(len(np.unique(r))),9)
    return (ds,ncol,min(len(prog),L))

def mutate(prog, names, weights, L):
    p=list(prog); op=random.random()
    def pick():
        return random.choices(names,weights=[weights.get(n,1.0) for n in names])[0]
    if op<0.30 and len(p)<L: p.insert(random.randint(0,len(p)),pick())   # insere
    elif op<0.55 and len(p)>1: del p[random.randrange(len(p))]            # remove
    elif op<0.85 and p: p[random.randrange(len(p))]=pick()               # troca
    else: p.append(pick())                                               # append
    return p[:L]

def crossover(a,b,L):
    if not a or not b: return (a or b)[:L]
    ca=a[:random.randint(1,len(a))]+b[random.randint(0,len(b)-1):]
    return ca[:L] if ca else a[:L]

def evolve(train, names, fns, weights, budget, L=8, pop=64):
    grid={}
    def consider(prog):
        if not prog: return 0.0
        f=score_prog(prog,fns,train); bd=behavior(prog,fns,train,L)
        if bd not in grid or grid[bd][0]<f: grid[bd]=(f,prog)
        return f
    best=(0.0,None)
    for _ in range(pop):
        prog=[random.choices(names,weights=[weights.get(n,1.0) for n in names])[0] for _ in range(random.randint(1,3))]
        f=consider(prog)
        if f>best[0]: best=(f,prog)
    t0=time.time()
    while time.time()-t0<budget and best[0]<1.0:
        vals=list(grid.values()); _,parent=random.choice(vals)
        child=mutate(parent,names,weights,L)
        if random.random()<0.3 and len(vals)>1:
            _,other=random.choice(vals); child=crossover(child,other,L)
        f=consider(child)
        if f>best[0]: best=(f,child)
    return best

def main():
    t0=time.time()
    budget=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--budget=')),'1.2'))
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    weights=Counter()                      # priores de operador auto-adaptativos (entre tarefas)
    solved=set(); depths=[]
    for k,fp in enumerate(files):
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        pool=build_pool(train); names=list(pool)
        f,prog=evolve(train,names,pool,weights,budget)
        if prog is not None and f>=1.0-1e-9:
            # gate exato nos demos
            if all((lambda r,o:r is not None and r.shape==o.shape and np.array_equal(r,o))(apply_prog(prog,pool,i),o) for i,o in train):
                outs=[apply_prog(prog,pool,xi) for xi,_ in tests]
                if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
                    solved.add(name); depths.append(len(prog))
                    for nm in prog: weights[nm]+=1.0     # aprende: ops que resolvem ganham peso
        if (k+1)%50==0:
            print(f"  ... {k+1}/400 (solved={len(solved)}, NOVAS={len(solved-union)})"); sys.stdout.flush()
    json.dump(sorted(solved),open(os.path.join(HERE,"h89_evo_solved.json"),"w"))
    novas=solved-union
    top=Counter(weights).most_common(12)
    print("="*72)
    print(f"H89 Sintese evolutiva de programas (MAP-Elites + priores auto-adaptativos)")
    print(f"  budget/tarefa={budget}s | union(H85)={len(union)}")
    print(f"  evolucao resolveu: {len(solved)}/400 = {100*len(solved)/400:.1f}%")
    print(f"  NOVAS vs union(busca fixa): {len(novas)} {sorted(novas)[:14]}")
    print(f"  nova UNIAO: {len(union|solved)}/400 = {100*len(union|solved)/400:.1f}%  (era {100*len(union)/400:.1f}%)")
    if depths: print(f"  profundidade dos programas que resolveram: media {np.mean(depths):.1f} max {max(depths)} (busca fixa ia ate ~4)")
    print(f"  ops mais uteis aprendidos: {[(n,int(w)) for n,w in top]}")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: se NOVAS>0, profundidade/diversidade evolutiva era gargalo real (eixo novo).")
    print("  Se NOVAS=0, o teto e LARGURA de primitiva (Solomonoff), nao a busca — confirma H62/H63.")

if __name__=="__main__": main()

"""
H63 — A hipotese do usuario: a eficiencia do coordenador (CX) VIRA cobertura no ARC?
=====================================================================================
Dois efeitos separaveis, testados com dado:
 (A) EFEITO COORDENACAO: com orcamento FIXO por tarefa, o CX prioriza as primitivas certas
     primeiro -> recupera solves dentro do orcamento que a ordem ingenua perderia. Mede a
     curva cobertura-vs-orcamento (CX vs fixo vs leaky) a partir do cache do H62.
 (B) EFEITO LARGURA: adiciona um LOTE de primitivas novas (relacionais/parametricas) ao pool
     e mede a cobertura EXAUSTIVA nas 400 — o lever real. Compara com o pool atual (H54).
Sintese honesta: largura adiciona solves (real); coordenacao mantem isso tratavel; mas
cobertura exaustiva e o teto — coordenacao so importa SE o orcamento for o gargalo.
"""
import os, json, glob, numpy as np
from collections import deque, Counter
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PU=_load("exp_h54_pulse.py")            # FIXED, comps, grav, infer_cmap, cmap, square_ops
AB=_load("exp_h55_abduce.py")           # abduce (para a uniao)
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

# ============ (A) curva cobertura-vs-orcamento (efeito coordenacao) ============
def coverage_budget():
    cache=os.path.join(os.path.dirname(__file__),"h62_labels.json")
    data=[(np.array(f),op) for f,op in json.load(open(cache))]
    prims=sorted(set(op for _,op in data)); P=len(prims)
    feats=np.array([f for f,_ in data])
    # fluxo por similaridade (como H62)
    order=[0]; used={0}
    while len(order)<len(data):
        d=np.linalg.norm(feats-feats[order[-1]],axis=1)
        for u in used: d[u]=1e9
        order.append(int(d.argmin())); used.add(order[-1])
    stream=[data[i] for i in order]
    # embute operadores -> anel (como H62)
    import importlib
    FC=_load("exp_h57_flycoord.py"); N=FC.N; TH=FC.TH
    pf={p:np.mean([f for f,op in data if op==p],0) for p in prims}
    M=np.array([pf[p] for p in prims]); M=M-M.mean(0)
    _,_,vt=np.linalg.svd(M,full_matrices=False); proj=M@vt[:2].T
    ang={p:np.arctan2(proj[k,1],proj[k,0]) for k,p in enumerate(prims)}
    def costs(kind):
        out=[]
        if kind=='cx':
            W=FC.cx_connectivity(); r=np.zeros(N)
            w2=lambda a:int(round((a%(2*np.pi))/(2*np.pi/N)))%N
            for f,win in stream:
                sc={p:r[w2(ang[p])] for p in prims}
                out.append(sorted(prims,key=lambda p:-sc[p]).index(win)+1)
                wd=w2(ang[win])
                for _ in range(4):
                    r=r+0.5*(-r+np.maximum(0.0,(W@r)+FC.vonmises_input(TH[wd]))); s=r.sum(); r=r*(6/s) if s>6 else r
        elif kind=='leaky':
            w={p:0.0 for p in prims}
            for f,win in stream:
                out.append(sorted(prims,key=lambda p:-w[p]).index(win)+1)
                for p in w:w[p]*=0.8
                w[win]+=1
        else:
            u=Counter()
            for f,win in stream:
                out.append(sorted(prims,key=lambda p:-u[p]).index(win)+1); u[win]+=1
        return np.array(out)
    cc,cl,cf=costs('cx'),costs('leaky'),costs('fixed')
    print("(A) EFEITO COORDENACAO — cobertura(%) atingida com orcamento B (nº ops tentadas/tarefa):")
    print(f"    {'B':>4} | {'fixo':>6} {'leaky':>6} {'CX':>6}")
    for B in (2,4,6,8,12,P):
        print(f"    {B:>4} | {100*np.mean(cf<=B):>5.0f}% {100*np.mean(cl<=B):>5.0f}% {100*np.mean(cc<=B):>5.0f}%")
    print(f"    => CX atinge cobertura plena com orcamento ~{int(np.ceil(cc.max()))} vs fixo ~{int(np.ceil(cf.max()))} "
          f"(libera ~{cf.mean()/cc.mean():.1f}x de orcamento).\n")

# ============ (B) efeito LARGURA: primitivas novas, cobertura exaustiva ============
def comps(g): return PU.comps(g)
def grav_dir(g,dy,dx):
    o=np.zeros_like(g)
    if dy==1:    # baixo
        for c in range(g.shape[1]):
            col=[v for v in g[:,c] if v!=0]; o[g.shape[0]-len(col):,c]=col
    elif dy==-1: # cima
        for c in range(g.shape[1]):
            col=[v for v in g[:,c] if v!=0]; o[:len(col),c]=col
    elif dx==1:  # direita
        for r in range(g.shape[0]):
            row=[v for v in g[r] if v!=0]; o[r,g.shape[1]-len(row):]=row
    else:        # esquerda
        for r in range(g.shape[0]):
            row=[v for v in g[r] if v!=0]; o[r,:len(row)]=row
    return o
def connect(g,axis):
    o=g.copy(); lines=g if axis==1 else g.T; oo=o if axis==1 else o.T
    for r in range(lines.shape[0]):
        for c in set(lines[r][lines[r]!=0]):
            pos=np.where(lines[r]==c)[0]
            if len(pos)>=2: oo[r,pos.min():pos.max()+1]=c
    return o
def dedup(g,axis):
    if axis==0:
        rows=[g[0]]; [rows.append(g[i]) for i in range(1,g.shape[0]) if not np.array_equal(g[i],g[i-1])]
        return np.array(rows)
    return dedup(g.T,0).T
def border_objs(g,keep):
    lab,n=comps(g); o=np.zeros_like(g); H,W=g.shape
    for i in range(1,n+1):
        ys,xs=np.where(lab==i); touch=ys.min()==0 or xs.min()==0 or ys.max()==H-1 or xs.max()==W-1
        if touch==keep: o[lab==i]=g[lab==i]
    return o
def swap2(g):
    vals=g[g!=0]
    if len(vals)==0: return g.copy()
    v,c=np.unique(vals,return_counts=True)
    if len(v)<2: return g.copy()
    a,b=v[np.argsort(c)[::-1][:2]]; o=g.copy(); o[g==a]=b; o[g==b]=a; return o
def sym4(g):
    if g.shape[0]!=g.shape[1]: return None
    o=g.copy()
    for k in (1,2,3):
        m=np.rot90(g,k); o[o==0]=m[o==0]
    return o

NEW={'grav_u':lambda g:grav_dir(g,-1,0),'grav_l':lambda g:grav_dir(g,0,-1),'grav_r':lambda g:grav_dir(g,0,1),
 'connect_h':lambda g:connect(g,1),'connect_v':lambda g:connect(g,0),
 'dedup_rows':lambda g:dedup(g,0),'dedup_cols':lambda g:dedup(g,1),
 'rm_border':lambda g:border_objs(g,False),'keep_border':lambda g:border_objs(g,True),
 'swap2':swap2,'sym4':sym4}

def induce(train, pool):
    names=list(pool); fns=pool
    def ok(f):
        try:
            for i,o in train:
                r=f(i)
                if r is None or not isinstance(r,np.ndarray) or r.shape!=o.shape or not np.array_equal(r,o): return False
            return True
        except Exception: return False
    for nm in names:
        if ok(fns[nm]): return fns[nm]
    cm=PU.infer_cmap(train)
    if cm is not None and ok(lambda g:PU.cmap(g,cm)): return lambda g:PU.cmap(g,cm)
    for n1 in names:
        f1=fns[n1]
        try:
            t2=[(f1(i),o) for i,o in train if f1(i) is not None]
            if len(t2)==len(train):
                cm2=PU.infer_cmap(t2)
                if cm2 is not None and ok(lambda g,f1=f1,cm2=cm2:None if f1(g) is None else PU.cmap(f1(g),cm2)):
                    return lambda g,f1=f1,cm2=cm2:None if f1(g) is None else PU.cmap(f1(g),cm2)
            for n2 in names:
                f2=fns[n2]
                if ok(lambda g,f1=f1,f2=f2:None if f1(g) is None else f2(f1(g))):
                    return lambda g,f1=f1,f2=f2:None if f1(g) is None else f2(f1(g))
        except Exception: pass
    return None

def coverage(pool_extra):
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); solved=set()
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        pool=dict(PU.FIXED)
        if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(PU.square_ops())
        pool.update(pool_extra)
        prog=induce(train,pool)
        if prog is None: continue
        if all((lambda r,yo:r is not None and r.shape==yo.shape and np.array_equal(r,yo))(prog(xi),yo) for xi,yo in tests):
            solved.add(os.path.basename(fp))
    return solved

def main():
    print("="*70); print("H63 — a eficiencia do CX vira cobertura no ARC? (efeitos separados)\n")
    coverage_budget()
    print(f"(B) EFEITO LARGURA — primitivas novas (+{len(NEW)}: grav 3-dir, connect, dedup, border-objs, swap2, sym4):")
    base=coverage({}); ext=coverage(NEW)
    print(f"    pool atual (H54)        : {len(base)}/400 = {100*len(base)/400:.1f}%")
    print(f"    pool + {len(NEW)} primitivas : {len(ext)}/400 = {100*len(ext)/400:.1f}%  (delta {len(ext)-len(base):+d})")
    novas=ext-base
    print(f"    tarefas NOVAS pela largura: {len(novas)} {sorted(novas)[:12]}")
    # uniao com ABDUCE
    print("\n  Uniao com ABDUCE (Tier A/B/C objeto-centrico):")
    ab=set()
    for fp in sorted(glob.glob(os.path.join(DATA,"*.json"))):
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        r,_,_=AB.abduce(train)
        if r is not None and all((lambda o,yo:o is not None and isinstance(o,np.ndarray) and o.shape==yo.shape and np.array_equal(o,yo))(r(xi),yo) for xi,yo in tests):
            ab.add(os.path.basename(fp))
    union=ext|ab
    print(f"    UNIAO (pool-estendido + ABDUCE): {len(union)}/400 = {100*len(union)/400:.1f}%  (era 12,5% = 50/400)")
    print("\n  HONESTO: (A) coordenacao libera orcamento (so importa se orcamento e gargalo);")
    print("  (B) LARGURA e o que move a cobertura. Se a uniao subir, foi a largura nova — o CX")
    print("  apenas a deixaria mais barata. Coordenacao != capacidade nova; a parede e largura.")

if __name__=="__main__": main()

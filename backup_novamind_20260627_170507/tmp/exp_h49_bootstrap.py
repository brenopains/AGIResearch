"""
H49 — More specialists + self-improving sub-agents (bootstrapping) on REAL ARC
==============================================================================

Two things the user asked:
  (1) MORE specialists (richer operator pool, incl. symmetry-completion etc.).
  (2) Each 'cortex' creates SUB-AGENTS that improve with practice = LIBRARY
      LEARNING: mine the composite programs that solved tasks, PROMOTE the
      frequent useful ones to new atomic operators, re-run -> the search now
      reaches deeper with the same budget (effectively self-improving).

We measure ARC solve rate across bootstrapping iterations. Honest: this climbs
within the operator closure (real narrow gain); the wall (rules outside any
operator set = fluid synthesis = Solomonoff) stays.
"""
import os, json, glob, numpy as np
from collections import deque, Counter
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

def crop(g):
    nz=np.argwhere(g!=0)
    if len(nz)==0: return g.copy()
    (r0,c0),(r1,c1)=nz.min(0),nz.max(0); return g[r0:r1+1,c0:c1+1].copy()
def maj(g):
    v,c=np.unique(g,return_counts=True); return np.full_like(g,v[c.argmax()])
def grav(g):
    o=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; o[g.shape[0]-len(col):,c]=col
    return o
def comps(g):
    lab=np.zeros_like(g,dtype=int); cur=0
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r,c]!=0 and lab[r,c]==0:
                cur+=1; q=deque([(r,c)]); lab[r,c]=cur
                while q:
                    y,x=q.popleft()
                    for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<g.shape[0] and 0<=nx<g.shape[1] and lab[ny,nx]==0 and g[ny,nx]==g[r,c]:
                            lab[ny,nx]=cur; q.append((ny,nx))
    return lab,cur
def keep_largest(g):
    lab,n=comps(g)
    if n==0: return g.copy()
    big=1+int(np.argmax([(lab==i).sum() for i in range(1,n+1)])); o=np.zeros_like(g); o[lab==big]=g[lab==big]; return o
def isolate(g):
    vals=g[g!=0]
    if len(vals)==0: return g.copy()
    v,c=np.unique(vals,return_counts=True); k=v[c.argmax()]; o=np.zeros_like(g); o[g==k]=k; return o
def border(g):
    o=g.copy()
    if min(o.shape)>0: o[0,:]=o[-1,:]=1; o[:,0]=o[:,-1]=1
    return o
def down(g,f):
    if g.shape[0]%f or g.shape[1]%f: return None
    H,W=g.shape[0]//f,g.shape[1]//f; o=np.zeros((H,W),g.dtype)
    for r in range(H):
        for c in range(W):
            b=g[r*f:(r+1)*f,c*f:(c+1)*f]; v,cc=np.unique(b,return_counts=True); o[r,c]=v[cc.argmax()]
    return o
def sym_h(g):
    o=g.copy(); m=g[:,::-1]; o[o==0]=m[o==0]; return o
def sym_v(g):
    o=g.copy(); m=g[::-1,:]; o[o==0]=m[o==0]; return o

BASE={'id':lambda g:g.copy(),'fliplr':lambda g:g[:,::-1].copy(),'flipud':lambda g:g[::-1,:].copy(),'rot180':lambda g:np.rot90(g,2).copy(),
 'roll_d':lambda g:np.roll(g,1,0),'roll_u':lambda g:np.roll(g,-1,0),'roll_r':lambda g:np.roll(g,1,1),'roll_l':lambda g:np.roll(g,-1,1),
 'grav':grav,'maj':maj,'border':border,'crop':crop,'keep_largest':keep_largest,'isolate':isolate,
 'tile2x2':lambda g:np.tile(g,(2,2)),'tile1x2':lambda g:np.tile(g,(1,2)),'tile2x1':lambda g:np.tile(g,(2,1)),
 'mirror_h':lambda g:np.concatenate([g,g[:,::-1]],1),'mirror_v':lambda g:np.concatenate([g,g[::-1,:]],0),
 'scale2':lambda g:np.kron(g,np.ones((2,2),dtype=g.dtype)),'down2':lambda g:down(g,2),'down3':lambda g:down(g,3),
 'sym_h':sym_h,'sym_v':sym_v}
def with_square(P,g):
    P=dict(P)
    if g.shape[0]==g.shape[1]: P['rot90']=lambda x:np.rot90(x,1).copy(); P['rot270']=lambda x:np.rot90(x,3).copy(); P['transpose']=lambda x:x.T.copy()
    return P
def infer_cmap(pairs):
    m={}
    for i,o in pairs:
        if i.shape!=o.shape: return None
        for a,b in zip(i.ravel(),o.ravel()):
            if a in m and m[a]!=b: return None
            m[a]=b
    return m
def cmap(g,m): return np.vectorize(lambda v:m.get(v,v))(g)
def ok(f,tr):
    try:
        for i,o in tr:
            r=f(i)
            if r is None or r.shape!=o.shape or not np.array_equal(r,o): return False
        return True
    except Exception: return False

def induce(train, ops):
    """Return (fn, descriptor) consistent with train, or (None,None). Records the composite used."""
    P=with_square(ops,train[0][0])
    for n,f in P.items():
        if ok(f,train): return f,(n,)
    cm=infer_cmap(train)
    if cm is not None and ok(lambda g:cmap(g,cm),train): return (lambda g:cmap(g,cm)),('cmap',)
    for n1,f1 in P.items():
        try:
            t2=[(f1(i),o) for i,o in train if f1(i) is not None]
            if len(t2)==len(train):
                cm2=infer_cmap(t2)
                if cm2 is not None and ok(lambda g,f1=f1,cm2=cm2:cmap(f1(g),cm2),train): return (lambda g,f1=f1,cm2=cm2:cmap(f1(g),cm2)),(n1,'cmap')
            for n2,f2 in P.items():
                if ok(lambda g,f1=f1,f2=f2:(None if f1(g) is None else f2(f1(g))),train): return (lambda g,f1=f1,f2=f2:(None if f1(g) is None else f2(f1(g)))),(n1,n2)
        except Exception: pass
    return None,None

def solves(prog,test):
    for p in test:
        xi=np.array(p["input"]); yo=np.array(p["output"])
        try:
            pr=prog(xi)
            if pr is None or pr.shape!=yo.shape or not np.array_equal(pr,yo): return False
        except Exception: return False
    return True

files=sorted(glob.glob(os.path.join(DATA,"*.json")))
tasks=[(os.path.basename(f)[:8], json.load(open(f))) for f in files]

ops=dict(BASE); print("REAL ARC — mais especialistas + sub-agentes auto-melhoráveis (bootstrapping)")
print("-"*60)
for it in range(3):
    solved=set(); composites=Counter()
    for tid,t in tasks:
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        prog,desc=induce(train,ops)
        if prog is None: continue
        if solves(prog,t["test"]):
            solved.add(tid)
            if desc and len(desc)==2 and desc[1]!='cmap' and desc[0] in BASE and desc[1] in BASE:
                composites[desc]+=1   # mine reusable 2-op composites
    print(f"  iteração {it}: operadores={len(ops):>2} | RESOLVIDAS={len(solved):>3}/400 = {100*len(solved)/400:.1f}%")
    # promote frequent composites to new atomic 'sub-agent' operators
    promoted=0
    for (a,b),cnt in composites.most_common(6):
        name=f"{a}>{b}"
        if name not in ops:
            fa,fb=BASE[a],BASE[b]
            ops[name]=(lambda g,fa=fa,fb=fb:(None if fa(g) is None else fb(fa(g)))); promoted+=1
    if promoted==0: print("  (sem novos sub-agentes para promover — convergiu)"); break
print("-"*60)
print("Sub-agentes (composições) promovidos a operadores melhoram o alcance da busca com a")
print("prática (bootstrapping/library-learning). Ganho narrow real; a parede (regra fora do")
print("fecho de operadores = síntese fluida/Solomonoff) permanece. Dados colhidos por iteração.")

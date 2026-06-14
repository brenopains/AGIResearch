"""
H48 — Five inducer variants + ensemble (Nash/coordination) on REAL ARC. Collect data.
=====================================================================================

Per the user: test multiple distinct hypotheses on the real ARC set, gather data,
even failures teach. We build 4 SPECIALIST inducers (each a different operator
family + color), measure each on the 400 ARC tasks, and test the COORDINATION idea
(ensemble = a task is solved if ANY specialist solves it). Reports per-specialist
solve rate, ensemble (union), and which tasks each catches.
"""
import os, json, glob, numpy as np
from collections import deque
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

SPATIAL={'id':lambda g:g.copy(),'fliplr':lambda g:g[:,::-1].copy(),'flipud':lambda g:g[::-1,:].copy(),'rot180':lambda g:np.rot90(g,2).copy(),
         'roll_d':lambda g:np.roll(g,1,0),'roll_u':lambda g:np.roll(g,-1,0),'roll_r':lambda g:np.roll(g,1,1),'roll_l':lambda g:np.roll(g,-1,1)}
OBJECTS={'id':lambda g:g.copy(),'keep_largest':keep_largest,'isolate':isolate,'crop':crop}
SIZE={'id':lambda g:g.copy(),'crop':crop,'tile2x2':lambda g:np.tile(g,(2,2)),'tile1x2':lambda g:np.tile(g,(1,2)),'tile2x1':lambda g:np.tile(g,(2,1)),
      'mirror_h':lambda g:np.concatenate([g,g[:,::-1]],1),'mirror_v':lambda g:np.concatenate([g,g[::-1,:]],0),
      'scale2':lambda g:np.kron(g,np.ones((2,2),dtype=g.dtype)),'down2':lambda g:down(g,2),'down3':lambda g:down(g,3)}
PHYSICS={'id':lambda g:g.copy(),'grav':grav,'maj':maj,'border':border}
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
def ok(f,train):
    try:
        for i,o in train:
            r=f(i)
            if r is None or r.shape!=o.shape or not np.array_equal(r,o): return False
        return True
    except Exception: return False

def induce(train, base):
    P=with_square(base,train[0][0])
    for f in P.values():
        if ok(f,train): return f
    cm=infer_cmap(train)
    if cm is not None and ok(lambda g:cmap(g,cm),train): return lambda g:cmap(g,cm)
    for f1 in P.values():
        try:
            t2=[(f1(i),o) for i,o in train if f1(i) is not None]
            if len(t2)==len(train):
                cm2=infer_cmap(t2)
                if cm2 is not None and ok(lambda g,f1=f1,cm2=cm2:cmap(f1(g),cm2),train): return lambda g,f1=f1,cm2=cm2:cmap(f1(g),cm2)
            for f2 in P.values():
                if ok(lambda g,f1=f1,f2=f2:(None if f1(g) is None else f2(f1(g))),train): return lambda g,f1=f1,f2=f2:(None if f1(g) is None else f2(f1(g)))
        except Exception: pass
    return None

files=sorted(glob.glob(os.path.join(DATA,"*.json")))
SPECIALISTS={'spatial':SPATIAL,'objects':OBJECTS,'size':SIZE,'physics':PHYSICS}
solved={k:set() for k in SPECIALISTS}
for fp in files:
    t=json.load(open(fp)); name=os.path.basename(fp)[:8]
    train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
    for sname,base in SPECIALISTS.items():
        prog=induce(train,base)
        if prog is None: continue
        good=True
        for p in t["test"]:
            xi=np.array(p["input"]); yo=np.array(p["output"])
            try:
                pr=prog(xi)
                if pr is None or pr.shape!=yo.shape or not np.array_equal(pr,yo): good=False; break
            except Exception: good=False; break
        if good: solved[sname].add(name)

ensemble=set().union(*solved.values())
print("REAL ARC-AGI (400) — 5 hipóteses: 4 especialistas + ENSEMBLE (coordenação/Nash)")
print("-"*56)
for k,v in solved.items(): print(f"  especialista {k:>8}: {len(v):>3}/400 = {100*len(v)/400:.1f}%")
print(f"  ENSEMBLE (união) : {len(ensemble):>3}/400 = {100*len(ensemble)/400:.1f}%")
print("-"*56)
# what coordination adds beyond the best single
best=max(solved.values(),key=len)
print(f"melhor especialista sozinho: {len(best)}; ensemble: {len(ensemble)} (+{len(ensemble)-len(best)} tarefas pela coordenação)")
print("Dado honesto: especialistas diversos pegam subconjuntos diferentes; a UNIÃO resolve mais")
print("(coordenação ajuda, narrow). Mesmo o ensemble fica MUITO abaixo do SOTA/humano -- a parede")
print("e a regra fora de QUALQUER conjunto de operadores (síntese fluida = Solomonoff, incomputável).")

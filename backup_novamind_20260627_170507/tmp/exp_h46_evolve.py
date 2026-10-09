"""
H46 — Evolve the inducer: more operators (re-representation) on REAL ARC
=======================================================================

The user's insight: humans RE-REPRESENT problems (objects, symmetry, summaries)
to make them digestible. We add operators that 're-see' the grid -- connected
components (objects), keep-largest, color-isolate, downscale (mode-pool) -- plus
the H44 set, with simplicity-first (MDL) search. Re-measure on the real ARC-AGI
training set. Honest: this is real narrow climbing (more operators -> more tasks
in the closure), not beating SOTA on a 3050.
"""
import os, json, glob, numpy as np
from collections import deque
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

def crop_content(g):
    nz=np.argwhere(g!=0)
    if len(nz)==0: return g.copy()
    (r0,c0),(r1,c1)=nz.min(0),nz.max(0); return g[r0:r1+1,c0:c1+1].copy()
def majority_fill(g):
    v,c=np.unique(g,return_counts=True); return np.full_like(g,v[c.argmax()])
def gravity_v(g):
    o=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; o[g.shape[0]-len(col):,c]=col
    return o
def components(g):
    """4-connectivity connected components of equal nonzero color."""
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
    lab,n=components(g)
    if n==0: return g.copy()
    sizes=[(lab==i).sum() for i in range(1,n+1)]; big=1+int(np.argmax(sizes))
    o=np.zeros_like(g); o[lab==big]=g[lab==big]; return o
def color_isolate(g):
    vals=g[g!=0]
    if len(vals)==0: return g.copy()
    v,c=np.unique(vals,return_counts=True); keep=v[c.argmax()]
    o=np.zeros_like(g); o[g==keep]=keep; return o
def downscale(g,f):
    if g.shape[0]%f or g.shape[1]%f: return None
    H,W=g.shape[0]//f,g.shape[1]//f; o=np.zeros((H,W),dtype=g.dtype)
    for r in range(H):
        for c in range(W):
            block=g[r*f:(r+1)*f,c*f:(c+1)*f]; v,cc=np.unique(block,return_counts=True); o[r,c]=v[cc.argmax()]
    return o

def prims(g):
    P={'id':lambda x:x.copy(),'fliplr':lambda x:x[:,::-1].copy(),'flipud':lambda x:x[::-1,:].copy(),
       'rot180':lambda x:np.rot90(x,2).copy(),'grav_v':gravity_v,'maj_fill':majority_fill,
       'crop':crop_content,'keep_largest':keep_largest,'color_isolate':color_isolate,
       'tile2x2':lambda x:np.tile(x,(2,2)),'mirror_h':lambda x:np.concatenate([x,x[:,::-1]],1),
       'mirror_v':lambda x:np.concatenate([x,x[::-1,:]],0),'scale2':lambda x:np.kron(x,np.ones((2,2),dtype=x.dtype)),
       'down2':lambda x:downscale(x,2),'down3':lambda x:downscale(x,3)}
    if g.shape[0]==g.shape[1]:
        P['rot90']=lambda x:np.rot90(x,1).copy(); P['rot270']=lambda x:np.rot90(x,3).copy(); P['transpose']=lambda x:x.T.copy()
    return P
def infer_cmap(pairs):
    m={}
    for i,o in pairs:
        if i.shape!=o.shape: return None
        for a,b in zip(i.ravel(),o.ravel()):
            if a in m and m[a]!=b: return None
            m[a]=b
    return m
def apply_cmap(g,m): return np.vectorize(lambda v:m.get(v,v))(g)
def consistent(f,train):
    try:
        for i,o in train:
            r=f(i)
            if r is None or r.shape!=o.shape or not np.array_equal(r,o): return False
        return True
    except Exception: return False
def induce(train):
    P=prims(train[0][0])
    for f in P.values():
        if consistent(f,train): return f
    cm=infer_cmap(train)
    if cm is not None and consistent(lambda g:apply_cmap(g,cm),train): return lambda g:apply_cmap(g,cm)
    for f1 in P.values():
        try:
            t2=[(f1(i),o) for i,o in train if f1(i) is not None]
            if len(t2)==len(train):
                cm2=infer_cmap(t2)
                if cm2 is not None:
                    f=lambda g,f1=f1,cm2=cm2:apply_cmap(f1(g),cm2)
                    if consistent(f,train): return f
            for f2 in P.values():
                f=lambda g,f1=f1,f2=f2:(None if f1(g) is None else f2(f1(g)))
                if consistent(f,train): return f
        except Exception: pass
    return None

files=sorted(glob.glob(os.path.join(DATA,"*.json"))); solved=0
for fp in files:
    t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
    prog=induce(train)
    if prog is None: continue
    ok=True
    for p in t["test"]:
        xi=np.array(p["input"]); yo=np.array(p["output"])
        try:
            pr=prog(xi)
            if pr is None or pr.shape!=yo.shape or not np.array_equal(pr,yo): ok=False; break
        except Exception: ok=False; break
    if ok: solved+=1
print(f"REAL ARC-AGI (400) — evolved inducer (objects/symmetry/downscale + MDL search)")
print(f"  SOLVED: {solved}/400 = {100*solved/400:.1f}%   (3.5% small DSL -> 7.0% H44 -> now)")
print(f"  Field SOTA ~30-55%; humans ~85%. Honest: real narrow climb via more operators.")

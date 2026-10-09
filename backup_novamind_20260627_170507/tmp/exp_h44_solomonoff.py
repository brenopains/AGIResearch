"""
H44 — Approximating Solomonoff on REAL ARC: bigger DSL + simplicity-first search
================================================================================

The legitimate, classical approximation of Solomonoff induction = Levin/MDL search:
enumerate programs from SIMPLE to complex, return the shortest one consistent with
the demos (Occam's razor = highest prior probability). We also EXPAND the DSL with
ARC-relevant primitives (tiling, cropping, scaling, mirroring, fills) so more tasks
become expressible. Then re-run on the real ARC-AGI training set.

This is the real way to raise the number (no quantum, no fractals, no magic). It is
narrow engineering toward the field's DSL solvers -- bounded by the DSL (the
Solomonoff-uncomputability wall: no fixed DSL covers all ARC).
"""
import os, json, glob, numpy as np
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

# ---- primitives (ordered roughly simple -> complex; search returns the first/simplest match) ----
def crop_content(g):
    nz=np.argwhere(g!=0)
    if len(nz)==0: return g.copy()
    (r0,c0),(r1,c1)=nz.min(0),nz.max(0); return g[r0:r1+1,c0:c1+1].copy()
def majority_fill(g):
    v,c=np.unique(g,return_counts=True); return np.full_like(g,v[c.argmax()])
def minority_fill(g):
    v,c=np.unique(g,return_counts=True); return np.full_like(g,v[c.argmin()])
def border(g):
    o=g.copy()
    if o.shape[0]>0: o[0,:]=o[-1,:]=1
    if o.shape[1]>0: o[:,0]=o[:,-1]=1
    return o
def gravity_v(g):
    o=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; o[g.shape[0]-len(col):,c]=col
    return o

PRIMS_BASE={  # shape-preserving
 'id':lambda g:g.copy(),'fliplr':lambda g:g[:,::-1].copy(),'flipud':lambda g:g[::-1,:].copy(),
 'rot180':lambda g:np.rot90(g,2).copy(),'grav_v':gravity_v,
 'maj_fill':majority_fill,'min_fill':minority_fill,'border':border}
PRIMS_SQUARE={'rot90':lambda g:np.rot90(g,1).copy(),'rot270':lambda g:np.rot90(g,3).copy(),'transpose':lambda g:g.T.copy()}
PRIMS_SHAPE={  # shape-changing (expand the space toward real ARC)
 'crop':crop_content,
 'tile2x2':lambda g:np.tile(g,(2,2)),'tile1x2':lambda g:np.tile(g,(1,2)),'tile2x1':lambda g:np.tile(g,(2,1)),
 'mirror_h':lambda g:np.concatenate([g,g[:,::-1]],1),'mirror_v':lambda g:np.concatenate([g,g[::-1,:]],0),
 'scale2':lambda g:np.kron(g,np.ones((2,2),dtype=g.dtype))}

def all_prims(g):
    P=dict(PRIMS_BASE); P.update(PRIMS_SHAPE)
    if g.shape[0]==g.shape[1]: P.update(PRIMS_SQUARE)
    return P

def infer_colormap(pairs):
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
            if r.shape!=o.shape or not np.array_equal(r,o): return False
        return True
    except Exception: return False

def induce(train):
    P=all_prims(train[0][0])
    # depth-1 (simplest first)
    for f in P.values():
        if consistent(f,train): return f
    cm=infer_colormap(train)
    if cm is not None and consistent(lambda g:apply_cmap(g,cm),train): return lambda g:apply_cmap(g,cm)
    # depth-2 (prim then prim, prim then colormap)
    for f1 in P.values():
        d2=[(f1(i),o) for i,o in train if True]
        try:
            cm2=infer_colormap([(f1(i),o) for i,o in train])
            if cm2 is not None:
                f=lambda g,f1=f1,cm2=cm2:apply_cmap(f1(g),cm2)
                if consistent(f,train): return f
        except Exception: pass
        for f2 in P.values():
            f=lambda g,f1=f1,f2=f2:f2(f1(g))
            if consistent(f,train): return f
    return None

files=sorted(glob.glob(os.path.join(DATA,"*.json")))
solved=0
for fp in files:
    t=json.load(open(fp))
    train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
    prog=induce(train)
    if prog is None: continue
    ok=True
    for p in t["test"]:
        xi=np.array(p["input"]); yo=np.array(p["output"])
        try:
            pred=prog(xi)
            if pred.shape!=yo.shape or not np.array_equal(pred,yo): ok=False; break
        except Exception: ok=False; break
    if ok: solved+=1

print(f"REAL ARC-AGI (400 tasks) — expanded DSL + simplicity-first search (Solomonoff approx)")
print(f"  SOLVED: {solved}/400 = {100*solved/400:.1f}%   (was 3.5% with the small DSL)")
print(f"  Field SOTA ~30-55%; humans ~85%.")
print(f"\nHonest: expanding the DSL + Occam search raises the number (real, narrow engineering).")
print(f"Still bounded by the DSL -- no fixed primitive set covers all ARC (Solomonoff is uncomputable).")

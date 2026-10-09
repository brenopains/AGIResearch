"""
H43 — Our rule-creating system on the REAL ARC-AGI dataset (400 training tasks)
==============================================================================

We run the H42 program-inducer (search over our DSL) on the actual ARC-AGI
training set. For each task it must INDUCE a program consistent with ALL train
pairs, then we check it on the test input(s). Honest, no cheating.

Our DSL is small (flips/rotations/transpose/rolls/color-map/gravity + depth-2),
so it can only solve the small subset of ARC whose rules live in this space.
Expected: a low single-digit %, consistent with simple DSL solvers. The field's
SOTA (huge DSLs + neural-guided search + TTT) reaches ~30-55% with massive effort.
"""
import os, json, glob, numpy as np
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

def fliplr(g): return g[:, ::-1].copy()
def flipud(g): return g[::-1, :].copy()
def rot180(g): return np.rot90(g,2).copy()
def rot90(g):  return np.rot90(g,1).copy()
def rot270(g): return np.rot90(g,3).copy()
def transpose(g): return g.T.copy()
def roll(g,a,s): return np.roll(g,s,axis=a)
def gravity_v(g):
    o=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; o[g.shape[0]-len(col):,c]=col
    return o
def gravity_h(g):
    o=np.zeros_like(g)
    for r in range(g.shape[0]):
        row=[v for v in g[r,:] if v!=0]; o[r,g.shape[1]-len(row):]=row
    return o

def shape_preserving_prims(g):
    P={'id':lambda x:x.copy(),'fliplr':fliplr,'flipud':flipud,'rot180':rot180,
       'roll_d':lambda x:roll(x,0,1),'roll_u':lambda x:roll(x,0,-1),
       'roll_r':lambda x:roll(x,1,1),'roll_l':lambda x:roll(x,1,-1),
       'grav_v':gravity_v,'grav_h':gravity_h}
    if g.shape[0]==g.shape[1]:
        P.update({'rot90':rot90,'rot270':rot270,'transpose':transpose})
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

def induce(train):
    # all pairs must be shape-preserving for our transforms
    if any(i.shape!=o.shape for i,o in train): return None
    P=shape_preserving_prims(train[0][0])
    ok=lambda f: all(f(i).shape==o.shape and np.array_equal(f(i),o) for i,o in train)
    for f in P.values():
        try:
            if ok(f): return f
        except Exception: pass
    cm=infer_colormap(train)
    if cm is not None:
        f=lambda g,cm=cm:apply_cmap(g,cm)
        if ok(f): return f
    for n1,f1 in P.items():
        try:
            d2=[(f1(i),o) for i,o in train]
            cm2=infer_colormap(d2)
            if cm2 is not None:
                f=lambda g,f1=f1,cm2=cm2:apply_cmap(f1(g),cm2)
                if ok(f): return f
            for n2,f2 in P.items():
                f=lambda g,f1=f1,f2=f2:f2(f1(g))
                if ok(f): return f
        except Exception: pass
    return None

files=sorted(glob.glob(os.path.join(DATA,"*.json")))
solved=0; attempted=0; solved_ids=[]
for fp in files:
    t=json.load(open(fp))
    train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
    prog=induce(train)
    if prog is None: continue
    attempted+=1
    ok=True
    for p in t["test"]:
        xi=np.array(p["input"]); yo=np.array(p["output"])
        try: pred=prog(xi)
        except Exception: ok=False; break
        if pred.shape!=yo.shape or not np.array_equal(pred,yo): ok=False; break
    if ok: solved+=1; solved_ids.append(os.path.basename(fp)[:8])

print(f"REAL ARC-AGI training set: {len(files)} tasks")
print(f"Our inducer found a consistent program for: {attempted} tasks")
print(f"SOLVED (correct on test): {solved}/{len(files)} = {100*solved/len(files):.1f}%")
print(f"solved task ids: {solved_ids}")
print(f"\nHonest: our small-DSL inducer solves only the few ARC tasks whose rule lives in its")
print(f"space. Field SOTA ~30-55% (huge DSL + neural search + TTT); humans ~85%. This is the")
print(f"real, no-cheating ARC number for the system we built.")

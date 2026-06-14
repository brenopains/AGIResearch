"""
H42 — The rule-creating system (program induction), and how far it reaches
==========================================================================

The user identified the CONSTANT behind every 'open problem': a system that
CREATES the rule itself. Its optimal form (Solomonoff induction / AIXI) is
provably UNCOMPUTABLE. Every buildable version is a bounded approximation:
program induction = SEARCH a hypothesis space for a program consistent with the
demos. Within the space it genuinely creates the rule; outside it cannot (theorem).

Here we build a real program-inducer over an expressive DSL and test it on a
BATTERY of diverse rules it was NEVER given -- it must induce each from 3 demos.
We report per-rule success and the overall solve rate. This is exactly how
practical ARC solvers work (and why they reach ~30-55%, not 100%).
"""
import itertools, numpy as np
rng=np.random.RandomState(0); G,C=5,4

def gravity(g):
    o=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; o[g.shape[0]-len(col):,c]=col
    return o
def gravity_h(g):
    o=np.zeros_like(g)
    for r in range(g.shape[0]):
        row=[v for v in g[r,:] if v!=0]; o[r,g.shape[1]-len(row):]=row
    return o
def majority_fill(g):
    v,c=np.unique(g,return_counts=True); return np.full_like(g,v[c.argmax()])
def flood_corner(g):
    return np.full_like(g,g[0,0])
def border(g):
    o=g.copy(); o[0,:]=1; o[-1,:]=1; o[:,0]=1; o[:,-1]=1; return o

# DSL the inducer can search over (its hypothesis space / primitives)
PRIMS={'id':lambda g:g.copy(),'fliplr':lambda g:np.fliplr(g),'flipud':lambda g:np.flipud(g),
 'rot90':lambda g:np.rot90(g,1),'rot180':lambda g:np.rot90(g,2),'rot270':lambda g:np.rot90(g,3),
 'transpose':lambda g:g.T.copy(),'roll_d':lambda g:np.roll(g,1,0),'roll_u':lambda g:np.roll(g,-1,0),
 'roll_r':lambda g:np.roll(g,1,1),'roll_l':lambda g:np.roll(g,-1,1),'gravity':gravity,'gravity_h':gravity_h}

def infer_colormap(demos):
    m={}
    for i,o in demos:
        for a,b in zip(i.ravel(),o.ravel()):
            if a in m and m[a]!=b: return None
            m[a]=b
    return m
def cmap_fn(m): return lambda g:np.vectorize(lambda v:m.get(v,v))(g)

def induce(demos):
    """Search the DSL for a program consistent with all demos = CREATE the rule."""
    ok=lambda f: all(np.array_equal(f(i),o) for i,o in demos)
    for f in PRIMS.values():
        if ok(f): return f
    cm=infer_colormap(demos)
    if cm is not None and ok(cmap_fn(cm)): return cmap_fn(cm)
    for f1 in PRIMS.values():
        for f2 in PRIMS.values():
            g=lambda x,f1=f1,f2=f2:f2(f1(x))
            if ok(g): return g
        d2=[(f1(i),o) for i,o in demos]; cm2=infer_colormap(d2)
        if cm2 is not None:
            g=lambda x,f1=f1,cm2=cm2:cmap_fn(cm2)(f1(x))
            if ok(g): return g
    return None

def make_task(rule):
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,np.array(rule(g))) for g in gs[:3]], gs[3], np.array(rule(gs[3]))

def solve_rate(rule,n=120):
    s=0
    for _ in range(n):
        d,q,qo=make_task(rule); f=induce(d)
        if f is not None and np.array_equal(f(q),qo): s+=1
    return s/n

# battery of diverse rules the inducer must CREATE from demos (never told which)
def perm(): p=rng.permutation(C); return lambda g:p[g]
BATTERY=[('fliplr',lambda g:np.fliplr(g)),('rot90',lambda g:np.rot90(g,1)),
 ('transpose',lambda g:g.T.copy()),('color-perm',perm()),
 ('flip+color',(lambda p:(lambda g:p[np.fliplr(g)]))(rng.permutation(C))),
 ('gravity',gravity),('gravity_h',gravity_h),
 ('gravity+color',(lambda p:(lambda g:p[gravity(g)]))(rng.permutation(C))),
 ('shift-by-2',lambda g:np.roll(g,2,1)),
 ('majority-fill',majority_fill),('flood-corner',flood_corner),('border',border)]

print("RULE-CREATING SYSTEM (program induction) — induces each rule from demos alone")
print("-"*60)
solved=0
for name,rule in BATTERY:
    r=solve_rate(rule); solved+= (r>0.8)
    tag = "CREATED" if r>0.8 else ("partial" if r>0.2 else "FAILED (outside DSL)")
    print(f"  {name:>16}: {r:>4.0%}   {tag}")
print("-"*60)
print(f"Autonomously created {solved}/{len(BATTERY)} rule types.")
print("Real: it CREATES rules within its DSL (no rule pre-given). Bounded: rules outside the")
print("DSL (majority-fill, flood, border) = ~0%. Optimal universal inducer (Solomonoff/AIXI)")
print("is UNCOMPUTABLE -> every system has a bounded space -> ARC always includes outside it.")

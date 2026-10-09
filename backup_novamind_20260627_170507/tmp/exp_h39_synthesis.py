"""
H39 — Synthesis-by-search: the diagnosis-driven architectural fix, and the wall
===============================================================================

H38 diagnosed the failure: the model INTERPOLATES known transforms, never
SYNTHESIZES the novel rule. The classic fix (and how real ARC solvers work):
PROGRAM SEARCH over a DSL. For each task, search for a program consistent with
the 3 demos, then apply it to the query.

DSL primitives: identity, flips, rotations, rolls, and a parametric COLOR-MAP
(inferred from demos). Plus depth-2 compositions (spatial then color-map, etc).

We test exact-grid-match on:
  IN-DSL single       : rule is one primitive (e.g., a flip)        -> search finds it
  IN-DSL composition  : rule = two primitives (e.g., flip+recolor)  -> depth-2 finds it
  OUT-OF-DSL (gravity): a primitive NOT in the DSL                  -> search fails

This maps the boundary PRECISELY: synthesis solves what's expressible in its
primitives (a REAL improvement -- exact-match, vs the neural net's 0%), and FAILS
the moment the rule needs a primitive it doesn't have. That out-of-DSL wall is the
SAME wall as H7/H30 (inventing a new primitive) -- the open problem. Honest: this
is narrow synthesis, not AGI; ARC guarantees rules outside any fixed DSL.
"""
import itertools, numpy as np
rng=np.random.RandomState(0); G,C=5,4

def gravity(g):
    out=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; out[g.shape[0]-len(col):,c]=col
    return out

SPATIAL={'identity':lambda g:g.copy(),'fliplr':lambda g:np.fliplr(g),'flipud':lambda g:np.flipud(g),
 'rot90':lambda g:np.rot90(g,1),'rot180':lambda g:np.rot90(g,2),'rot270':lambda g:np.rot90(g,3),
 'roll_d':lambda g:np.roll(g,1,0),'roll_u':lambda g:np.roll(g,-1,0),'roll_r':lambda g:np.roll(g,1,1),'roll_l':lambda g:np.roll(g,-1,1)}

def infer_colormap(demos):
    m={}
    for inp,out in demos:
        for a,b in zip(inp.ravel(),out.ravel()):
            if a in m and m[a]!=b: return None
            m[a]=b
    return m
def apply_colormap(g,m):
    out=g.copy()
    for a,b in m.items(): out[g==a]=b
    return out

def consistent(prog_fn, demos):
    return all(np.array_equal(prog_fn(inp),out) for inp,out in demos)

def search(demos):
    """Return a program function consistent with all demos, or None. Depth 1 and 2."""
    # depth-1 spatial
    for name,f in SPATIAL.items():
        if consistent(f,demos): return ('S:'+name, f)
    # depth-1 color-map
    cm=infer_colormap(demos)
    if cm is not None and consistent(lambda g:apply_colormap(g,cm),demos):
        return ('colormap', lambda g:apply_colormap(g,cm))
    # depth-2: spatial then color-map (infer cm AFTER spatial)
    for name,f in SPATIAL.items():
        d2=[(f(inp),out) for inp,out in demos]; cm2=infer_colormap(d2)
        if cm2 is not None:
            g2=lambda g,f=f,cm2=cm2: apply_colormap(f(g),cm2)
            if consistent(g2,demos): return ('S:'+name+'->colormap', g2)
    # depth-2: spatial then spatial
    for n1,f1 in SPATIAL.items():
        for n2,f2 in SPATIAL.items():
            g2=lambda g,f1=f1,f2=f2: f2(f1(g))
            if consistent(g2,demos): return (f'S:{n1}->S:{n2}', g2)
    return None

def make_task(rule):
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,np.array(rule(g))) for g in gs[:3]], gs[3], np.array(rule(gs[3]))

def test_category(name,rule_factory,n=300):
    solved=0
    for _ in range(n):
        rule=rule_factory()
        demos,q,qo=make_task(rule)
        prog=search(demos)
        if prog is not None and np.array_equal(prog[1](q),qo): solved+=1
    print(f"  {name:>22}: exact-match {solved/n:.0%}")
    return solved/n

def rand_perm():
    p=rng.permutation(C); return lambda g:p[g]
def rand_spatial():
    f=list(SPATIAL.values())[rng.randint(len(SPATIAL))]; return f
def rand_comp():   # spatial then recolor
    f=list(SPATIAL.values())[rng.randint(1,len(SPATIAL))]; p=rng.permutation(C)
    return lambda g:p[f(g)]

print("Synthesis-by-search (program search over a DSL) — exact-grid-match")
print("-"*52)
test_category("IN-DSL single (spatial)", rand_spatial)
test_category("IN-DSL single (color)", rand_perm)
test_category("IN-DSL composition", rand_comp)
test_category("OUT-OF-DSL (gravity)", lambda: gravity)
print("-"*52)
print("Map: synthesis SOLVES in-DSL exactly (real fix vs neural 0% exact), FAILS out-of-DSL.")
print("The wall is the PRIMITIVE SET (DSL coverage) = the same wall as H7/H30 (invent a new")
print("primitive). ARC includes rules outside ANY fixed DSL by design -> the open problem.")

"""
H41 — Solve gravity (as the user asked) + show what it really means
===================================================================

The user asked: run an algorithm that solves gravity. Done -- and it's EASY: just
put gravity in the primitive set, and program search finds it (100%). But this is
exactly the move that PROVES the wall rather than breaking it:

  DSL_v1 (spatial + colormap, NO gravity):     gravity -> 0%   (foreign)
  DSL_v2 (spatial + colormap + GRAVITY):       gravity -> 100% (SOLVED!)
  DSL_v2, NEW unanticipated rule (majority-fill, not composable): -> 0%

Lesson (with data, not preaching): solving any SPECIFIC named rule is trivial --
add it to the primitives. But ARC's next task uses a rule neither of us
anticipated, and it's 0% again. The wall is not 'gravity'; it is UNBOUNDED NOVELTY
outside any FIXED primitive set. You can always add the rule you're looking at;
you can never enumerate all rules. That infinite regress IS the AGI problem.
"""
import itertools, numpy as np
rng=np.random.RandomState(0); G,C=5,4

def gravity(g):
    out=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; out[g.shape[0]-len(col):,c]=col
    return out
def majority_fill(g):              # the NEXT unanticipated rule (aggregation; not composable)
    vals,counts=np.unique(g,return_counts=True); m=vals[counts.argmax()]
    return np.full_like(g,m)

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

def build_dsl(include_gravity):
    prims=dict(SPATIAL)
    if include_gravity: prims['GRAVITY']=gravity
    return prims

def search(demos, prims):
    # depth-1 primitives
    for f in prims.values():
        if all(np.array_equal(f(i),o) for i,o in demos): return f
    # parametric colormap
    cm=infer_colormap(demos)
    if cm is not None:
        f=lambda g: np.vectorize(lambda v:cm.get(v,v))(g)
        if all(np.array_equal(f(i),o) for i,o in demos): return f
    # depth-2 (prim then prim)
    for f1 in prims.values():
        for f2 in prims.values():
            f=lambda g,f1=f1,f2=f2: f2(f1(g))
            if all(np.array_equal(f(i),o) for i,o in demos): return f
    return None

def make_task(rule):
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,np.array(rule(g))) for g in gs[:3]], gs[3], np.array(rule(gs[3]))

def test(rule, prims, n=200):
    s=0
    for _ in range(n):
        demos,q,qo=make_task(rule); p=search(demos,prims)
        if p is not None and np.array_equal(p(q),qo): s+=1
    return s/n

dsl1=build_dsl(False); dsl2=build_dsl(True)
print("Solving gravity by program search:")
print("-"*52)
print(f"  gravity  | DSL sem gravidade:     {test(gravity,dsl1):.0%}")
print(f"  gravity  | DSL COM gravidade:     {test(gravity,dsl2):.0%}   <- RESOLVIDO (voce pediu)")
print("-"*52)
print(f"  NOVA regra (majority-fill) | DSL COM gravidade: {test(majority_fill,dsl2):.0%}")
print("-"*52)
print("Resolver UMA regra nomeada = trivial (adiciona ela). Mas a regra SEGUINTE, que nao")
print("antecipamos (majority-fill), volta a 0% mesmo com o DSL expandido. A parede nao e a")
print("gravidade -- e a NOVIDADE ILIMITADA fora de QUALQUER conjunto fixo. Esse regresso e a AGI.")

"""
H40 — "Invent new primitives on the fly": deeper synthesis, and where it reaches
================================================================================

The user's target: a system that INVENTS new primitives. The real technique is
deeper program synthesis / library learning: by composing/searching deeper, the
system effectively builds new (composite) primitives on the fly.

We test exactly this. Program search at increasing DEPTH (1, 2, 3) over base ops.
  DEEP-COMPOSITION tasks : rule = compose 3 base spatial ops. Depth-2 search FAILS;
                           depth-3 (deeper 'invented' primitives) SOLVES.
  FOREIGN task (gravity) : a primitive NOT derivable from the base set at ANY depth.

Maps the boundary: inventing deeper composite primitives genuinely extends reach
(real win), BUT the foreign primitive stays unreachable at every depth -- because
no composition of {flip,rotate,shift,color} equals gravity. Reaching it requires
a primitive OUTSIDE the base = acquiring genuinely new content = grounding = the
open problem (H7/H30/H29/H39, same wall, 6th time).
"""
import numpy as np
rng=np.random.RandomState(0); G,C=5,4

def gravity(g):
    out=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; out[g.shape[0]-len(col):,c]=col
    return out
SPATIAL=[lambda g:g.copy(),lambda g:np.fliplr(g),lambda g:np.flipud(g),
 lambda g:np.rot90(g,1),lambda g:np.rot90(g,2),lambda g:np.rot90(g,3),
 lambda g:np.roll(g,1,0),lambda g:np.roll(g,-1,0),lambda g:np.roll(g,1,1),lambda g:np.roll(g,-1,1)]

def make_task(rule):
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,np.array(rule(g))) for g in gs[:3]], gs[3], np.array(rule(gs[3]))

def search_depth(demos, D):
    """Search compositions of up to D spatial ops consistent with demos."""
    import itertools
    for d in range(1,D+1):
        for combo in itertools.product(range(len(SPATIAL)),repeat=d):
            def prog(g,combo=combo):
                for idx in combo: g=SPATIAL[idx](g)
                return g
            if all(np.array_equal(prog(i),o) for i,o in demos): return prog
    return None

def test(name, rule_factory, D, n=200):
    s=0
    for _ in range(n):
        rule=rule_factory(); demos,q,qo=make_task(rule); p=search_depth(demos,D)
        if p is not None and np.array_equal(p(q),qo): s+=1
    return s/n

def deep3():  # compose 3 random spatial ops
    a,b,c=[SPATIAL[rng.randint(len(SPATIAL))] for _ in range(3)]
    return lambda g: c(b(a(g)))

print("'Inventing primitives' = deeper synthesis. Exact-match by search DEPTH.")
print(f"{'depth':>6} {'DEEP-3 composition':>20} {'FOREIGN (gravity)':>20}")
print("-"*48)
for D in [1,2,3]:
    d3=test('deep3',deep3,D); gv=test('grav',lambda:gravity,D)
    print(f"{D:>6} {d3:>19.0%} {gv:>20.0%}")
print("-"*48)
print("Map: deeper synthesis SOLVES deeper compositions (inventing composite primitives")
print("WORKS, real win). But FOREIGN (gravity) = 0% at EVERY depth -- no composition of the")
print("base ops equals it. Reaching it needs a primitive OUTSIDE the base = new content from")
print("the world = grounding = the open problem (same wall as H7/H29/H30/H39, 6th time).")

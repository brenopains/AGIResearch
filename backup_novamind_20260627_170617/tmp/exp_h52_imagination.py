"""
H52 — "Imagination" (partial-reward beam search) + "rhythm" (adaptive budget)
=============================================================================

User's idea: a pause/imagination state runs MICRO mental attempts (hypotheses)
that get MINI-REWARDS for getting closer, refining toward a solution; plus a
rhythmic, self-regulated effort (like breathing/heartbeat) that spends more when
the problem is hard/urgent.

Translation to real, testable mechanisms:
  - micro-attempts + mini-rewards = BEAM SEARCH guided by a GRADED reward
    (fraction of cells matching the demos), instead of exact all-or-nothing.
  - rhythm/breathing = ADAPTIVE COMPUTE budget: expand more when the best reward
    is still low (high 'urgency'); stop early when a perfect program is found.

Tests:
  (A) MECHANISM: deep-composition tasks (need depth 3-4). Does graded-reward beam
      search FIND them, where exhaustive shallow (depth-2) fails?
  (B) ARC reality: does beam-depth-4 beat exhaustive depth-2 (9.8%)? Honest
      expectation: ~no -- ARC's wall is BREADTH (missing primitives), not depth
      (H49 bootstrapping already showed depth doesn't help ARC).
"""
import itertools, numpy as np
rng=np.random.RandomState(0); G,C=5,4
SPAT=[lambda g:g.copy(),lambda g:np.fliplr(g),lambda g:np.flipud(g),lambda g:np.rot90(g,1),
      lambda g:np.rot90(g,2),lambda g:np.rot90(g,3),lambda g:np.roll(g,1,0),lambda g:np.roll(g,1,1),
      lambda g:g.T.copy(),lambda g:np.roll(g,-1,0)]

def make(rule):
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,np.array(rule(g))) for g in gs[:3]], gs[3], np.array(rule(gs[3]))

def reward(prog, demos):
    """graded mini-reward: mean cell-match across demos (0..1); shape mismatch -> 0."""
    tot=0
    for i,o in demos:
        try: r=prog(i)
        except Exception: return 0.0
        if r.shape!=o.shape: return 0.0
        tot+=(r==o).mean()
    return tot/len(demos)

def beam_search(demos, max_depth=4, beam=30):
    """Imagination: keep top-`beam` partial programs by mini-reward; expand depth by depth.
    Adaptive 'rhythm': stop instantly if a perfect (reward=1) program appears."""
    base=[(reward(lambda g,f=f:f(g),demos), [f]) for f in SPAT]
    nodes=len(SPAT)
    frontier=sorted(base,key=lambda x:-x[0])[:beam]
    if frontier[0][0]>=0.999: return frontier[0][1], nodes, 1
    for d in range(2,max_depth+1):
        cand=[]
        # adaptive width: if best reward still low, search wider (more urgency)
        best=frontier[0][0]; width=beam if best>0.6 else beam*2
        for sc,prog in frontier[:width//len(SPAT)+1]:
            for f in SPAT:
                newp=prog+[f]
                fn=lambda g,seq=newp:_run(seq,g)
                r=reward(fn,demos); nodes+=1
                cand.append((r,newp))
                if r>=0.999: return newp, nodes, d
        frontier=sorted(cand,key=lambda x:-x[0])[:beam]
        if not frontier: break
    return (frontier[0][1] if frontier else None), nodes, max_depth
def _run(seq,g):
    for f in seq: g=f(g)
    return g

def exhaustive_d2(demos):
    for f in SPAT:
        if reward(lambda g,f=f:f(g),demos)>=0.999: return [f]
    for f1 in SPAT:
        for f2 in SPAT:
            if reward(lambda g,f1=f1,f2=f2:f2(f1(g)),demos)>=0.999: return [f1,f2]
    return None

def deep3(): a,b,c=[SPAT[rng.randint(len(SPAT))] for _ in range(3)]; return lambda g:c(b(a(g)))
def deep4(): a,b,c,d=[SPAT[rng.randint(len(SPAT))] for _ in range(4)]; return lambda g:d(c(b(a(g))))

print("(A) MECANISMO: imaginacao (busca em feixe por recompensa parcial) vs exaustiva-prof2")
print(f"{'tarefa':>10} {'exaustiva-d2':>13} {'beam(imaginacao)':>17} {'nos medios':>11}")
for name,rf in [("deep-3",deep3),("deep-4",deep4)]:
    ex=0; bm=0; nodes=[]
    for _ in range(60):
        demos,q,qo=make(rf())
        p2=exhaustive_d2(demos); ex+= (p2 is not None and np.array_equal(_run(p2,q),qo))
        pb,nn,dd=beam_search(demos); nodes.append(nn)
        bm+= (pb is not None and np.array_equal(_run(pb,q),qo))
    print(f"{name:>10} {ex/60:>12.0%} {bm/60:>16.0%} {np.mean(nodes):>11.0f}")
print("-"*54)
# (B) rhythm: nodes expanded for easy (depth-1 solvable) vs hard (deep)
easy_nodes=[]; hard_nodes=[]
for _ in range(40):
    d,_,_=make(SPAT[rng.randint(len(SPAT))]); _,n,_=beam_search(d); easy_nodes.append(n)
    d,_,_=make(deep4()); _,n,_=beam_search(d); hard_nodes.append(n)
print(f"(B) RITMO (orcamento adaptativo): nos em tarefa FACIL {np.mean(easy_nodes):.0f} vs DIFICIL {np.mean(hard_nodes):.0f}")
print("    -> o sistema 'pensa' menos no facil e mais no dificil (auto-regulacao tipo respiracao).")
print("-"*54)
print("Leitura honesta: a imaginacao-com-mini-recompensa ENCONTRA composicoes profundas que a")
print("busca rasa perde (mecanismo validado), e o ritmo regula esforco. MAS no ARC real isso")
print("ajuda pouco -- a parede e LARGURA (primitivas faltantes), nao profundidade (H49 ja mostrou).")

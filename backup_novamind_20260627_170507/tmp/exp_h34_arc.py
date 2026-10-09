"""
H34 — mini-ARC: the REAL non-narrow AGI test (few-shot rule induction)
=====================================================================

The user keeps asking for a NON-NARROW AGI test. The field's gold standard is
ARC-AGI (Chollet): infer a NOVEL transformation rule from a few examples and
apply it -- impossible to memorize, tests fluid/general intelligence.

Mini version: GxG color grids. A task = a hidden rule R. The model sees 3 demo
(input, R(input)) pairs + a query input, and must produce R(query).
  TRAIN on SEEN rule type   : SPATIAL transforms (flip/rotate/transpose/shift).
  TEST  on NOVEL rule type  : COLOR-LOGIC transforms (color swaps/permutations),
                              a genuinely different KIND, never seen in training.

A meta-learner that truly INDUCES rules (general) would transfer to the novel
rule type. A narrow pattern-matcher will pass SEEN and FAIL NOVEL.
Metric: exact-grid-match (and per-cell acc) on seen vs novel rules.

HONEST prediction: high on SEEN, ~chance on NOVEL -> narrow, not AGI. This is the
non-narrow test done right (novel rule TYPE = fluid generality, the ARC hallmark).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
G,C=5,4                      # grid size, colors

# ---- rule families ----
def spatial_rules():
    R=[]
    R.append(lambda g: np.fliplr(g)); R.append(lambda g: np.flipud(g))
    R.append(lambda g: np.rot90(g,1)); R.append(lambda g: np.rot90(g,2)); R.append(lambda g: np.rot90(g,3))
    R.append(lambda g: g.T)
    R.append(lambda g: np.roll(g,1,axis=0)); R.append(lambda g: np.roll(g,1,axis=1))
    R.append(lambda g: np.roll(g,-1,axis=0)); R.append(lambda g: np.roll(g,-1,axis=1))
    return R
def color_rules():   # NOVEL kind: relabel colors (spatial layout unchanged)
    R=[]
    for _ in range(10):
        perm=np.random.RandomState(_+99).permutation(C)
        R.append(lambda g,p=perm: p[g])
    return R

def make_task(rule,rng):
    grids=[rng.randint(0,C,(G,G)) for _ in range(4)]
    demos=[(g,np.array(rule(g))) for g in grids[:3]]
    q=grids[3]; qo=np.array(rule(q))
    return demos,q,qo

def onehot(g): return np.eye(C,dtype=np.float32)[g].transpose(2,0,1)  # (C,G,G)

class MetaARC(nn.Module):
    def __init__(s,h=128):
        super().__init__()
        s.pair=nn.Sequential(nn.Conv2d(2*C,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),
                             nn.Flatten(),nn.Linear(64*G*G,h))            # encode a demo (in,out)
        s.q=nn.Sequential(nn.Conv2d(C,64,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(64*G*G,h))
        s.dec=nn.Sequential(nn.Linear(2*h,256),nn.ReLU(),nn.Linear(256,C*G*G))
    def forward(s,demos,q):
        # demos: (B,3,2C,G,G) ; q: (B,C,G,G)
        B=q.shape[0]
        de=s.pair(demos.reshape(B*3,2*C,G,G)).reshape(B,3,-1).mean(1)     # rule embedding
        qe=s.q(q)
        out=s.dec(torch.cat([de,qe],1)).reshape(B,C,G,G)
        return out

def batch(rules,n,rng):
    D=[];Q=[];Y=[]
    for _ in range(n):
        r=rules[rng.randint(len(rules))]; demos,q,qo=make_task(r,rng)
        D.append(np.stack([np.concatenate([onehot(i),onehot(o)],0) for i,o in demos],0))
        Q.append(onehot(q)); Y.append(qo)
    return (torch.tensor(np.array(D)).to(DEVICE),torch.tensor(np.array(Q)).to(DEVICE),
            torch.tensor(np.array(Y),dtype=torch.long).to(DEVICE))

def evaluate(model,rules,rng,n=400):
    D,Q,Y=batch(rules,n,rng); model.eval()
    with torch.no_grad():
        pred=model(D,Q).argmax(1)
    cell=(pred==Y).float().mean().item()
    exact=(pred==Y).reshape(n,-1).all(1).float().mean().item()
    model.train(); return cell,exact

def run(seed):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    SEEN=spatial_rules(); NOVEL=color_rules()
    m=MetaARC().to(DEVICE); opt=torch.optim.Adam(m.parameters(),2e-3)
    for step in range(1500):
        D,Q,Y=batch(SEEN,64,rng)
        opt.zero_grad(); F.cross_entropy(m(D,Q),Y).backward(); opt.step()
    seen_cell,seen_exact=evaluate(m,SEEN,rng)
    nov_cell,nov_exact=evaluate(m,NOVEL,rng)
    return seen_cell,seen_exact,nov_cell,nov_exact

print(f"Device: {DEVICE} | mini-ARC | grid {G}x{G}, {C} colors | cell-chance {100/C:.0f}%")
sc=[];se=[];nc=[];ne=[]
for s in range(3):
    a,b,c,d=run(s); sc.append(a);se.append(b);nc.append(c);ne.append(d)
import numpy as np
sc,se,nc,ne=map(lambda v:np.array(v),(sc,se,nc,ne))
print("-"*60)
print(f"  SEEN rule type (spatial):  cell {sc.mean():.0%} | EXACT-grid {se.mean():.0%}")
print(f"  NOVEL rule type (color):   cell {nc.mean():.0%} | EXACT-grid {ne.mean():.0%}   (chance cell {100/C:.0f}%)")
print("-"*60)
print(f"VERDICT: {'GENERALIZED to novel rules (AGI-like fluid reasoning)' if ne.mean()>0.4 else 'FAILS on novel rule type -> narrow pattern-matching, NOT fluid/general (ARC gap = the AGI wall)'}")

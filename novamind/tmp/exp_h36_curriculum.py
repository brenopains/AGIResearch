"""
H36 — "Train on logic first, then ARC" (the user's hypothesis, tested fairly)
=============================================================================

User's idea: humans aren't blank slates; train the system on logic/patterns
FIRST, then test on ARC. (Correct that humans need priors -- and this is the
field-standard approach.) We test it: a meta-learner trained on a BROAD curriculum
of rule families, then evaluated on a HELD-OUT, genuinely-novel family.

  TRAIN families : flip, rotate, shift, color-permutation  (4 kinds of logic)
  TEST  family   : GRAVITY (non-zero cells fall down each column) -- never seen,
                   a different KIND of rule (per-column reasoning about emptiness).

If broad logic training transfers to the novel family -> the user is right.
HONEST prediction: good on TRAINED families, ~chance on the HELD-OUT family ->
the ARC gap persists by design (training similar tasks doesn't grant fluid
induction of genuinely new rules).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
G,C=5,4

def gravity(g):
    out=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; out[g.shape[0]-len(col):,c]=col
    return out
FAMILIES={
 'flip':   [lambda g:np.fliplr(g), lambda g:np.flipud(g)],
 'rotate': [lambda g:np.rot90(g,1), lambda g:np.rot90(g,2), lambda g:np.rot90(g,3)],
 'shift':  [lambda g:np.roll(g,1,0), lambda g:np.roll(g,-1,0), lambda g:np.roll(g,1,1), lambda g:np.roll(g,-1,1)],
 'color':  [(lambda p: (lambda g: p[g]))(np.random.RandomState(i+5).permutation(C)) for i in range(8)],
 'gravity':[gravity],
}
TRAIN_FAMS=['flip','rotate','shift','color']; HELDOUT='gravity'

def make_task(rule,rng):
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    demos=[(g,np.array(rule(g))) for g in gs[:3]]; q=gs[3]; return demos,q,np.array(rule(q))
def oh(g): return np.eye(C,dtype=np.float32)[g].transpose(2,0,1)

class Meta(nn.Module):
    def __init__(s,h=192):
        super().__init__()
        s.pair=nn.Sequential(nn.Conv2d(2*C,96,3,padding=1),nn.ReLU(),nn.Conv2d(96,96,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(96*G*G,h))
        s.q=nn.Sequential(nn.Conv2d(C,96,3,padding=1),nn.ReLU(),nn.Conv2d(96,96,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(96*G*G,h))
        s.dec=nn.Sequential(nn.Linear(2*h,384),nn.ReLU(),nn.Linear(384,384),nn.ReLU(),nn.Linear(384,C*G*G))
    def forward(s,D,Q):
        B=Q.shape[0]; de=s.pair(D.reshape(B*3,2*C,G,G)).reshape(B,3,-1).mean(1); qe=s.q(Q)
        return s.dec(torch.cat([de,qe],1)).reshape(B,C,G,G)

def batch(fams,n,rng):
    D=[];Q=[];Y=[]
    for _ in range(n):
        fam=fams[rng.randint(len(fams))]; r=FAMILIES[fam][rng.randint(len(FAMILIES[fam]))]
        demos,q,qo=make_task(r,rng)
        D.append(np.stack([np.concatenate([oh(i),oh(o)],0) for i,o in demos],0)); Q.append(oh(q)); Y.append(qo)
    return (torch.tensor(np.array(D)).to(DEVICE),torch.tensor(np.array(Q)).to(DEVICE),torch.tensor(np.array(Y),dtype=torch.long).to(DEVICE))

def evalf(m,fams,rng,n=600):
    D,Q,Y=batch(fams,n,rng); m.eval()
    with torch.no_grad(): p=m(D,Q).argmax(1)
    m.train(); return (p==Y).float().mean().item(),(p==Y).reshape(n,-1).all(1).float().mean().item()

def run(seed):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    m=Meta().to(DEVICE); opt=torch.optim.Adam(m.parameters(),1.5e-3)
    for step in range(3000):
        D,Q,Y=batch(TRAIN_FAMS,96,rng); opt.zero_grad(); F.cross_entropy(m(D,Q),Y).backward(); opt.step()
    tr_cell,tr_ex=evalf(m,TRAIN_FAMS,rng); ho_cell,ho_ex=evalf(m,[HELDOUT],rng)
    return tr_cell,tr_ex,ho_cell,ho_ex

print(f"Device: {DEVICE} | broad curriculum {TRAIN_FAMS} -> held-out '{HELDOUT}' | cell-chance {100/C:.0f}%")
TC=[];TE=[];HC=[];HE=[]
for s in range(3):
    a,b,c,d=run(s); TC.append(a);TE.append(b);HC.append(c);HE.append(d)
import numpy as np
TC,TE,HC,HE=map(lambda v:np.array(v),(TC,TE,HC,HE))
print("-"*60)
print(f"  TRAINED families:  cell {TC.mean():.0%} | exact {TE.mean():.0%}")
print(f"  HELD-OUT '{HELDOUT}': cell {HC.mean():.0%} | exact {HE.mean():.0%}   (chance cell {100/C:.0f}%)")
print("-"*60)
print(f"VERDICT: {'transfers to novel family (user right!)' if HE.mean()>0.4 else 'broad logic training does NOT transfer to the novel family -> the ARC gap persists by design (fluid induction unsolved)'}")

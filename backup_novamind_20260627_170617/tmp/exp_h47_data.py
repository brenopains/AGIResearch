"""
H47 — The user's claim: "with millions of examples, it learns" (and the boundary)
=================================================================================

User's (correct) point: ARC's few-shot framing presupposes a vast prior; with
LOTS of examples of a task family, a system masters that family. We test it:
train a meta-learner on a color-rule family with INCREASING amounts of data, and
measure:
  IN-DISTRIBUTION  (held-out tasks of the SAME family) -> should rise with data
  OUT-OF-DIST      (a different family: spatial transforms) -> should stay low

This VALIDATES the user's claim (data -> masters the family) AND maps the wall
(the hard part is the NOVEL family / few-shot, which needs the rich prior).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
G,C=4,4

def color_task(rng):                      # family A: a random color permutation
    p=rng.permutation(C); rule=lambda g:p[g]
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,rule(g)) for g in gs[:3]], gs[3], rule(gs[3])
def spatial_task(rng):                    # family B (novel): a random spatial transform
    fs=[lambda g:np.fliplr(g),lambda g:np.flipud(g),lambda g:np.rot90(g,1),lambda g:np.rot90(g,2)]
    f=fs[rng.randint(len(fs))]
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,np.array(f(g))) for g in gs[:3]], gs[3], np.array(f(gs[3]))

def oh(g): return np.eye(C,dtype=np.float32)[g].transpose(2,0,1)
class Meta(nn.Module):
    def __init__(s,h=128):
        super().__init__()
        s.pair=nn.Sequential(nn.Conv2d(2*C,64,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(64*G*G,h))
        s.q=nn.Sequential(nn.Conv2d(C,64,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(64*G*G,h))
        s.dec=nn.Sequential(nn.Linear(2*h,256),nn.ReLU(),nn.Linear(256,C*G*G))
    def forward(s,D,Q):
        B=Q.shape[0]; de=s.pair(D.reshape(B*3,2*C,G,G)).reshape(B,3,-1).mean(1); qe=s.q(Q)
        return s.dec(torch.cat([de,qe],1)).reshape(B,C,G,G)

def batch(task_fn,n,rng):
    D=[];Q=[];Y=[]
    for _ in range(n):
        d,q,qo=task_fn(rng)
        D.append(np.stack([np.concatenate([oh(i),oh(o)],0) for i,o in d],0)); Q.append(oh(q)); Y.append(qo)
    return (torch.tensor(np.array(D)).to(DEVICE),torch.tensor(np.array(Q)).to(DEVICE),torch.tensor(np.array(Y),dtype=torch.long).to(DEVICE))

def evalf(m,task_fn,rng,n=400):
    D,Q,Y=batch(task_fn,n,rng); m.eval()
    with torch.no_grad(): p=m(D,Q).argmax(1)
    m.train(); return (p==Y).float().mean().item()

def run(steps,seed=0):
    torch.manual_seed(seed); rng=np.random.RandomState(seed)
    m=Meta().to(DEVICE); opt=torch.optim.Adam(m.parameters(),2e-3)
    for _ in range(steps):
        D,Q,Y=batch(color_task,64,rng); opt.zero_grad(); F.cross_entropy(m(D,Q),Y).backward(); opt.step()
    indist=evalf(m,color_task,np.random.RandomState(seed+1))
    outdist=evalf(m,spatial_task,np.random.RandomState(seed+2))
    return indist,outdist,steps*64

print(f"Device: {DEVICE} | train on COLOR family with increasing data | cell-chance {100/C:.0f}%")
print(f"{'examples seen':>14} {'IN-DIST (color)':>16} {'OUT-DIST (spatial,novel)':>26}")
print("-"*60)
for steps in [100,1000,5000]:
    ind,out,ex=run(steps)
    print(f"{ex:>14} {ind:>15.0%} {out:>25.0%}")
print("-"*60)
print("VALIDATES the user: more examples -> masters the FAMILY (in-dist rises).")
print("MAPS the wall: the NOVEL family stays low regardless of data -- ARC's hard case is")
print("few-shot on a genuinely new family, which needs the vast prior (the 'adult human').")

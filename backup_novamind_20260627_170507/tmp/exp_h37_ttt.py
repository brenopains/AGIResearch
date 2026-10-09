"""
H37 — Test-Time Training (TTT): the real unexplored ARC lever
=============================================================

We never let the model ADAPT to the test task. TTT (the technique that drove the
2024 ARC jump) does: for each held-out task, fine-tune on its OWN demos before
predicting. We test it on the held-out novel family (gravity).

  base meta-learner: trained on TRAIN_FAMS (flip/rotate/shift/color).
  ZERO-SHOT : predict the gravity query directly (H36 baseline).
  TTT       : for each gravity task, fine-tune a copy of base on its 3 demos via
              LEAVE-ONE-OUT (predict demo_i's output from the other 2 demos + its
              input) for a few steps, THEN predict the real query.

Honest: TTT can genuinely improve held-out results (adapts to the task). It does
NOT make it AGI (frontier TTT gets ~5% on ARC-2). Data decides how much it helps.
"""
import copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
G,C=5,4
def gravity(g):
    out=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; out[g.shape[0]-len(col):,c]=col
    return out
FAMILIES={'flip':[lambda g:np.fliplr(g),lambda g:np.flipud(g)],
 'rotate':[lambda g:np.rot90(g,1),lambda g:np.rot90(g,2),lambda g:np.rot90(g,3)],
 'shift':[lambda g:np.roll(g,1,0),lambda g:np.roll(g,-1,0),lambda g:np.roll(g,1,1),lambda g:np.roll(g,-1,1)],
 'color':[(lambda p:(lambda g:p[g]))(np.random.RandomState(i+5).permutation(C)) for i in range(8)]}
TRAIN_FAMS=list(FAMILIES)

def task(rule,rng):
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,np.array(rule(g))) for g in gs[:3]], gs[3], np.array(rule(gs[3]))
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

def encode(demos,q):
    D=torch.tensor(np.stack([np.concatenate([oh(i),oh(o)],0) for i,o in demos],0)[None]).to(DEVICE)
    Q=torch.tensor(oh(q)[None]).to(DEVICE); return D,Q

def train_base(seed):
    rng=np.random.RandomState(seed); torch.manual_seed(seed); m=Meta().to(DEVICE); opt=torch.optim.Adam(m.parameters(),1.5e-3)
    for _ in range(3000):
        Ds=[];Qs=[];Ys=[]
        for _ in range(96):
            fam=TRAIN_FAMS[rng.randint(len(TRAIN_FAMS))]; r=FAMILIES[fam][rng.randint(len(FAMILIES[fam]))]
            demos,q,qo=task(r,rng)
            Ds.append(np.stack([np.concatenate([oh(i),oh(o)],0) for i,o in demos],0)); Qs.append(oh(q)); Ys.append(qo)
        D=torch.tensor(np.array(Ds)).to(DEVICE);Q=torch.tensor(np.array(Qs)).to(DEVICE);Y=torch.tensor(np.array(Ys),dtype=torch.long).to(DEVICE)
        opt.zero_grad(); F.cross_entropy(m(D,Q),Y).backward(); opt.step()
    return m

def ttt_adapt(base,demos,steps=25,lr=1e-3):
    m=copy.deepcopy(base); opt=torch.optim.Adam(m.parameters(),lr)
    # leave-one-out: predict demo_i output from other 2 demos as context + demo_i input
    for _ in range(steps):
        loss=0
        for i in range(3):
            ctx=[demos[j] for j in range(3) if j!=i]+[demos[i]]   # 3 context demos (reuse), query=demo_i input
            D=torch.tensor(np.stack([np.concatenate([oh(a),oh(b)],0) for a,b in ctx],0)[None]).to(DEVICE)
            Q=torch.tensor(oh(demos[i][0])[None]).to(DEVICE)
            Y=torch.tensor(demos[i][1][None],dtype=torch.long).to(DEVICE)
            loss=loss+F.cross_entropy(m(D,Q),Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def run(seed,n_test=120):
    base=train_base(seed); rng=np.random.RandomState(seed+1)
    zs_cell=zs_ex=tt_cell=tt_ex=0; tot=0
    for _ in range(n_test):
        demos,q,qo=task(gravity,rng); D,Q=encode(demos,q); Y=torch.tensor(qo[None],dtype=torch.long).to(DEVICE)
        with torch.no_grad(): p0=base(D,Q).argmax(1)
        zs_cell+=(p0==Y).float().mean().item(); zs_ex+=int((p0==Y).all().item())
        m=ttt_adapt(base,demos)
        with torch.no_grad(): p1=m(D,Q).argmax(1)
        tt_cell+=(p1==Y).float().mean().item(); tt_ex+=int((p1==Y).all().item()); tot+=1
    return zs_cell/tot,zs_ex/tot,tt_cell/tot,tt_ex/tot

print(f"Device: {DEVICE} | held-out 'gravity' | ZERO-SHOT vs TEST-TIME TRAINING | cell-chance {100/C:.0f}%")
Z=[];ZE=[];T=[];TE=[]
for s in range(3):
    a,b,c,d=run(s); Z.append(a);ZE.append(b);T.append(c);TE.append(d)
import numpy as np
Z,ZE,T,TE=map(lambda v:np.array(v),(Z,ZE,T,TE))
print("-"*56)
print(f"  ZERO-SHOT (no adapt):  cell {Z.mean():.0%} | exact {ZE.mean():.0%}")
print(f"  TEST-TIME TRAINING:    cell {T.mean():.0%} | exact {TE.mean():.0%}")
print("-"*56)
d=T.mean()-Z.mean()
print(f"TTT effect on held-out novel family: {d:+.0%} cell, {TE.mean()-ZE.mean():+.0%} exact")
print(f"VERDICT: {'TTT genuinely helps on the novel family (real, unexplored lever)' if d>0.05 else 'TTT did not help here (honest)'} -- improves NARROW results, still not AGI.")

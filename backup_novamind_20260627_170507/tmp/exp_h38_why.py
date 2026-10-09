"""
H38 — Failure analysis: WHY does it fail the novel rule? (the diagnostic the user wants)
========================================================================================

Instead of bolting on an LLM (which would just reproduce 'LLMs fail ARC-2'), we
directly diagnose OUR model's failure on the held-out novel family (gravity).

For each held-out query, we compare the model's PREDICTION against a set of
hypotheses and see which it is CLOSEST to:
  - identity (copy the input)
  - each TRAINED transform applied to the input (fliplr, flipud, rot*, rolls, a color perm)
  - the TRUE rule (gravity)

If the prediction is usually closest to a TRAINED transform or identity (NOT
gravity), the failure mode is exposed: the model INTERPOLATES rules it already
knows; it never represents the novel rule. That is exactly the ARC failure mode
(interpolative pattern-matching vs on-the-fly program synthesis), now measured
on our own system.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
G,C=5,4
def gravity(g):
    out=np.zeros_like(g)
    for c in range(g.shape[1]):
        col=[v for v in g[:,c] if v!=0]; out[g.shape[0]-len(col):,c]=col
    return out
TRAINED={'identity':lambda g:g,'fliplr':lambda g:np.fliplr(g),'flipud':lambda g:np.flipud(g),
 'rot90':lambda g:np.rot90(g,1),'rot180':lambda g:np.rot90(g,2),'rot270':lambda g:np.rot90(g,3),
 'roll_d':lambda g:np.roll(g,1,0),'roll_u':lambda g:np.roll(g,-1,0),'roll_r':lambda g:np.roll(g,1,1),'roll_l':lambda g:np.roll(g,-1,1)}
FAMILIES={'flip':[TRAINED['fliplr'],TRAINED['flipud']],'rotate':[TRAINED['rot90'],TRAINED['rot180'],TRAINED['rot270']],
 'shift':[TRAINED['roll_d'],TRAINED['roll_u'],TRAINED['roll_r'],TRAINED['roll_l']],
 'color':[(lambda p:(lambda g:p[g]))(np.random.RandomState(i+5).permutation(C)) for i in range(8)]}
TRAIN_FAMS=list(FAMILIES)
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

def task(rule,rng):
    gs=[rng.randint(0,C,(G,G)) for _ in range(4)]
    return [(g,np.array(rule(g))) for g in gs[:3]], gs[3], np.array(rule(gs[3]))

def train_base(seed):
    rng=np.random.RandomState(seed); torch.manual_seed(seed); m=Meta().to(DEVICE); opt=torch.optim.Adam(m.parameters(),1.5e-3)
    for _ in range(3000):
        Ds=[];Qs=[];Ys=[]
        for _ in range(96):
            fam=TRAIN_FAMS[rng.randint(len(TRAIN_FAMS))]; r=FAMILIES[fam][rng.randint(len(FAMILIES[fam]))]
            d,q,qo=task(r,rng); Ds.append(np.stack([np.concatenate([oh(i),oh(o)],0) for i,o in d],0)); Qs.append(oh(q)); Ys.append(qo)
        D=torch.tensor(np.array(Ds)).to(DEVICE);Q=torch.tensor(np.array(Qs)).to(DEVICE);Y=torch.tensor(np.array(Ys),dtype=torch.long).to(DEVICE)
        opt.zero_grad(); F.cross_entropy(m(D,Q),Y).backward(); opt.step()
    return m

def run(seed,n=300):
    m=train_base(seed); rng=np.random.RandomState(seed+1); m.eval()
    closest={k:0 for k in list(TRAINED)+['GRAVITY(true)']}
    match_true=0; match_best_trained=0
    for _ in range(n):
        d,q,qo=task(gravity,rng)
        D=torch.tensor(np.stack([np.concatenate([oh(i),oh(o)],0) for i,o in d],0)[None]).to(DEVICE)
        Q=torch.tensor(oh(q)[None]).to(DEVICE)
        with torch.no_grad(): pred=m(D,Q).argmax(1)[0].cpu().numpy()
        # candidates
        cands={k:np.array(f(q)) for k,f in TRAINED.items()}; cands['GRAVITY(true)']=qo
        sims={k:(pred==v).mean() for k,v in cands.items()}
        best=max(sims,key=sims.get); closest[best]+=1
        match_true+=(pred==qo).mean()
        match_best_trained+=max(sims[k] for k in TRAINED)
    print(f"  seed {seed}: pred-cell-match to TRUE gravity {match_true/n:.0%} | to best TRAINED rule {match_best_trained/n:.0%}")
    return closest,n

print(f"Device: {DEVICE} | WHY does it fail 'gravity'? what is the prediction CLOSEST to?")
agg={}
for s in range(3):
    c,n=run(s)
    for k,v in c.items(): agg[k]=agg.get(k,0)+v
tot=sum(agg.values())
print("-"*56)
print("Prediction is CLOSEST to (over all held-out tasks):")
for k,v in sorted(agg.items(),key=lambda x:-x[1]):
    if v>0: print(f"   {k:>16}: {v/tot:.0%}")
print("-"*56)
print("Diagnosis: if predictions are closest to TRAINED rules / identity (not GRAVITY),")
print("the model INTERPOLATES known transforms -- it never represents the novel rule.")
print("That IS the ARC failure mode: pattern-matching within training, not fluid synthesis.")

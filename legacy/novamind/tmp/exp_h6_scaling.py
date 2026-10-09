"""
H6 — Scaling law of compositional grounding (2-factor)
======================================================

Does zero-shot generalization improve PREDICTABLY as the model sees more
attribute values (richer support for each factor)? We sweep N in {3,4,5} for an
NxN color/shape world, hold out a fixed ~20% of combinations, and run 5 seeds
per N. Report mean +/- std. A clean monotone curve = a real scaling signal.
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB={"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"yellow":(1,1,0),"cyan":(0,1,1)}
ALL_COLORS=list(COLOR_RGB); ALL_SHAPES=["circle","square","triangle","cross","diamond"]; G=14

def shape_mask(shape):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; r=G*0.40
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=r*r: m[y,x]=1
            elif shape=="square" and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
            elif shape=="triangle" and -r<=dy<=r and abs(dx)<=(dy+r)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=r*0.32 or abs(dx)<=r*0.32): m[y,x]=1
            elif shape=="diamond" and abs(dy)+abs(dx)<=r: m[y,x]=1
    return m
MASKS={s:shape_mask(s) for s in ALL_SHAPES}

def render(colors,ci,si,rng,noise=0.05):
    m=MASKS[ALL_SHAPES[si]]
    return np.stack([m*c for c in COLOR_RGB[colors[ci]]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*noise

class Img(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.c=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(24*16,d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)
class Txt(nn.Module):
    def __init__(self,nin,d=64):
        super().__init__(); self.m=nn.Sequential(nn.Linear(nin,64),nn.ReLU(),nn.Linear(64,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)

def run(N,seed,epochs=400):
    torch.manual_seed(seed); rng=np.random.RandomState(seed)
    colors,shapes=ALL_COLORS[:N],ALL_SHAPES[:N]
    ALL=list(itertools.product(range(N),range(N)))
    nheld=max(2,int(round(0.2*len(ALL))))
    held=set();
    for t in rng.permutation(len(ALL)):
        if len(held)>=nheld: break
        held.add(ALL[t])
    TRAIN=[c for c in ALL if c not in held]; TEST=[c for c in ALL if c in held]
    def tv(ci,si):
        v=np.zeros(2*N,np.float32); v[ci]=1; v[N+si]=1; return v
    def batch(combos,reps):
        I,T=[],[]
        for (ci,si) in combos:
            for _ in range(reps): I.append(render(colors,ci,si,rng)); T.append(tv(ci,si))
        return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(T)).to(DEVICE)
    ie,te=Img().to(DEVICE),Txt(2*N).to(DEVICE)
    opt=torch.optim.Adam(list(ie.parameters())+list(te.parameters()),3e-3); tau=0.07
    for _ in range(epochs):
        im,tx=batch(TRAIN,6); zi,zt=ie(im),te(tx); lg=zi@zt.t()/tau
        lb=torch.arange(len(im),device=DEVICE)
        loss=(F.cross_entropy(lg,lb)+F.cross_entropy(lg.t(),lb))/2
        opt.zero_grad(); loss.backward(); opt.step()
    ie.eval(); te.eval()
    with torch.no_grad():
        cand=torch.tensor(np.array([tv(*c) for c in ALL])).to(DEVICE); zta=te(cand)
        hi,_=batch(TEST,40); pred=(ie(hi)@zta.t()).argmax(1).cpu().numpy()
        h=[ALL.index(c) for c in TEST for _ in range(40)]
        return float(np.mean([p==g for p,g in zip(pred,h)])), 1.0/len(ALL)

print(f"Device: {DEVICE}")
print(f"{'N (CxS)':>8} {'combos':>7} {'zero-shot mean':>15} {'std':>6} {'chance':>7}")
print("-"*52)
for N in [3,4,5]:
    accs=[]; ch=None
    for s in range(5):
        a,ch=run(N,s); accs.append(a)
    accs=np.array(accs)
    print(f"{N}x{N:>4} {N*N:>7} {accs.mean():>14.0%} {accs.std():>6.0%} {ch:>6.0%}")
print("\n(if mean rises with N -> coverage scaling law holds)")

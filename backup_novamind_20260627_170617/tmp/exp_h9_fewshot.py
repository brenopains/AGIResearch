"""
H9 — Few-shot acquisition of a NOVEL primitive (the integration test)
=====================================================================

H7 wall: with ZERO examples of a shape, the model grounds it at ~1%. The user's
hypothesis: the pieces are coming together; with the right architecture, a bit
of data should teach a new concept. Let's test exactly that.

Protocol: hold out one shape T as 'novel'. Train on all triples WITHOUT T, plus
k example triples WITH T (the 'shots'). Then test on the REMAINING T-triples
(unseen combinations of the new concept). Sweep k in {0,1,2,4,8}, 5 seeds each.

Question: does shape-accuracy on the novel concept climb with k?
  - If it rises fast -> genuine few-shot concept acquisition (real signal).
  - If it stays near floor -> the wall holds; data alone via this architecture
    does not grant new-concept grounding.

chance for shape = 1/4 = 25%.
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB={"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"yellow":(1,1,0)}
COLORS=list(COLOR_RGB); SHAPES=["circle","square","triangle","cross"]
SIZES={"small":0.30,"medium":0.42,"large":0.54}; SZ=list(SIZES)
C,S,Z=len(COLORS),len(SHAPES),len(SZ); G=16

def shape_mask(shape,rfrac):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; r=G*rfrac
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=r*r: m[y,x]=1
            elif shape=="square" and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
            elif shape=="triangle" and -r<=dy<=r and abs(dx)<=(dy+r)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=r*0.35 or abs(dx)<=r*0.35) and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
    return m
def render(ci,si,zi,rng,noise=0.05):
    m=shape_mask(SHAPES[si],SIZES[SZ[zi]])
    return np.stack([m*c for c in COLOR_RGB[COLORS[ci]]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*noise
def tv(ci,si,zi):
    v=np.zeros(C+S+Z,np.float32); v[ci]=1; v[C+si]=1; v[C+S+zi]=1; return v

class Img(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.c=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(24*16,d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)
class Txt(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.m=nn.Sequential(nn.Linear(C+S+Z,96),nn.ReLU(),nn.Linear(96,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)

ALL=list(itertools.product(range(C),range(S),range(Z)))

def run(k,seed,epochs=700):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+7)
    T=rng.randint(S)                                  # the novel shape
    base=[t for t in ALL if t[1]!=T]
    Tt=[t for t in ALL if t[1]==T]
    rng.shuffle(Tt)
    shots=Tt[:k]; test=Tt[k:]
    TRAIN=base+shots
    def batch(combos,reps):
        I,Tx=[],[]
        for (ci,si,zi) in combos:
            for _ in range(reps): I.append(render(ci,si,zi,rng)); Tx.append(tv(ci,si,zi))
        return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(Tx)).to(DEVICE)
    ie,te=Img().to(DEVICE),Txt().to(DEVICE)
    opt=torch.optim.Adam(list(ie.parameters())+list(te.parameters()),2e-3); tau=0.07
    for _ in range(epochs):
        im,tx=batch(TRAIN,4); zi,zt=ie(im),te(tx); lg=zi@zt.t()/tau
        lb=torch.arange(len(im),device=DEVICE)
        loss=(F.cross_entropy(lg,lb)+F.cross_entropy(lg.t(),lb))/2
        opt.zero_grad(); loss.backward(); opt.step()
    ie.eval(); te.eval()
    with torch.no_grad():
        cand=torch.tensor(np.array([tv(*t) for t in ALL])).to(DEVICE); zta=te(cand)
        hi,_=batch(test,40); pred=(ie(hi)@zta.t()).argmax(1).cpu().numpy()
        h=[ALL.index(t) for t in test for _ in range(40)]
        shape_acc=np.mean([ALL[p][1]==ALL[g][1] for p,g in zip(pred,h)])
        full=np.mean([p==g for p,g in zip(pred,h)])
    return shape_acc,full

print(f"Device: {DEVICE} | novel-shape acquisition | shape chance = {100/S:.0f}%")
print(f"{'k shots':>8} {'shape-acc (mean±std)':>22} {'full zero-shot':>16}")
print("-"*50)
for k in [0,1,2,4,8]:
    sh=[]; fu=[]
    for s in range(5):
        a,f=run(k,s); sh.append(a); fu.append(f)
    sh=np.array(sh); fu=np.array(fu)
    print(f"{k:>8} {sh.mean():>13.0%} ± {sh.std():>3.0%}        {fu.mean():>10.0%}")
print("\nReads: does grounding a NEW concept improve with a few real examples?")

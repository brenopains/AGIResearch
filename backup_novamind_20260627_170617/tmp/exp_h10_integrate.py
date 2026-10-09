"""
H10 — The integration test: learn a NEW concept without forgetting the OLD
==========================================================================

This is the real test of the user's "pieces are coming together" hypothesis.
Acquiring a new concept (H9) and retaining old ones (H1) must happen in ONE
system at the SAME time. Do the pieces integrate, or compete?

Protocol (genuinely online, NOT a fresh retrain):
  Phase 1: train a base model on all triples whose shape != T (3 known shapes).
           Measure OLD-concept accuracy (retrieval on held-out base triples).
  Phase 2: fine-tune the SAME model online on k=8 examples of the new shape T.
           NAIVE  : only T's examples.
           REPLAY : T's examples + a small buffer of old examples.
  Measure after Phase 2:  NEW(T) accuracy  AND  OLD-concept accuracy.

Integration succeeds only if BOTH stay high. If naive learns T but OLD collapses
-> the pieces compete (catastrophic forgetting), i.e. NOT integrated by default.
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

def make_batch(combos,reps,rng):
    I,T=[],[]
    for (ci,si,zi) in combos:
        for _ in range(reps): I.append(render(ci,si,zi,rng)); T.append(tv(ci,si,zi))
    return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(T)).to(DEVICE)

def step(ie,te,opt,combos,reps,rng,extra=None):
    im,tx=make_batch(combos,reps,rng)
    if extra:
        eim,etx=extra; im=torch.cat([im,eim]); tx=torch.cat([tx,etx])
    zi,zt=ie(im),te(tx); lg=zi@zt.t()/0.07
    lb=torch.arange(len(im),device=DEVICE)
    loss=(F.cross_entropy(lg,lb)+F.cross_entropy(lg.t(),lb))/2
    opt.zero_grad(); loss.backward(); opt.step()

def acc_on(ie,te,combos,rng):
    with torch.no_grad():
        cand=torch.tensor(np.array([tv(*t) for t in ALL])).to(DEVICE); zta=te(cand)
        hi,_=make_batch(combos,30,rng); pred=(ie(hi)@zta.t()).argmax(1).cpu().numpy()
        h=[ALL.index(t) for t in combos for _ in range(30)]
        return float(np.mean([p==g for p,g in zip(pred,h)]))

def run(seed, mode):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+3)
    T=rng.randint(S)
    old=[t for t in ALL if t[1]!=T]; new=[t for t in ALL if t[1]==T]
    rng.shuffle(new); shots=new[:8]; new_test=new[8:]
    # held-out old test combos (a fraction never trained, to measure real generalization)
    rng.shuffle(old); old_test=old[:8]; old_train=old[8:]

    ie,te=Img().to(DEVICE),Txt().to(DEVICE)
    opt=torch.optim.Adam(list(ie.parameters())+list(te.parameters()),2e-3)
    for _ in range(600): step(ie,te,opt,old_train,4,rng)
    old_before=acc_on(ie,te,old_test,rng)

    # Phase 2: online fine-tune on the new concept
    buf=None
    if mode=="replay":
        buf=make_batch(old_train[:12],2,rng)
    opt2=torch.optim.Adam(list(ie.parameters())+list(te.parameters()),2e-3)
    for _ in range(200): step(ie,te,opt2,shots,4,rng,extra=buf)

    new_acc=acc_on(ie,te,new_test,rng)
    old_after=acc_on(ie,te,old_test,rng)
    return old_before,old_after,new_acc

print(f"Device: {DEVICE} | chance {100/len(ALL):.1f}%")
print(f"{'mode':>8} {'OLD before':>11} {'OLD after':>10} {'NEW(T)':>8}  verdict")
print("-"*56)
for mode in ["naive","replay"]:
    ob,oa,na=[],[],[]
    for s in range(5):
        b,a,n=run(s,mode); ob.append(b); oa.append(a); na.append(n)
    ob,oa,na=np.array(ob),np.array(oa),np.array(na)
    kept=oa.mean()/max(ob.mean(),1e-9)
    verdict="integrates" if (oa.mean()>0.6 and na.mean()>0.5) else "competes"
    print(f"{mode:>8} {ob.mean():>10.0%} {oa.mean():>10.0%} {na.mean():>8.0%}  {verdict} (kept {kept:.0%} of old)")
print("\nIntegration = learning the NEW concept while KEEPING the old ones.")

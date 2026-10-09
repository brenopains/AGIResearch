"""
H12 — Which curiosity signal actually helps? (refining H11's negative result)
=============================================================================

H11 showed ERROR-seeking curiosity ('learn what you're worst at') ties random.
H6 showed generalization is driven by COVERAGE of attribute values. So the
refined hypothesis: a curiosity signal that seeks NOVELTY/COVERAGE (explore
under-represented attributes) should beat random, even though error-seeking
didn't.

Three agents, identical labeling budget, fixed held-out TEST:
  RANDOM        : uniform random concept choice.
  ERROR-CURIOUS : acquire the concept it currently scores worst on (H11).
  NOVELTY-CURIOUS: acquire the concept whose attribute values are least
                   represented so far (maximize coverage). Uses only the agent's
                   own observed attribute structure -- no peeking at TEST.

If NOVELTY beats both -> 'the right curiosity signal matters', a real and
honest nuance (autonomy helps only when the drive is well-chosen).
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

class Agent:
    def __init__(self,seed):
        torch.manual_seed(seed); self.rng=np.random.RandomState(seed+11)
        self.ie,self.te=Img().to(DEVICE),Txt().to(DEVICE)
        self.opt=torch.optim.Adam(list(self.ie.parameters())+list(self.te.parameters()),2e-3)
    def batch(self,combos,reps):
        I,T=[],[]
        for (ci,si,zi) in combos:
            for _ in range(reps): I.append(render(ci,si,zi,self.rng)); T.append(tv(ci,si,zi))
        return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(T)).to(DEVICE)
    def train(self,combos,steps):
        if not combos: return
        for _ in range(steps):
            im,tx=self.batch(combos,4); zi,zt=self.ie(im),self.te(tx); lg=zi@zt.t()/0.07
            lb=torch.arange(len(im),device=DEVICE)
            loss=(F.cross_entropy(lg,lb)+F.cross_entropy(lg.t(),lb))/2
            self.opt.zero_grad(); loss.backward(); self.opt.step()
    def acc(self,combos):
        self.ie.eval(); self.te.eval()
        with torch.no_grad():
            cand=torch.tensor(np.array([tv(*t) for t in ALL])).to(DEVICE); zta=self.te(cand)
            hi,_=self.batch(combos,30); pred=(self.ie(hi)@zta.t()).argmax(1).cpu().numpy()
            h=[ALL.index(t) for t in combos for _ in range(30)]
            a=float(np.mean([p==g for p,g in zip(pred,h)]))
        self.ie.train(); self.te.train(); return a
    def error(self,combo):
        self.ie.eval(); self.te.eval()
        with torch.no_grad():
            cand=torch.tensor(np.array([tv(*t) for t in ALL])).to(DEVICE); zta=self.te(cand)
            hi,_=self.batch([combo],6); pred=(self.ie(hi)@zta.t()).argmax(1).cpu().numpy()
            e=1.0-float(np.mean([ALL[p]==combo for p in pred]))
        self.ie.train(); self.te.train(); return e

def coverage_pick(cands, acquired, rng):
    """Pick concept whose attribute values are least represented in acquired."""
    cc=np.zeros(C); sc=np.zeros(S); zc=np.zeros(Z)
    for (c,s,z) in acquired: cc[c]+=1; sc[s]+=1; zc[z]+=1
    best=None; best_score=1e9
    for (c,s,z) in cands:
        score=cc[c]+sc[s]+zc[z]+rng.rand()*0.01   # less represented = better
        if score<best_score: best_score=score; best=(c,s,z)
    return best

def run(seed, mode, budget=16, seed_n=4):
    rng=np.random.RandomState(seed)
    perm=list(rng.permutation(len(ALL)))
    TEST=[ALL[i] for i in perm[:8]]; POOL=[ALL[i] for i in perm[8:]]
    ag=Agent(seed)
    acquired=[POOL[i] for i in rng.permutation(len(POOL))[:seed_n]]
    ag.train(acquired, steps=200)
    if mode=="random":
        rem=[c for c in POOL if c not in acquired]
        acquired+=[rem[i] for i in rng.permutation(len(rem))[:budget-seed_n]]
        ag.train(acquired, steps=500)
    elif mode=="error":
        while len(acquired)<budget:
            cands=[c for c in POOL if c not in acquired]
            cands.sort(key=lambda c:-ag.error(c)); acquired.append(cands[0])
            ag.train(acquired, steps=120)
        ag.train(acquired, steps=200)
    else:  # novelty/coverage
        while len(acquired)<budget:
            cands=[c for c in POOL if c not in acquired]
            acquired.append(coverage_pick(cands, acquired, rng))
            ag.train(acquired, steps=120)
        ag.train(acquired, steps=200)
    return ag.acc(TEST)

print(f"Device: {DEVICE} | budget=16 of 40 | chance {100/len(ALL):.1f}%")
print(f"{'mode':>16} {'held-out acc (mean±std)':>26}")
print("-"*46)
res={}
for mode in ["random","error","novelty"]:
    accs=np.array([run(s,mode) for s in range(5)]); res[mode]=accs
    print(f"{mode:>16} {accs.mean():>16.0%} ± {accs.std():>3.0%}")
print(f"\nnovelty vs random: {res['novelty'].mean()-res['random'].mean():+.0%}")
print(f"error   vs random: {res['error'].mean()-res['random'].mean():+.0%}")
print("Honest: same budget. If novelty>random>~error, the curiosity SIGNAL is what matters.")

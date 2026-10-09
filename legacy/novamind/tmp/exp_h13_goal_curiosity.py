"""
H13 — Goal-directed curiosity (testing the user's Q4 introspection)
==================================================================

H11/H12 found that GOAL-FREE curiosity (error- or novelty-seeking) ties random.
The user's introspection (Q4): human curiosity is ALWAYS subordinated to a goal
('I want a Jarvis -> ... -> survival'), and it 'hooks' when there's progress
TOWARD the answer. Falsifiable prediction: with a GOAL, directed exploration
should beat undirected (random).

Setup: the agent has a GOAL concept G it must learn to recognize, but it may
NEVER directly acquire G (G is held out). It has a budget to acquire OTHER
concepts. To recognize G zero-shot it must ground G's parts (its color, shape,
size) from other combinations.

  RANDOM        : acquire concepts uniformly at random.
  GOAL-DIRECTED : acquire concepts that share attributes with G (build the parts
                  needed to compose G) -- curiosity steered by the goal.

Metric: zero-shot accuracy at recognizing G. If goal-directed > random, the
user's introspective claim holds in this toy world. Multi-seed for power.
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB={"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"yellow":(1,1,0)}
COLORS=list(COLOR_RGB); SHAPES=["circle","square","triangle","cross"]
SIZES={"small":0.30,"medium":0.42,"large":0.54}; SZ=list(SIZES)
C,S,Z=len(COLORS),len(SHAPES),len(SZ); G_=16

def shape_mask(shape,rfrac):
    m=np.zeros((G_,G_),np.float32); cy=cx=(G_-1)/2; r=G_*rfrac
    for y in range(G_):
        for x in range(G_):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=r*r: m[y,x]=1
            elif shape=="square" and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
            elif shape=="triangle" and -r<=dy<=r and abs(dx)<=(dy+r)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=r*0.35 or abs(dx)<=r*0.35) and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
    return m
def render(ci,si,zi,rng,noise=0.05):
    m=shape_mask(SHAPES[si],SIZES[SZ[zi]])
    return np.stack([m*c for c in COLOR_RGB[COLORS[ci]]],0)+rng.standard_normal((3,G_,G_)).astype(np.float32)*noise
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

def shared_attrs(a,b):
    return int(a[0]==b[0])+int(a[1]==b[1])+int(a[2]==b[2])

def run(seed, mode, budget=10):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+5)
    G=ALL[rng.randint(len(ALL))]
    POOL=[c for c in ALL if c!=G]
    if mode=="random":
        acquired=[POOL[i] for i in rng.permutation(len(POOL))[:budget]]
    else:  # goal-directed: prefer concepts sharing attributes with G
        scored=sorted(POOL, key=lambda c:(-shared_attrs(c,G), rng.rand()))
        acquired=scored[:budget]
    ie,te=Img().to(DEVICE),Txt().to(DEVICE)
    opt=torch.optim.Adam(list(ie.parameters())+list(te.parameters()),2e-3)
    def batch(combos,reps):
        I,T=[],[]
        for (ci,si,zi) in combos:
            for _ in range(reps): I.append(render(ci,si,zi,rng)); T.append(tv(ci,si,zi))
        return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(T)).to(DEVICE)
    for _ in range(600):
        im,tx=batch(acquired,4); zi,zt=ie(im),te(tx); lg=zi@zt.t()/0.07
        lb=torch.arange(len(im),device=DEVICE)
        loss=(F.cross_entropy(lg,lb)+F.cross_entropy(lg.t(),lb))/2
        opt.zero_grad(); loss.backward(); opt.step()
    ie.eval(); te.eval()
    with torch.no_grad():
        cand=torch.tensor(np.array([tv(*t) for t in ALL])).to(DEVICE); zta=te(cand)
        gi,_=batch([G],40); pred=(ie(gi)@zta.t()).argmax(1).cpu().numpy()
        acc_G=float(np.mean([ALL[p]==G for p in pred]))
    return acc_G

print(f"Device: {DEVICE} | budget=10 | recognize the GOAL concept zero-shot | chance {100/len(ALL):.1f}%")
print(f"{'mode':>14} {'goal acc (mean±std)':>22}")
print("-"*40)
res={}
for mode in ["random","goal_directed"]:
    accs=np.array([run(s,mode) for s in range(8)]); res[mode]=accs
    print(f"{mode:>14} {accs.mean():>13.0%} ± {accs.std():>3.0%}")
d=res["goal_directed"].mean()-res["random"].mean()
print(f"\ngoal-directed vs random: {d:+.0%}")
print("Honest: same budget. Tests whether steering exploration by a GOAL beats undirected.")

"""
H7 — Break generalization on purpose (find the wall)
====================================================

Random held-out triples gave ~62% zero-shot (the model recombines SEEN parts).
But what if a part is NEVER seen? We compare two splits on the 4x4x3 world:

  RANDOM   : hold out random triples (every attribute value still trained).
             -> model can recombine -> expect well above chance.
  NOVEL-PRIMITIVE : remove an ENTIRE shape value from ALL training triples.
             The text one-hot for that shape is never trained.
             -> model has no grounding for it -> expect collapse toward chance
                on that shape, even though color/size are fine.

This isolates the real limit: current models recombine known primitives but
cannot ground a primitive they've never observed. Multi-seed, honest.
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

def train_eval(TRAIN,TEST,seed,epochs=700):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+100)
    def batch(combos,reps):
        I,T=[],[]
        for (ci,si,zi) in combos:
            for _ in range(reps): I.append(render(ci,si,zi,rng)); T.append(tv(ci,si,zi))
        return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(T)).to(DEVICE)
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
        hi,_=batch(TEST,40); pred=(ie(hi)@zta.t()).argmax(1).cpu().numpy()
        h=[ALL.index(t) for t in TEST for _ in range(40)]
        zs=np.mean([p==g for p,g in zip(pred,h)]); ca=np.zeros(3)
        for p,g in zip(pred,h):
            for k in range(3): ca[k]+=(ALL[p][k]==ALL[g][k])
        ca/=len(h)
    return zs,ca

def random_split(seed):
    rng=np.random.RandomState(seed); held=set()
    for t in rng.permutation(len(ALL)):
        if len(held)>=7: break
        held.add(ALL[t])
    return [t for t in ALL if t not in held],[t for t in ALL if t in held]

def novel_primitive_split(seed):
    rng=np.random.RandomState(seed); drop_shape=rng.randint(S)
    TEST=[t for t in ALL if t[1]==drop_shape]
    TRAIN=[t for t in ALL if t[1]!=drop_shape]
    return TRAIN,TEST,drop_shape

print(f"Device: {DEVICE} | chance {100/len(ALL):.1f}%\n{'split':>16} {'zero-shot':>10} {'color':>6} {'shape':>6} {'size':>6}")
print("-"*52)
zr=[]; cr=[]
for s in range(5):
    tr,te=random_split(s); zs,ca=train_eval(tr,te,s); zr.append(zs); cr.append(ca)
zr=np.array(zr); cr=np.array(cr)
print(f"{'RANDOM':>16} {zr.mean():>9.0%} {cr[:,0].mean():>6.0%} {cr[:,1].mean():>6.0%} {cr[:,2].mean():>6.0%}")
zn=[]; cn=[]
for s in range(5):
    tr,te,ds=novel_primitive_split(s); zs,ca=train_eval(tr,te,s); zn.append(zs); cn.append(ca)
zn=np.array(zn); cn=np.array(cn)
print(f"{'NOVEL-PRIMITIVE':>16} {zn.mean():>9.0%} {cn[:,0].mean():>6.0%} {cn[:,1].mean():>6.0%} {cn[:,2].mean():>6.0%}")
print(f"\nshape-acc collapse: random {cr[:,1].mean():.0%} -> novel-primitive {cn[:,1].mean():.0%} (chance {100/S:.0f}%)")
print("Interpretation: recombining seen parts works; grounding an UNSEEN primitive does not.")

"""
EXPERIMENT 3e — Honest error bars (the step the 5 prior sessions skipped)
=========================================================================

v3 (seed 0) said shape generalization = 35%. v4 (seed reset) said 81%. SAME
architecture. That gap = high variance on tiny data. A single run is not a
result. So we run the WEAK 3-factor grounding across MANY seeds and report the
distribution. Only then can we honestly say what the signal IS.

Reports: mean +/- std, min, max over seeds, for zero-shot top-1 and per-attribute.
Also varies the random held-out split per seed so we're not memorizing one split.
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def text_vec(ci,si,zi):
    v=np.zeros(C+S+Z,np.float32); v[ci]=1; v[C+si]=1; v[C+S+zi]=1; return v

class WeakImg(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),
                             nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(24*16,d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)

class TxtEnc(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.m=nn.Sequential(nn.Linear(C+S+Z,96),nn.ReLU(),nn.Linear(96,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)

ALL=list(itertools.product(range(C),range(S),range(Z)))

def one_seed(seed,epochs=700):
    torch.manual_seed(seed); np.random.seed(seed)
    rng=np.random.RandomState(seed)
    held=set()
    for t in rng.permutation(len(ALL)):
        if len(held)>=7: break
        held.add(ALL[t])
    TRAIN=[t for t in ALL if t not in held]; TEST=[t for t in ALL if t in held]
    def batch(combos,reps):
        I,T=[],[]
        for (ci,si,zi) in combos:
            for _ in range(reps): I.append(render(ci,si,zi,rng)); T.append(text_vec(ci,si,zi))
        return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(T)).to(DEVICE)
    ie,te=WeakImg().to(DEVICE),TxtEnc().to(DEVICE)
    opt=torch.optim.Adam(list(ie.parameters())+list(te.parameters()),2e-3); tau=0.07
    for _ in range(epochs):
        imgs,txts=batch(TRAIN,4); zi,zt=ie(imgs),te(txts)
        logits=zi@zt.t()/tau; lbl=torch.arange(len(imgs),device=DEVICE)
        loss=(F.cross_entropy(logits,lbl)+F.cross_entropy(logits.t(),lbl))/2
        opt.zero_grad(); loss.backward(); opt.step()
    ie.eval(); te.eval()
    with torch.no_grad():
        cand=torch.tensor(np.array([text_vec(*t) for t in ALL])).to(DEVICE); zt_all=te(cand)
        hi,_=batch(TEST,40); pred=(ie(hi)@zt_all.t()).argmax(1).cpu().numpy()
        h=[ALL.index(t) for t in TEST for _ in range(40)]
        zs=np.mean([p==g for p,g in zip(pred,h)])
        ca=np.zeros(3)
        for p,g in zip(pred,h):
            for k in range(3): ca[k]+=(ALL[p][k]==ALL[g][k])
        ca/=len(h)
    return zs,ca

SEEDS=list(range(8))
zss=[]; cas=[]
for s in SEEDS:
    zs,ca=one_seed(s); zss.append(zs); cas.append(ca)
    print(f"  seed {s}: zero-shot {zs:.0%} | color {ca[0]:.0%} shape {ca[1]:.0%} size {ca[2]:.0%}")
zss=np.array(zss); cas=np.array(cas)
print("="*60)
print(f"zero-shot top-1: mean {zss.mean():.0%}  std {zss.std():.0%}  min {zss.min():.0%}  max {zss.max():.0%}  (chance {100/len(ALL):.1f}%)")
print(f"per-attr mean:   color {cas[:,0].mean():.0%}  shape {cas[:,1].mean():.0%}  size {cas[:,2].mean():.0%}")
print(f"shape range over seeds: {cas[:,1].min():.0%} .. {cas[:,1].max():.0%}")
print("="*60)

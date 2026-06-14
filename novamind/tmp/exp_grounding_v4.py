"""
EXPERIMENT 3d — H5: is SHAPE the bottleneck because the encoder smears geometry?
================================================================================

v3 found: on unseen (color,shape,size) triples the model nails color (100%) and
size (100%) but shape only 35%. Hypothesis H5: aggressive global pooling in the
image encoder averages away spatial geometry -> color/size (global statistics)
survive, shape (spatial) is lost.

Test: SAME data, SAME everything, two image encoders side by side:
  WEAK    = AdaptiveAvgPool2d(4)              (v3-style, smears geometry)
  SPATIAL = stronger conv + maxpool, keeps 8x8 spatial detail before flatten
Prediction: SPATIAL raises per-attribute SHAPE accuracy on unseen triples.

Honest: report both, per attribute. If SPATIAL doesn't help shape, H5 is wrong.
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
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

def render(ci,si,zi,noise=0.05):
    m=shape_mask(SHAPES[si],SIZES[SZ[zi]])
    return np.stack([m*c for c in COLOR_RGB[COLORS[ci]]],0)+np.random.randn(3,G,G).astype(np.float32)*noise

def text_vec(ci,si,zi):
    v=np.zeros(C+S+Z,np.float32); v[ci]=1; v[C+si]=1; v[C+S+zi]=1; return v

class WeakImg(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),
                             nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(24*16,d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)

class SpatialImg(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.c=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),   # 16->8
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),                  # keep 8x8
            nn.Flatten(),nn.Linear(32*8*8,d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)

class TxtEnc(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.m=nn.Sequential(nn.Linear(C+S+Z,96),nn.ReLU(),nn.Linear(96,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)

ALL=list(itertools.product(range(C),range(S),range(Z)))
rng=np.random.RandomState(1); held=set()
for t in rng.permutation(len(ALL)):
    if len(held)>=max(6,int(0.15*len(ALL))): break
    held.add(ALL[t])
TRAIN=[t for t in ALL if t not in held]; TEST=[t for t in ALL if t in held]

def batch(combos,reps):
    I,T=[],[]
    for (ci,si,zi) in combos:
        for _ in range(reps): I.append(render(ci,si,zi)); T.append(text_vec(ci,si,zi))
    return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(T)).to(DEVICE)

def train_eval(ImgCls,epochs=700):
    torch.manual_seed(0)
    ie,te=ImgCls().to(DEVICE),TxtEnc().to(DEVICE)
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
        h_lbl=[ALL.index(t) for t in TEST for _ in range(40)]
        zs=np.mean([p==g for p,g in zip(pred,h_lbl)])
        ca=np.zeros(3)
        for p,g in zip(pred,h_lbl):
            for k in range(3): ca[k]+=(ALL[p][k]==ALL[g][k])
        ca/=len(h_lbl)
    return zs,ca

print(f"Device: {DEVICE} | held-out triples: {len(TEST)} | chance {100/len(ALL):.1f}%")
print(f"{'encoder':>8} | {'zero-shot':>9} | color  shape  size")
print("-"*48)
for name,cls in [("WEAK",WeakImg),("SPATIAL",SpatialImg)]:
    zs,ca=train_eval(cls)
    print(f"{name:>8} | {zs:>8.0%}  | {ca[0]:>4.0%}  {ca[1]:>4.0%}  {ca[2]:>4.0%}")
print(f"\nVRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU")

"""
EXPERIMENT 3c — H4: does compositional grounding survive a THIRD factor?
========================================================================

v2 showed 80% zero-shot on unseen (color, shape) pairs (20x chance). Now we
raise the bar: 3 factors -> (color, shape, SIZE). A real world-model should
bind THREE independent attributes and recombine them. We hold out triples
never seen together and test zero-shot retrieval over ALL color*shape*size
candidates.

Hypothesis H4: if the shared space truly factorizes the world, zero-shot on
unseen triples stays well above chance even though chance is now tiny.

Self-contained. Honest result either way.
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB = {"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"yellow":(1,1,0)}
COLORS = list(COLOR_RGB)
SHAPES = ["circle","square","triangle","cross"]
SIZES  = {"small":0.28, "medium":0.40, "large":0.52}   # mask radius fraction
SZ = list(SIZES)
C, S, Z = len(COLORS), len(SHAPES), len(SZ)
G = 16


def shape_mask(shape, rfrac):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; r=G*rfrac
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=r*r: m[y,x]=1
            elif shape=="square" and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
            elif shape=="triangle" and dy>=-r and dy<=r and abs(dx)<=(dy+r)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=r*0.35 or abs(dx)<=r*0.35) and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
    return m


def render(ci,si,zi,noise=0.05):
    m=shape_mask(SHAPES[si], SIZES[SZ[zi]])
    img=np.stack([m*c for c in COLOR_RGB[COLORS[ci]]],0)
    return img+np.random.randn(*img.shape).astype(np.float32)*noise


def text_vec(ci,si,zi):
    v=np.zeros(C+S+Z,np.float32); v[ci]=1; v[C+si]=1; v[C+S+zi]=1; return v


class ImgEnc(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),
                             nn.Conv2d(24,24,3,padding=1),nn.ReLU(),
                             nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(24*16,d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)

class TxtEnc(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.m=nn.Sequential(nn.Linear(C+S+Z,96),nn.ReLU(),nn.Linear(96,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)


ALL=list(itertools.product(range(C),range(S),range(Z)))
rng=np.random.RandomState(1)
# hold out ~15% of triples at random, ensuring each attribute value still appears in train
held=set()
for t in rng.permutation(len(ALL)):
    cand=ALL[t]
    if len(held)>= max(6, int(0.15*len(ALL))): break
    held.add(cand)
TRAIN=[t for t in ALL if t not in held]
TEST=[t for t in ALL if t in held]


def batch(combos,reps):
    I,T=[],[]
    for (ci,si,zi) in combos:
        for _ in range(reps):
            I.append(render(ci,si,zi)); T.append(text_vec(ci,si,zi))
    return torch.tensor(np.array(I)).to(DEVICE), torch.tensor(np.array(T)).to(DEVICE)


ie,te=ImgEnc().to(DEVICE),TxtEnc().to(DEVICE)
opt=torch.optim.Adam(list(ie.parameters())+list(te.parameters()),2e-3)
tau=0.07
print(f"Device: {DEVICE} | {C}x{S}x{Z}={len(ALL)} triples | train {len(TRAIN)} held-out {len(TEST)}")
for ep in range(700):
    imgs,txts=batch(TRAIN,4)
    zi,zt=ie(imgs),te(txts)
    logits=zi@zt.t()/tau; lbl=torch.arange(len(imgs),device=DEVICE)
    loss=(F.cross_entropy(logits,lbl)+F.cross_entropy(logits.t(),lbl))/2
    opt.zero_grad(); loss.backward(); opt.step()

ie.eval(); te.eval()
with torch.no_grad():
    cand=torch.tensor(np.array([text_vec(*t) for t in ALL])).to(DEVICE)
    zt_all=te(cand)
    ti,_=batch(TRAIN,10); s_tr=ie(ti)@zt_all.t()
    tr_lbl=torch.tensor([ALL.index(t) for t in TRAIN for _ in range(10)],device=DEVICE)
    seen=(s_tr.argmax(1)==tr_lbl).float().mean().item()
    hi,_=batch(TEST,40); s_h=ie(hi)@zt_all.t()
    h_lbl=torch.tensor([ALL.index(t) for t in TEST for _ in range(40)],device=DEVICE)
    pred=s_h.argmax(1)
    zs=(pred==h_lbl).float().mean().item()
    # partial credit: how many of the 3 attributes are right on average
    def attrs(idx): t=ALL[idx]; return t
    correct_attr=np.zeros(3)
    for p,g in zip(pred.cpu().numpy(), h_lbl.cpu().numpy()):
        for k in range(3): correct_attr[k]+= (ALL[p][k]==ALL[g][k])
    correct_attr/=len(h_lbl)

chance=100.0/len(ALL)
print("="*60)
print(f"Seen triples    | top-1: {seen:.0%}")
print(f"UNSEEN triples  | zero-shot top-1: {zs:.0%}   (chance {chance:.1f}%, {zs/(chance/100):.0f}x)")
print(f"per-attribute right on unseen: color {correct_attr[0]:.0%} | shape {correct_attr[1]:.0%} | size {correct_attr[2]:.0%}")
print("="*60)
print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU run")

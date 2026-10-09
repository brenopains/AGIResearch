"""
H17 — Scaling study: what scales vs what doesn't (the user's map)
================================================================

Falsifiable test of "scaling -> generality". Vary model capacity (width x0.5..x4)
and measure TWO tasks at each scale:

  TASK A (narrow): compositional generalization. Random held-out (color,shape,size)
                   triples; every attribute value WAS seen in training.
  TASK B (the wall): novel-primitive grounding. A whole SHAPE is never seen in
                     training; test recognizing that shape's combos.

Factored anchors (anchor = E_color[c]+E_shape[s]+E_size[z]) so held-out concepts
have valid anchors. Image-only (color/shape/size all visible here).

PREDICTION (pre-registered):
  A improves with scale (a narrow scaling law).
  B stays ~chance regardless of scale (you cannot ground a never-observed
  primitive: its E_shape never trains, its image never seen -> no signal to scale).
If B rises with scale -> a genuine (weak) pro-scaling signal; reported honestly.
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB=[(1,0,0),(0,1,0),(0,0,1),(1,1,0)]
SHAPES=["circle","square","triangle","cross"]
SIZES=[0.30,0.42,0.54]
C,S,Z=4,4,3; G=18
ALL=list(itertools.product(range(C),range(S),range(Z)))
CS=torch.tensor([t[0] for t in ALL]);SS=torch.tensor([t[1] for t in ALL]);ZS=torch.tensor([t[2] for t in ALL])
def cid(t): return ALL.index(t)

def shape_mask(shape,r):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; rr=G*r
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=rr*rr: m[y,x]=1
            elif shape=="square" and abs(dy)<=rr and abs(dx)<=rr: m[y,x]=1
            elif shape=="triangle" and -rr<=dy<=rr and abs(dx)<=(dy+rr)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=rr*0.32 or abs(dx)<=rr*0.32) and abs(dy)<=rr and abs(dx)<=rr: m[y,x]=1
    return m
def render(c,s,z,rng,noise=0.06):
    m=shape_mask(SHAPES[s],SIZES[z])
    return np.stack([m*ch for ch in COLOR_RGB[c]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*noise

class ImgEnc(nn.Module):
    def __init__(s,width=1.0):
        super().__init__(); ch=max(4,int(16*width)); d=max(8,int(32*width))
        s.d=d; s.c=nn.Sequential(nn.Conv2d(3,ch,3,padding=1),nn.ReLU(),
            nn.Conv2d(ch,ch,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(ch*16,d))
    def forward(s,x): return F.normalize(s.c(x),dim=1)

def train_eval(width, mode, seed, steps=1200, per_cell=8, pool=48):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+9)
    if mode=="A":  # compositional: random held-out, all attrs seen
        held=set()
        for i in rng.permutation(len(ALL)):
            if len(held)>=8: break
            held.add(ALL[i])
        TRAIN=[t for t in ALL if t not in held]; TEST=[t for t in ALL if t in held]
    else:          # novel primitive: a whole shape never trained
        drop=rng.randint(S)
        TRAIN=[t for t in ALL if t[1]!=drop]; TEST=[t for t in ALL if t[1]==drop]
    Itr=torch.tensor(np.array([[render(*t,rng) for _ in range(pool)] for t in TRAIN])).to(DEVICE)
    d=ImgEnc(width).d
    ie=ImgEnc(width).to(DEVICE)
    Ecol=nn.Parameter(torch.randn(C,d).to(DEVICE)*0.3)
    Esha=nn.Parameter(torch.randn(S,d).to(DEVICE)*0.3)
    Esiz=nn.Parameter(torch.randn(Z,d).to(DEVICE)*0.3)
    csd,ssd,zsd=CS.to(DEVICE),SS.to(DEVICE),ZS.to(DEVICE)
    anchors=lambda: F.normalize(Ecol[csd]+Esha[ssd]+Esiz[zsd],dim=1)
    opt=torch.optim.Adam(list(ie.parameters())+[Ecol,Esha,Esiz],1.5e-3)
    tr_ids=torch.tensor([cid(t) for t in TRAIN],device=DEVICE).repeat_interleave(per_cell)
    base=torch.arange(len(TRAIN),device=DEVICE).repeat_interleave(per_cell)
    for _ in range(steps):
        idx=torch.randint(0,pool,(len(TRAIN)*per_cell,),device=DEVICE)
        anc=anchors(); zi=ie(Itr[base,idx])
        loss=F.cross_entropy(zi@anc.t()/0.07,tr_ids)
        opt.zero_grad(); loss.backward(); opt.step()
    ie.eval(); anc=anchors().detach()
    with torch.no_grad():
        accs=[]
        for t in TEST:
            I=torch.tensor(np.array([render(*t,rng) for _ in range(40)])).to(DEVICE)
            pred=(ie(I)@anc.t()).argmax(1).cpu().numpy()
            if mode=="A": accs.append((pred==cid(t)).mean())                       # full concept
            else: accs.append(np.mean([ALL[p][1]==t[1] for p in pred]))            # shape correct
        return float(np.mean(accs))

print(f"Device: {DEVICE}")
print(f"{'width':>6} {'paramsD':>8} | {'A: composition':>15} | {'B: novel primitive (shape)':>27}")
print("-"*60)
for w in [0.5,1,2,4]:
    A=np.mean([train_eval(w,'A',s) for s in range(4)])
    B=np.mean([train_eval(w,'B',s) for s in range(4)])
    print(f"x{w:<5} {ImgEnc(w).d:>8} | {A:>14.0%} | {B:>20.0%}   (chance {100/S:.0f}%)")
print("\nPrediction: A rises with scale; B stays ~chance (scale can't ground the never-seen).")
print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU")

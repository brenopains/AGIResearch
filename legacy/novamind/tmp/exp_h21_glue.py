"""
H21 — The "glue" attempt: can ONE shared architecture serve many capabilities?
==============================================================================

User's request: use the search algorithm to 'glue everything'. Honest proxy:
search for ONE shared encoder architecture that supports MULTIPLE capabilities
at once (recognize color, shape, AND size), and ask whether sharing HELPS
(positive transfer = a weak integration signal) or HURTS (interference =
capabilities fight for the same representation).

  SPECIALISTS : 3 separate encoders, each trained on one factor.
  SHARED      : 1 encoder + 3 heads, trained on all factors jointly.
Both evaluated on HELD-OUT (color,shape,size) combinations (compositional test).
A small GA searches the shared architecture (width/depth/dropout).

HONEST: this is a proxy for integration, not 'the cola'. Likely outcome: shared
~ specialists (mild transfer or mild interference), confirming there's no free
integration bonus. Reported as-is.
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB=[(1,0,0),(0,1,0),(0,0,1),(1,1,0)]
SHAPES=["circle","square","triangle","cross"]; SIZES=[0.30,0.42,0.54]
C,S,Z=4,4,3; G=18
ALL=list(itertools.product(range(C),range(S),range(Z)))

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

class Enc(nn.Module):
    def __init__(s,width,depth,dp,nheads_dims):
        super().__init__(); ch=width
        layers=[nn.Conv2d(3,ch,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)]
        for _ in range(depth-1): layers+=[nn.Conv2d(ch,ch,3,padding=1),nn.ReLU()]
        layers+=[nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Dropout(dp),nn.Linear(ch*16,64)]
        s.b=nn.Sequential(*layers)
        s.heads=nn.ModuleList([nn.Linear(64,d) for d in nheads_dims])
    def forward(s,x):
        h=s.b(x); return [head(h) for head in s.heads]

def cache(combos,rng,pool=48):
    return torch.tensor(np.array([[render(*t,rng) for _ in range(pool)] for t in combos])).to(DEVICE)

def split(seed):
    rng=np.random.RandomState(seed); held=set()
    for i in rng.permutation(len(ALL)):
        if len(held)>=8: break
        held.add(ALL[i])
    return [t for t in ALL if t not in held],[t for t in ALL if t in held]

def train_eval(g, seed, shared=True):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+1)
    TRAIN,TEST=split(seed)
    Itr=cache(TRAIN,rng); pool=Itr.shape[1]
    labels={'c':torch.tensor([t[0] for t in TRAIN],device=DEVICE),
            's':torch.tensor([t[1] for t in TRAIN],device=DEVICE),
            'z':torch.tensor([t[2] for t in TRAIN],device=DEVICE)}
    dims=[C,S,Z]
    if shared:
        enc=Enc(g['width'],g['depth'],g['dropout'],dims).to(DEVICE)
        opt=torch.optim.Adam(enc.parameters(),2e-3)
        base=torch.arange(len(TRAIN),device=DEVICE).repeat_interleave(8)
        lc=labels['c'].repeat_interleave(8); ls=labels['s'].repeat_interleave(8); lz=labels['z'].repeat_interleave(8)
        for _ in range(700):
            idx=torch.randint(0,pool,(len(TRAIN)*8,),device=DEVICE)
            oc,os,oz=enc(Itr[base,idx])
            loss=F.cross_entropy(oc,lc)+F.cross_entropy(os,ls)+F.cross_entropy(oz,lz)
            opt.zero_grad(); loss.backward(); opt.step()
        encs=[enc]
    else:  # specialists: one encoder per factor
        encs=[]
        for fi,dim in enumerate(dims):
            e=Enc(g['width'],g['depth'],g['dropout'],[dim]).to(DEVICE)
            opt=torch.optim.Adam(e.parameters(),2e-3)
            lab=[labels['c'],labels['s'],labels['z']][fi].repeat_interleave(8)
            base=torch.arange(len(TRAIN),device=DEVICE).repeat_interleave(8)
            for _ in range(700):
                idx=torch.randint(0,pool,(len(TRAIN)*8,),device=DEVICE)
                out=e(Itr[base,idx])[0]
                loss=F.cross_entropy(out,lab)
                opt.zero_grad(); loss.backward(); opt.step()
            encs.append(e)
    # eval on held-out combos
    for e in encs: e.eval()
    accs=[0,0,0]
    with torch.no_grad():
        for t in TEST:
            I=torch.tensor(np.array([render(*t,rng) for _ in range(40)])).to(DEVICE)
            if shared:
                outs=encs[0](I)
            else:
                outs=[encs[fi](I)[0] for fi in range(3)]
            for fi,gt in enumerate(t):
                accs[fi]+=(outs[fi].argmax(1).cpu().numpy()==gt).mean()
    return np.array(accs)/len(TEST)

SPACE={'width':[16,32,64],'depth':[1,2],'dropout':[0.0,0.1,0.3]}
def rand_g(rng): return {k:v[rng.randint(len(v))] for k,v in SPACE.items()}

print(f"Device: {DEVICE} | held-out compositional eval | chance: color/shape 25%, size 33%")
# small GA for the SHARED architecture
rng=np.random.RandomState(0); pop=[rand_g(rng) for _ in range(6)]; best=None
for gen in range(3):
    scored=[]
    for g in pop:
        m=np.mean([train_eval(g,s,shared=True).mean() for s in range(2)])
        scored.append((m,g))
    scored.sort(key=lambda x:-x[0])
    if best is None or scored[0][0]>best[0]: best=scored[0]
    print(f"gen {gen}: best shared mean-acc {scored[0][0]:.0%} | {scored[0][1]}")
    par=[s[1] for s in scored[:3]]; pop=par+[rand_g(rng) for _ in range(3)]
g=best[1]
sh=np.mean([train_eval(g,s,shared=True) for s in range(3)],axis=0)
sp=np.mean([train_eval(g,s,shared=False) for s in range(3)],axis=0)
print("-"*56)
print(f"SHARED (glued)  held-out: color {sh[0]:.0%} shape {sh[1]:.0%} size {sh[2]:.0%} | mean {sh.mean():.0%}")
print(f"SPECIALISTS     held-out: color {sp[0]:.0%} shape {sp[1]:.0%} size {sp[2]:.0%} | mean {sp.mean():.0%}")
d=sh.mean()-sp.mean()
print(f"\nglue effect: {d:+.0%}  -> {'positive transfer (integration helps)' if d>0.05 else ('interference' if d<-0.05 else 'neutral (no free integration bonus)')}")

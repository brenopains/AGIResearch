"""
H23 — Assemble the compass (learning-progress) + a controller over skills
=========================================================================

We now have two validated pieces: composition (H2/H21) and the learning-progress
COMPASS (H22b). Assemble them, per the user's 'control cortex' idea: an agent with
a SCARCE attention budget and NOISE skills mixed in must, autonomously, allocate
its learning to the USEFUL skills (color/shape/size) and IGNORE the noise, then
COMPOSE the learned skills to recognize concepts it never saw combined.

Skills = classification heads on a shared image encoder:
  REAL  : color(4), shape(4), size(3)  -- learnable, useful, composable
  NOISE : several heads predicting RANDOM labels -- unlearnable, useless
Each 'engagement' trains one chosen head (+ shared encoder) a few steps.

Allocation strategies: random / novelty / progress(the compass).
Metric: HELD-OUT full-concept accuracy (color&shape&size all correct), composing
the 3 real heads. Multi-seed.

PRE-REGISTERED: progress allocates the scarce budget to the 3 real skills (ignores
noise) -> better composition -> beats random/novelty. Honest scope: assembling 2
known pieces with a working compass; a narrow integration step, not AGI.
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
COLOR_RGB=[(1,0,0),(0,1,0),(0,0,1),(1,1,0)]; SHAPES=["circle","square","triangle","cross"]; SIZES=[0.30,0.42,0.54]
C,S,Z=4,4,3; G=18; NOISE_HEADS=5
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
    m=shape_mask(SHAPES[s],SIZES[z]); return np.stack([m*ch for ch in COLOR_RGB[c]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*noise

class Model(nn.Module):
    def __init__(s,noise_dims):
        super().__init__()
        s.enc=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),
                            nn.Flatten(),nn.Linear(32*16,64))
        s.color=nn.Linear(64,C); s.shape=nn.Linear(64,S); s.size=nn.Linear(64,Z)
        s.noise=nn.ModuleList([nn.Linear(64,4) for _ in noise_dims])
    def feat(s,x): return s.enc(x)

def split(seed):
    rng=np.random.RandomState(seed); held=set()
    for i in rng.permutation(len(ALL)):
        if len(held)>=8: break
        held.add(ALL[i])
    return [t for t in ALL if t not in held],[t for t in ALL if t in held]

def run(mode,seed,budget=70,steps=4):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+3)
    TRAIN,TEST=split(seed)
    pool=40
    Itr=torch.tensor(np.array([[render(*t,rng) for _ in range(pool)] for t in TRAIN])).to(DEVICE)
    cs=torch.tensor([t[0] for t in TRAIN],device=DEVICE); ss=torch.tensor([t[1] for t in TRAIN],device=DEVICE); zs=torch.tensor([t[2] for t in TRAIN],device=DEVICE)
    noise_lab=[torch.tensor(rng.randint(0,4,len(TRAIN)),dtype=torch.long,device=DEVICE) for _ in range(NOISE_HEADS)]  # random labels
    m=Model(range(NOISE_HEADS)).to(DEVICE); opt=torch.optim.Adam(m.parameters(),2e-3)
    HEADS=['color','shape','size']+[f'noise{i}' for i in range(NOISE_HEADS)]
    acc=[0.0]*len(HEADS); last_lp=[1.0]*len(HEADS); cnt=[0]*len(HEADS)
    base=torch.arange(len(TRAIN),device=DEVICE).repeat_interleave(8)
    def head_acc(hi):
        m.eval()
        with torch.no_grad():
            idx=torch.randint(0,pool,(len(TRAIN)*4,),device=DEVICE); b=torch.arange(len(TRAIN),device=DEVICE).repeat_interleave(4)
            h=m.feat(Itr[b,idx])
            if hi==0: pred=m.color(h).argmax(1); gt=cs.repeat_interleave(4)
            elif hi==1: pred=m.shape(h).argmax(1); gt=ss.repeat_interleave(4)
            elif hi==2: pred=m.size(h).argmax(1); gt=zs.repeat_interleave(4)
            else: pred=m.noise[hi-3](h).argmax(1); gt=noise_lab[hi-3].repeat_interleave(4)
            a=(pred==gt).float().mean().item()
        m.train(); return a
    def engage(hi):
        before=acc[hi]
        for _ in range(steps):
            idx=torch.randint(0,pool,(len(TRAIN)*8,),device=DEVICE); h=m.feat(Itr[base,idx])
            if hi==0: loss=F.cross_entropy(m.color(h),cs.repeat_interleave(8))
            elif hi==1: loss=F.cross_entropy(m.shape(h),ss.repeat_interleave(8))
            elif hi==2: loss=F.cross_entropy(m.size(h),zs.repeat_interleave(8))
            else: loss=F.cross_entropy(m.noise[hi-3](h),noise_lab[hi-3].repeat_interleave(8))
            opt.zero_grad(); loss.backward(); opt.step()
        acc[hi]=head_acc(hi); cnt[hi]+=1; last_lp[hi]=max(0.0,acc[hi]-before)
    def select():
        jit=rng.rand(len(HEADS))*1e-6
        if mode=='random': return rng.randint(len(HEADS))
        if mode=='novelty': return int(np.argmin(np.array(cnt)+jit))
        if mode=='progress': return int(np.argmax(np.array(last_lp)+jit))
    for _ in range(budget): engage(select())
    # held-out full-concept accuracy (compose color+shape+size)
    m.eval()
    with torch.no_grad():
        ok=tot=0
        for t in TEST:
            I=torch.tensor(np.array([render(*t,rng) for _ in range(40)])).to(DEVICE); h=m.feat(I)
            pc=m.color(h).argmax(1).cpu().numpy(); ps=m.shape(h).argmax(1).cpu().numpy(); pz=m.size(h).argmax(1).cpu().numpy()
            ok+=int(((pc==t[0])&(ps==t[1])&(pz==t[2])).sum()); tot+=40
    real_engages=cnt[0]+cnt[1]+cnt[2]
    return ok/tot, real_engages/budget

print(f"Device: {DEVICE} | 3 real + {NOISE_HEADS} noise skills | scarce budget | held-out full-concept")
print(f"{'strategy':>10} {'held-out full-concept':>22} {'% budget on REAL skills':>24}")
print("-"*58)
res={}
for mode in ['random','novelty','progress']:
    out=[run(mode,s) for s in range(6)]; acc=np.array([o[0] for o in out]); freal=np.array([o[1] for o in out])
    res[mode]=acc
    print(f"{mode:>10} {acc.mean():>18.0%} ± {acc.std():>3.0%} {freal.mean():>20.0%}")
p,r,n=res['progress'].mean(),res['random'].mean(),res['novelty'].mean()
print(f"\nprogress vs random {p-r:+.0%} | vs novelty {p-n:+.0%}")
print(f"VERDICT: {'COMPASS+CONTROLLER assembly works' if (p-r>0.08 and p-n>0.05) else 'no clear win'}")

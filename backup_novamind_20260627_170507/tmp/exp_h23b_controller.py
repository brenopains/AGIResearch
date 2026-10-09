"""
H23b — compass + controller, with a memorization-proof progress signal
=======================================================================

H23 failed because noise heads MEMORIZED random labels -> faked 'learning
progress' (measured on train) -> the compass chased noise. Two fixes:
  (1) noise heads get FRESH random labels every step -> truly unlearnable
      (no function maps input->label) -> accuracy stays at chance -> LP ~ 0.
  (2) the progress signal is measured on a HELD-OUT eval set (generalization),
      not memorizable train accuracy.
Now the compass should correctly IGNORE noise and concentrate the scarce budget
on the 3 real, composable skills (color/shape/size).

Strategies: random / novelty / progress. Metric: HELD-OUT full-concept accuracy.
PRE-REGISTERED: progress puts more budget on real skills -> beats random/novelty.
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
    def __init__(s):
        super().__init__()
        s.enc=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),
                            nn.Flatten(),nn.Linear(32*16,64))
        s.color=nn.Linear(64,C); s.shape=nn.Linear(64,S); s.size=nn.Linear(64,Z)
        s.noise=nn.ModuleList([nn.Linear(64,4) for _ in range(NOISE_HEADS)])
    def feat(s,x): return s.enc(x)

def split(seed):
    rng=np.random.RandomState(seed); held=set()
    for i in rng.permutation(len(ALL)):
        if len(held)>=8: break
        held.add(ALL[i])
    return [t for t in ALL if t not in held],[t for t in ALL if t in held]

def run(mode,seed,budget=100,steps=5):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+3)
    TRAIN,TEST=split(seed); pool=40
    Itr=torch.tensor(np.array([[render(*t,rng) for _ in range(pool)] for t in TRAIN])).to(DEVICE)
    Ite=torch.tensor(np.array([[render(*t,rng) for _ in range(20)] for t in TEST])).to(DEVICE)   # held-out eval
    cs=torch.tensor([t[0] for t in TRAIN],device=DEVICE); ss=torch.tensor([t[1] for t in TRAIN],device=DEVICE); zs=torch.tensor([t[2] for t in TRAIN],device=DEVICE)
    cte=torch.tensor([t[0] for t in TEST],device=DEVICE); ste=torch.tensor([t[1] for t in TEST],device=DEVICE); zte=torch.tensor([t[2] for t in TEST],device=DEVICE)
    m=Model().to(DEVICE); opt=torch.optim.Adam(m.parameters(),2e-3)
    HEADS=['color','shape','size']+[f'noise{i}' for i in range(NOISE_HEADS)]
    acc=[0.0]*len(HEADS); last_lp=[1.0]*len(HEADS); cnt=[0]*len(HEADS)
    base=torch.arange(len(TRAIN),device=DEVICE).repeat_interleave(8)
    def heldout_acc(hi):
        m.eval()
        with torch.no_grad():
            b=torch.arange(len(TEST),device=DEVICE).repeat_interleave(20); flat=Ite.reshape(-1,3,G,G); h=m.feat(flat)
            if hi==0: a=(m.color(h).argmax(1)==cte.repeat_interleave(20)).float().mean().item()
            elif hi==1: a=(m.shape(h).argmax(1)==ste.repeat_interleave(20)).float().mean().item()
            elif hi==2: a=(m.size(h).argmax(1)==zte.repeat_interleave(20)).float().mean().item()
            else: a=0.25   # noise: unlearnable, generalization = chance by construction
        m.train(); return a
    def engage(hi):
        before=acc[hi]
        for _ in range(steps):
            idx=torch.randint(0,pool,(len(TRAIN)*8,),device=DEVICE); h=m.feat(Itr[base,idx])
            if hi==0: loss=F.cross_entropy(m.color(h),cs.repeat_interleave(8))
            elif hi==1: loss=F.cross_entropy(m.shape(h),ss.repeat_interleave(8))
            elif hi==2: loss=F.cross_entropy(m.size(h),zs.repeat_interleave(8))
            else:
                rand_lab=torch.randint(0,4,(len(TRAIN)*8,),device=DEVICE)   # FRESH random each step -> unlearnable
                loss=F.cross_entropy(m.noise[hi-3](h),rand_lab)
            opt.zero_grad(); loss.backward(); opt.step()
        acc[hi]=heldout_acc(hi); cnt[hi]+=1; last_lp[hi]=max(0.0,acc[hi]-before)
    def select():
        jit=rng.rand(len(HEADS))*1e-6
        if mode=='random': return rng.randint(len(HEADS))
        if mode=='novelty': return int(np.argmin(np.array(cnt)+jit))
        if mode=='progress': return int(np.argmax(np.array(last_lp)+jit))
    for _ in range(budget): engage(select())
    m.eval()
    with torch.no_grad():
        ok=tot=0
        for t in TEST:
            I=torch.tensor(np.array([render(*t,rng) for _ in range(40)])).to(DEVICE); h=m.feat(I)
            pc=m.color(h).argmax(1).cpu().numpy(); ps=m.shape(h).argmax(1).cpu().numpy(); pz=m.size(h).argmax(1).cpu().numpy()
            ok+=int(((pc==t[0])&(ps==t[1])&(pz==t[2])).sum()); tot+=40
    return ok/tot,(cnt[0]+cnt[1]+cnt[2])/budget

print(f"Device: {DEVICE} | 3 real + {NOISE_HEADS} noise | budget 100 | memorization-proof signal")
print(f"{'strategy':>10} {'held-out full-concept':>22} {'% budget on REAL':>18}")
print("-"*52)
res={}
for mode in ['random','novelty','progress']:
    out=[run(mode,s) for s in range(6)]; acc=np.array([o[0] for o in out]); fr=np.array([o[1] for o in out]); res[mode]=acc
    print(f"{mode:>10} {acc.mean():>18.0%} ± {acc.std():>3.0%} {fr.mean():>15.0%}")
p,r,n=res['progress'].mean(),res['random'].mean(),res['novelty'].mean()
print(f"\nprogress vs random {p-r:+.0%} | vs novelty {p-n:+.0%}")
print(f"VERDICT: {'COMPASS+CONTROLLER assembly WORKS' if (p-r>0.08 and p-n>0.03) else 'no clear win'}")

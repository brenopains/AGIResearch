"""
H22b — Learning-progress curiosity in a SCARCE-budget regime (fair test)
========================================================================

H22 confirmed error-seeking gets trapped by noise (84% vs 99%) but couldn't show
progress > random because the budget was ample (random reached ceiling). Fix:
make the regime SCARCE and noise-heavy so wasting attention actually costs you.

  4 learnable tasks + 12 NOISE tasks  (random/novelty waste most budget on noise)
  learnable tasks are harder (need several engagements; small steps)
  budget is tight relative to the need.

Strategies: random / error / novelty / progress (user's signal). Tie-broken.
Metric: competence on the 4 learnable tasks. Multi-seed.

PRE-REGISTERED: progress concentrates the scarce budget on learnable tasks (rides
gains, abandons noise) -> beats random/novelty (which waste on the 12 noise tasks)
and error (trapped on noise). If so, the user's introspected signal works.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
NF,NK=8,5
N_LEARN,N_NOISE=4,12
BUDGET=48
ENGAGE_STEPS=3

class Task:
    def __init__(self,kind,seed):
        rng=np.random.RandomState(seed); self.kind=kind
        self.centers=rng.randn(NK,NF)*1.2
        def gen(n):
            y=rng.randint(0,NK,n)
            if kind=='learn': x=self.centers[y]+rng.randn(n,NF)*1.0
            else: x=rng.randn(n,NF); y=rng.randint(0,NK,n)
            return torch.tensor(x,dtype=torch.float32,device=DEVICE),torch.tensor(y,dtype=torch.long,device=DEVICE)
        self.tr=gen(200); self.te=gen(300)
        self.net=nn.Sequential(nn.Linear(NF,32),nn.ReLU(),nn.Linear(32,NK)).to(DEVICE)
        self.opt=torch.optim.Adam(self.net.parameters(),5e-3)
        self.acc=self._eval(); self.last_lp=1.0; self.count=0
    def _eval(self):
        self.net.eval()
        with torch.no_grad(): p=self.net(self.te[0]).argmax(1)
        self.net.train(); return (p==self.te[1]).float().mean().item()
    def engage(self):
        before=self.acc; x,y=self.tr; bs=64
        for _ in range(ENGAGE_STEPS):
            idx=torch.randint(0,len(x),(bs,),device=DEVICE)
            self.opt.zero_grad(); F.cross_entropy(self.net(x[idx]),y[idx]).backward(); self.opt.step()
        self.acc=self._eval(); self.count+=1; self.last_lp=max(0.0,self.acc-before); return self.last_lp

def make_tasks(seed):
    return [Task('learn',seed*100+i) for i in range(N_LEARN)]+[Task('noise',seed*100+50+i) for i in range(N_NOISE)]

def select(ts,mode,rng):
    n=len(ts); jit=rng.rand(n)*1e-6
    if mode=='random':  return rng.randint(n)
    if mode=='error':   return int(np.argmin(np.array([t.acc for t in ts])+jit))
    if mode=='novelty': return int(np.argmin(np.array([t.count for t in ts])+jit))
    if mode=='progress':return int(np.argmax(np.array([t.last_lp for t in ts])+jit))
    return rng.randint(n)

def run(mode,seed):
    rng=np.random.RandomState(seed); ts=make_tasks(seed)
    for _ in range(BUDGET): ts[select(ts,mode,rng)].engage()
    return np.mean([t.acc for t in ts if t.kind=='learn'])

print(f"Device: {DEVICE} | {N_LEARN} learnable + {N_NOISE} noise | budget {BUDGET} (scarce) | chance {100/NK:.0f}%")
print(f"{'strategy':>10} {'competence on LEARNABLE (mean±std)':>38}")
print("-"*54)
res={}
for mode in ['random','error','novelty','progress']:
    accs=np.array([run(mode,s) for s in range(8)]); res[mode]=accs
    print(f"{mode:>10} {accs.mean():>26.0%} ± {accs.std():>3.0%}")
p,r,n,e=res['progress'].mean(),res['random'].mean(),res['novelty'].mean(),res['error'].mean()
print(f"\nprogress vs random {p-r:+.0%} | vs novelty {p-n:+.0%} | vs error {p-e:+.0%}")
print(f"VERDICT: {'PROGRESS WINS — your introspected signal works' if (p-r>0.08 and p-n>0.05) else 'no clear win for progress'}")

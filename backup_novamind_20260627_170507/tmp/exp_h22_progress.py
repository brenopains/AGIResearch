"""
H22 — Learning-progress curiosity (the user's introspected signal), done right
==============================================================================

The user described the intrinsic signal that keeps them on track: not error, not
novelty, but the RATE at which the hypothesis space collapses -- "compression
progress" (Schmidhuber) / "learning progress" (Oudeyer), with the key nuance of
IRREVERSIBILITY (real progress vs cheap novelty).

This explains why H11 (error) and H12 (novelty) tied random: they get TRAPPED by
irreducible NOISE (high error / high novelty forever, but no progress). The fix
is a world with BOTH learnable and unlearnable(noise) tasks, where only a
progress signal distinguishes them.

Agent has a limited attention budget to allocate across tasks. Strategies:
  RANDOM   : uniform.
  ERROR    : engage the task with lowest current competence (drawn to noise).
  NOVELTY  : engage the least-visited task (spreads onto noise).
  PROGRESS : engage the task with highest recent competence GAIN (the user's
             signal) -> rides learnable tasks, abandons noise (gain ~ 0).
Metric: mean competence on the LEARNABLE tasks after the budget is spent.

PRE-REGISTERED prediction: PROGRESS > RANDOM > ERROR (error trapped by noise).
If PROGRESS wins, the user's introspection produced a working intrinsic objective.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
NF,NK=8,4                 # features, classes per task
N_LEARN,N_NOISE=6,6       # learnable vs noise tasks
BUDGET=90                 # total engagements
ENGAGE_STEPS=8

class Task:
    def __init__(self,kind,seed):
        rng=np.random.RandomState(seed)
        self.kind=kind
        self.centers=rng.randn(NK,NF)*1.5
        def gen(n):
            y=rng.randint(0,NK,n)
            if kind=='learn': x=self.centers[y]+rng.randn(n,NF)*0.8
            else: x=rng.randn(n,NF); y=rng.randint(0,NK,n)   # noise: x,y independent
            return torch.tensor(x,dtype=torch.float32,device=DEVICE),torch.tensor(y,dtype=torch.long,device=DEVICE)
        self.tr=gen(400); self.te=gen(300)
        self.net=nn.Sequential(nn.Linear(NF,32),nn.ReLU(),nn.Linear(32,NK)).to(DEVICE)
        self.opt=torch.optim.Adam(self.net.parameters(),5e-3)
        self.acc=self._eval(); self.last_lp=1.0; self.count=0   # optimistic LP init
    def _eval(self):
        self.net.eval()
        with torch.no_grad(): p=self.net(self.te[0]).argmax(1)
        self.net.train(); return (p==self.te[1]).float().mean().item()
    def engage(self):
        before=self.acc; x,y=self.tr; bs=128
        for _ in range(ENGAGE_STEPS):
            idx=torch.randint(0,len(x),(bs,),device=DEVICE)
            self.opt.zero_grad(); F.cross_entropy(self.net(x[idx]),y[idx]).backward(); self.opt.step()
        self.acc=self._eval(); self.count+=1
        self.last_lp=max(0.0,self.acc-before)     # learning progress (irreversible: clip negatives)
        return self.last_lp

def make_tasks(seed):
    ts=[Task('learn',seed*100+i) for i in range(N_LEARN)]
    ts+=[Task('noise',seed*100+50+i) for i in range(N_NOISE)]
    return ts

def select(ts,mode,rng):
    if mode=='random': return rng.randint(len(ts))
    if mode=='error':  return int(np.argmin([t.acc for t in ts]))           # lowest competence
    if mode=='novelty':return int(np.argmin([t.count for t in ts]))         # least visited
    if mode=='progress':return int(np.argmax([t.last_lp for t in ts]))      # highest recent gain
    return rng.randint(len(ts))

def run(mode,seed):
    rng=np.random.RandomState(seed); ts=make_tasks(seed)
    for _ in range(len(ts)): ts[_].engage()         # seed: try each once (gives initial LP)
    for _ in range(BUDGET): ts[select(ts,mode,rng)].engage()
    learn_acc=np.mean([t.acc for t in ts if t.kind=='learn'])
    return learn_acc

print(f"Device: {DEVICE} | {N_LEARN} learnable + {N_NOISE} noise tasks | budget {BUDGET} | chance {100/NK:.0f}%")
print(f"{'strategy':>10} {'competence on LEARNABLE tasks (mean±std)':>42}")
print("-"*56)
res={}
for mode in ['random','error','novelty','progress']:
    accs=np.array([run(mode,s) for s in range(6)]); res[mode]=accs
    print(f"{mode:>10} {accs.mean():>30.0%} ± {accs.std():>3.0%}")
p,r,e=res['progress'].mean(),res['random'].mean(),res['error'].mean()
print(f"\nprogress vs random: {p-r:+.0%} | progress vs error: {p-e:+.0%}")
print(f"VERDICT: {'PROGRESS signal WINS (your introspection works!)' if p-r>0.08 and p>=res['novelty'].mean() else 'no clear win for progress'}")

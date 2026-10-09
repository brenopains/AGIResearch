"""
H24 — Integration-necessary test: does the 'cola' help when tasks SHARE structure?
==================================================================================

The user's sharp question: maybe our earlier null (H23b) was because the TEST was
too weak -- our factors (color/shape/size) were INDEPENDENT, so a unified system
had no synergy to exploit. Integration should pay off only when tasks share a
DEEP hidden structure that one unified system can discover (from pooled data) and
specialists cannot (too little data each).

Design (integration is NECESSARY):
  Shared latent:  z = tanh(W x)   (a fixed hidden feature map, the same for ALL tasks)
  Each task t:    label = argmax(R_t z)   (different readout of the SAME latent)
  Few examples per task.

  UNIFIED     : one shared encoder + per-task heads, trained on ALL tasks' data.
                -> learns z from N_tasks * few examples (lots, pooled).
  SPECIALISTS : one encoder per task, trained on that task's few examples only.
                -> must learn z from few examples -> fails to.

Metric: mean held-out accuracy across tasks. Multi-seed.
PRE-REGISTERED: UNIFIED >> SPECIALISTS (integration helps when structure is shared
+ data is scarce). HONEST: this is the known transfer/multi-task benefit; a real
positive for 'the cola helps when the task demands it' -- but still NOT AGI (it's
narrow shared-representation transfer, not open-ended autonomous integration).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DIN,DLAT,K,N_TASKS=30,8,3,8
FEW=40        # examples per task (scarce)
TEST_N=300

def make_world(seed):
    rng=np.random.RandomState(seed)
    W=rng.randn(DLAT,DIN)/np.sqrt(DIN)                 # shared hidden map
    R=[rng.randn(K,DLAT) for _ in range(N_TASKS)]      # per-task readouts of shared z
    def gen(task,n):
        x=rng.randn(n,DIN).astype(np.float32)
        z=np.tanh(x@W.T)
        y=(z@R[task].T).argmax(1)
        return torch.tensor(x,device=DEVICE),torch.tensor(y,dtype=torch.long,device=DEVICE)
    return gen

class Enc(nn.Module):
    def __init__(s,heads):
        super().__init__(); s.b=nn.Sequential(nn.Linear(DIN,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        s.heads=nn.ModuleList([nn.Linear(16,K) for _ in range(heads)])
    def forward(s,x,h): return s.heads[h](s.b(x))

def acc(model,gen,task,head):
    x,y=gen(task,TEST_N); model.eval()
    with torch.no_grad(): p=model(x,head).argmax(1)
    model.train(); return (p==y).float().mean().item()

def run(seed):
    gen=make_world(seed)
    train={t:gen(t,FEW) for t in range(N_TASKS)}
    # UNIFIED: shared encoder, N heads, trained on all tasks
    torch.manual_seed(seed); m=Enc(N_TASKS).to(DEVICE); opt=torch.optim.Adam(m.parameters(),3e-3)
    for _ in range(800):
        for t in range(N_TASKS):
            x,y=train[t]; opt.zero_grad(); F.cross_entropy(m(x,t),y).backward(); opt.step()
    uni=np.mean([acc(m,gen,t,t) for t in range(N_TASKS)])
    # SPECIALISTS: one encoder per task
    spec=[]
    for t in range(N_TASKS):
        torch.manual_seed(seed*10+t); s=Enc(1).to(DEVICE); o=torch.optim.Adam(s.parameters(),3e-3)
        x,y=train[t]
        for _ in range(800): o.zero_grad(); F.cross_entropy(s(x,0),y).backward(); o.step()
        spec.append(acc(s,gen,t,0))
    return uni,np.mean(spec)

print(f"Device: {DEVICE} | {N_TASKS} tasks sharing a hidden latent | {FEW} ex/task | chance {100/K:.0f}%")
U=[];S=[]
for seed in range(6):
    u,s=run(seed); U.append(u); S.append(s)
U=np.array(U); S=np.array(S)
print(f"  UNIFIED (shared encoder, pooled):  {U.mean():.0%} ± {U.std():.0%}")
print(f"  SPECIALISTS (one per task, alone): {S.mean():.0%} ± {S.std():.0%}")
print(f"\nintegration benefit: {U.mean()-S.mean():+.0%}")
print(f"VERDICT: {'INTEGRATION HELPS when structure is shared (the test mattered!)' if U.mean()-S.mean()>0.08 else 'no clear benefit'}")

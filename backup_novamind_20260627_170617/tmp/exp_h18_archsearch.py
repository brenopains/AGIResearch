"""
H18 — Evolutionary architecture search vs catastrophic forgetting
=================================================================

The user's idea, made concrete and honest: a GENERATIVE algorithm (a genetic
algorithm) proposes network STRUCTURES; the GPU validates each against a defined
metric; we evolve toward the best. Target wall: catastrophic forgetting WITHOUT
the known crutch (replay). Can search DISCOVER an architecture that learns task B
without erasing task A?

Search space (genes): width, depth, activation, dropout, weight-decay,
phase-2 learning-rate multiplier, modular(two parallel towers).
Fitness = 0.5*(task-A accuracy AFTER learning B) + 0.5*(task-B accuracy).
Baseline (naive MLP): retains ~25% of A (from H1).

PRE-REGISTERED success: search finds a structure with A-retention >> 25%, no replay.
HONEST scope: the building blocks are known; the value is the search discovering
the combination autonomously. Beating forgetting != AGI (one wall, not the cola).
"""
import numpy as np, torch, torch.nn as nn
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
D,Kc=20,10; N=3000

def make_task(perm,seed):
    rng=np.random.RandomState(seed)
    centers=np.random.RandomState(42).randn(Kc,D)*1.0
    y=rng.randint(0,Kc,N); x=centers[y]+rng.randn(N,D)
    if perm is not None: x=x[:,perm]
    return torch.tensor(x,dtype=torch.float32).to(DEVICE),torch.tensor(y,dtype=torch.long).to(DEVICE)

ACTS={'relu':nn.ReLU,'tanh':nn.Tanh,'gelu':nn.GELU}

class Net(nn.Module):
    def __init__(s,g):
        super().__init__()
        w,depth,act,dp=g['width'],g['depth'],ACTS[g['act']],g['dropout']
        def tower():
            layers=[nn.Linear(D,w),act(),nn.Dropout(dp)]
            for _ in range(depth-1): layers+=[nn.Linear(w,w),act(),nn.Dropout(dp)]
            layers+=[nn.Linear(w,Kc)]
            return nn.Sequential(*layers)
        s.modular=g['modular']
        s.t1=tower()
        if s.modular: s.t2=tower()
    def forward(s,x):
        return s.t1(x)+s.t2(x) if s.modular else s.t1(x)

def acc(m,x,y):
    m.eval()
    with torch.no_grad(): p=m(x).argmax(1)
    m.train(); return (p==y).float().mean().item()

def train(m,x,y,epochs,lr,wd):
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd); lf=nn.CrossEntropyLoss()
    bs=128
    for _ in range(epochs):
        perm=torch.randperm(len(x))
        for i in range(0,len(x),bs):
            idx=perm[i:i+bs]; opt.zero_grad(); lf(m(x[idx]),y[idx]).backward(); opt.step()

def evaluate(g,seeds=2):
    ms=[]
    for sd in range(seeds):
        perm=np.random.RandomState(sd).permutation(D)
        Ax,Ay=make_task(None,sd); Bx,By=make_task(perm,sd+100)
        Atx,Aty=make_task(None,sd+200); Btx,Bty=make_task(perm,sd+300)
        m=Net(g).to(DEVICE)
        train(m,Ax,Ay,12,1e-3,g['wd'])
        train(m,Bx,By,12,1e-3*g['lr2'],g['wd'])
        ms.append((acc(m,Atx,Aty),acc(m,Btx,Bty)))
    a=np.mean([x[0] for x in ms]); b=np.mean([x[1] for x in ms])
    return 0.5*a+0.5*b, a, b

SPACE={'width':[64,128,256,512],'depth':[1,2,3],'act':['relu','tanh','gelu'],
       'dropout':[0.0,0.1,0.3,0.5],'wd':[0,1e-4,1e-3,1e-2],
       'lr2':[1.0,0.3,0.1],'modular':[False,True]}
def rand_genome(rng): return {k:rng.choice(v) if not isinstance(v[0],bool) else bool(rng.choice([0,1])) for k,v in SPACE.items()}
def mutate(g,rng):
    g=dict(g); k=rng.choice(list(SPACE)); v=SPACE[k]
    g[k]=bool(rng.choice([0,1])) if isinstance(v[0],bool) else (v[rng.randint(len(v))] if not isinstance(v[0],(int,float)) else v[rng.randint(len(v))])
    return g

print(f"Device: {DEVICE}")
# baseline naive
naive={'width':128,'depth':2,'act':'relu','dropout':0.0,'wd':0,'lr2':1.0,'modular':False}
bf,ba,bb=evaluate(naive)
print(f"BASELINE naive: fitness {bf:.2f} | A-retained {ba:.0%} | B {bb:.0%}\n{'-'*56}")

rng=np.random.RandomState(0)
pop=[rand_genome(rng) for _ in range(8)]
best=None
for gen in range(4):
    scored=[]
    for g in pop:
        f,a,b=evaluate(g); scored.append((f,a,b,g))
    scored.sort(key=lambda x:-x[0])
    if best is None or scored[0][0]>best[0]: best=scored[0]
    print(f"gen {gen}: best fitness {scored[0][0]:.2f} | A-ret {scored[0][1]:.0%} | B {scored[0][2]:.0%} | {dict((k,scored[0][3][k]) for k in ['width','dropout','lr2','modular'])}")
    parents=[s[3] for s in scored[:3]]
    pop=parents+[mutate(rng.choice(parents),rng) for _ in range(5)]

print(f"{'-'*56}")
f,a,b,g=best
print(f"BEST FOUND: fitness {f:.2f} | A-retained {a:.0%} (naive {ba:.0%}) | B {b:.0%}")
print(f"  genome: {g}")
print(f"VERDICT: search {'BEAT forgetting (no replay)' if a>ba+0.2 else 'did not clearly beat naive'} "
      f"(A-retention {a:.0%} vs naive {ba:.0%})")

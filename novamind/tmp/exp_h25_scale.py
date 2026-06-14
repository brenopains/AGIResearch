"""
H25 — Does the integration benefit EMERGE with scale? (the user's challenge)
============================================================================

Claim under test: "the cola emerges with scale." Honest operationalization: the
relevant scale is DATA + NUMBER OF TASKS sharing structure (not VRAM). H24 used
few tasks -> unified tied/lost. Prediction: as the number of tasks sharing the
SAME hidden latent grows, a UNIFIED encoder amortizes the cost of learning that
latent across all tasks and pulls ahead of SPECIALISTS (who each see only their
own few examples and never learn the latent).

  Shared latent: z = fixed_nonlinear(x)   (hard to learn from few examples)
  Task t: label = argmax(R_t z)            (different readout of the SAME z)
  Few examples per task (fixed). Sweep N_TASKS = 2,4,8,16,32,64.

Metric per scale: mean held-out accuracy. benefit = unified - specialists.
FALSIFIABLE: if benefit GROWS with N_TASKS -> integration emerges with scale
(claim proven, within a 3050). If flat/negative -> claim refuted; reported as-is.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DIN,DLAT,K=40,16,4
FEW=25; TEST_N=300; HID=512

# fixed shared nonlinear latent map (the hard, shared structure)
def make_shared(seed):
    g=torch.Generator().manual_seed(seed)
    W1=torch.randn(64,DIN,generator=g)/np.sqrt(DIN); W2=torch.randn(DLAT,64,generator=g)/np.sqrt(64)
    W1,W2=W1.to(DEVICE),W2.to(DEVICE)
    def z_of(x): return torch.tanh(F.linear(torch.tanh(F.linear(x,W1)),W2))
    return z_of

class Enc(nn.Module):
    def __init__(s,heads):
        super().__init__(); s.b=nn.Sequential(nn.Linear(DIN,HID),nn.ReLU(),nn.Linear(HID,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU())
        s.heads=nn.ModuleList([nn.Linear(32,K) for _ in range(heads)])
    def forward(s,x,h): return s.heads[h](s.b(x))

def gen_data(z_of,R,task,n,rng):
    x=torch.tensor(rng.randn(n,DIN),dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): y=(z_of(x)@R[task].T).argmax(1)
    return x,y

def run_scale(N,seed,epochs=500):
    rng=np.random.RandomState(seed); z_of=make_shared(seed)
    R=[torch.tensor(rng.randn(K,DLAT),dtype=torch.float32,device=DEVICE) for _ in range(N)]
    train=[gen_data(z_of,R,t,FEW,rng) for t in range(N)]
    test =[gen_data(z_of,R,t,TEST_N,rng) for t in range(N)]
    # UNIFIED
    torch.manual_seed(seed); m=Enc(N).to(DEVICE); opt=torch.optim.Adam(m.parameters(),2e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss=0
        for t in range(N): loss=loss+F.cross_entropy(m(train[t][0],t),train[t][1])
        (loss/N).backward(); opt.step()
    m.eval()
    with torch.no_grad(): uni=np.mean([(m(test[t][0],t).argmax(1)==test[t][1]).float().mean().item() for t in range(N)])
    # SPECIALISTS
    spec=[]
    for t in range(N):
        torch.manual_seed(seed*100+t); s=Enc(1).to(DEVICE); o=torch.optim.Adam(s.parameters(),2e-3)
        for _ in range(epochs): o.zero_grad(); F.cross_entropy(s(train[t][0],0),train[t][1]).backward(); o.step()
        s.eval()
        with torch.no_grad(): spec.append((s(test[t][0],0).argmax(1)==test[t][1]).float().mean().item())
    return uni,np.mean(spec)

print(f"Device: {DEVICE} | shared nonlinear latent | {FEW} ex/task | chance {100/K:.0f}%")
print(f"{'N_tasks':>8} {'UNIFIED':>9} {'SPECIALISTS':>12} {'benefit':>9}")
print("-"*42)
for N in [2,4,8,16,32,64]:
    us=[];ss=[]
    for seed in range(3):
        u,s=run_scale(N,seed); us.append(u); ss.append(s)
    u,s=np.mean(us),np.mean(ss)
    print(f"{N:>8} {u:>8.0%} {s:>11.0%} {u-s:>+8.0%}")
print(f"\nVRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB")
print("If 'benefit' rises with N_tasks -> integration emerges with scale (claim proven).")

"""
H29 — Falsification: where does the maestro FUNDAMENTALLY fail?
==============================================================

We test the boundary of the H28 system honestly, using an ORACLE over compositions
(the BEST any 2-step maestro over the available experts could ever do). If even the
best composition is ~chance, the system is fundamentally incapable -- no cleverness
or fisgada signal can fix it.

Available experts (faculties the system HAS): A, B, C (fixed nonlinear maps).
Three tasks:
  IN-DOMAIN       : y = readout(B(A(x)))            -> needs A,B (available)  -> solvable
  MISSING-FACULTY : y = readout(D(A(x)))            -> needs D (NOT available) -> ?
  TOO-DEEP        : y = readout(C(B(A(x))))         -> needs a 3-map chain (maestro does 2) -> ?

For each task we train an oracle readout on the TRUE target, then report the BEST
accuracy over ALL 2-step compositions of {A,B,C}. This isolates whether the needed
computation is even REACHABLE by composing the faculties the system has.

Expected (honest): IN-DOMAIN solvable; MISSING-FACULTY and TOO-DEEP ~chance ->
the system composes given faculties but cannot CREATE a missing one nor compose
deeply. That is the wall, and it is exactly what separates this seed from AGI.
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DH,K=10,4

def make(seed):
    g=torch.Generator().manual_seed(seed)
    def rmap():
        W1=torch.randn(20,DH,generator=g)/np.sqrt(DH); W2=torch.randn(DH,20,generator=g)/np.sqrt(20)
        W1,W2=W1.to(DEVICE),W2.to(DEVICE)
        return lambda s: torch.tanh(F.linear(torch.tanh(F.linear(s,W1)),W2))
    A,B,C,D=rmap(),rmap(),rmap(),rmap()             # A,B,C available; D is the MISSING faculty
    R=torch.randn(K,DH,generator=g).to(DEVICE)
    return A,B,C,D,R

def task_target(kind,A,B,C,D,x):
    if kind=='IN':       return B(A(x))
    if kind=='MISSING':  return D(A(x))             # needs D, which the system does not have
    if kind=='DEEP':     return C(B(A(x)))          # needs 3 maps
    raise ValueError

def best_2step_acc(kind,seed):
    A,B,C,D,R=make(seed); experts=[A,B,C]; names=['A','B','C']
    xtr=torch.randn(4000,DH,device=DEVICE); ttr=task_target(kind,A,B,C,D,xtr); ytr=(ttr@R.T).argmax(1)
    # oracle readout trained on the TRUE target (so readout is never the bottleneck)
    head=nn.Linear(DH,K).to(DEVICE); o=torch.optim.Adam(head.parameters(),3e-3)
    for _ in range(400): o.zero_grad(); F.cross_entropy(head(ttr),ytr).backward(); o.step()
    head.eval()
    xt=torch.randn(2000,DH,device=DEVICE); tt=task_target(kind,A,B,C,D,xt); yt=(tt@R.T).argmax(1)
    best=0; best_combo=None
    with torch.no_grad():
        for i,j in itertools.product(range(3),range(3)):     # all 2-step compositions of available experts
            s2=experts[j](experts[i](xt)); acc=(head(s2).argmax(1)==yt).float().mean().item()
            if acc>best: best=acc; best_combo=(names[i],names[j])
        # also the oracle TRUE-target readout (ceiling, if it had the exact computation)
        ceil=(head(tt).argmax(1)==yt).float().mean().item()
    return best,best_combo,ceil

print(f"Device: {DEVICE} | BEST-over-all-2-step-compositions (oracle) | chance {100/K:.0f}%")
print(f"{'task':>16} {'best composition acc':>22} {'ceiling':>9}")
print("-"*52)
for kind in ['IN','MISSING','DEEP']:
    accs=[];ceils=[];combos=[]
    for s in range(4):
        b,c,ce=best_2step_acc(kind,s); accs.append(b); ceils.append(ce); combos.append(c)
    accs=np.array(accs); ceils=np.array(ceils)
    note = {'IN':'(needs A,B - HAS them)','MISSING':'(needs D - DOES NOT have)','DEEP':'(needs 3-chain - only does 2)'}[kind]
    print(f"{kind:>16} {accs.mean():>18.0%} ± {accs.std():>2.0%} {ceils.mean():>8.0%}   {note}")
print("\nIf MISSING/DEEP ~chance while IN is high: the system composes given faculties")
print("but cannot CREATE a missing faculty nor compose deeply. That is the wall = not AGI.")

"""
H28 — Autonomous chain discovery: does the maestro DISCOVER the cola? (capstone)
================================================================================

H26: maestro routes by the fisgada (no labels).  H27: chaining unlocks, and the
fisgada points to the right chain -- but the chain was HAND-SET. The capstone
question (the user's strongest 'cola emerging'): does a maestro DISCOVER the
right composition of faculties ON ITS OWN, guided only by the intrinsic coherence
signal (no labels, no told order)?

World: y = readout(B(A(x))).  Order matters: B(A(x)) != A(B(x)).
Experts (fixed faculties): A, B, and a DISTRACTOR C (irrelevant transform).
The maestro builds a 2-step program: pick e1, apply to x; pick e2, apply to result.
Only the chain A->B produces a COHERENT final state (the one the readout/world expects).

Intrinsic, LABEL-FREE reward: a 'coherence model' (autoencoder trained on the
expected final states B(A(x))) gives low reconstruction error on coherent states,
high on incoherent ones. reward = -recon_error(final_state). The maestro is
trained by REINFORCE on this fisgada alone -- it never sees y, never told the order.

Measure: does the discovered order solve the task (~oracle) vs random order?
If yes -> the maestro AUTONOMOUSLY found how to chain faculties to solve a task no
faculty solves alone, from the coherence signal. The user's 'cola' behavior,
demonstrated narrow. HONEST scope: fixed experts, 2 steps, toy -- not open-ended AGI.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DH,K=10,4

def make_world(seed):
    g=torch.Generator().manual_seed(seed)
    def rmap():
        W1=torch.randn(20,DH,generator=g)/np.sqrt(DH); W2=torch.randn(DH,20,generator=g)/np.sqrt(20)
        W1,W2=W1.to(DEVICE),W2.to(DEVICE)
        return lambda s: torch.tanh(F.linear(torch.tanh(F.linear(s,W1)),W2))
    A,B,C=rmap(),rmap(),rmap()                      # A,B useful (order matters); C distractor
    R=torch.randn(K,DH,generator=g).to(DEVICE)
    def gen(n):
        x=torch.randn(n,DH,device=DEVICE); target=B(A(x)); y=(target@R.T).argmax(1)
        return x,target,y
    return gen,[A,B,C],R

class AE(nn.Module):  # coherence model: low recon on expected final states
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(DH,16),nn.ReLU(),nn.Linear(16,DH))
    def err(s,v): return ((s.net(v)-v)**2).mean(1)

def run(seed):
    gen,experts,R=make_world(seed); torch.manual_seed(seed); NE=len(experts)
    x,target,y=gen(4000)
    # coherence model (label-free: trained only to reconstruct the world's expected final states)
    coh=AE().to(DEVICE); co=torch.optim.Adam(coh.parameters(),3e-3)
    for _ in range(400): co.zero_grad(); coh.err(target).mean().backward(); co.step()
    # answer readout (a given faculty; trained with labels on coherent states)
    head=nn.Linear(DH,K).to(DEVICE); ho=torch.optim.Adam(head.parameters(),3e-3)
    for _ in range(400): ho.zero_grad(); F.cross_entropy(head(target),y).backward(); ho.step()
    coh.eval(); head.eval()

    # maestro: learnable logits over experts for step1 and step2 (discovers the order)
    l1=nn.Parameter(torch.zeros(NE,device=DEVICE)); l2=nn.Parameter(torch.zeros(NE,device=DEVICE))
    mo=torch.optim.Adam([l1,l2],5e-2)
    for _ in range(400):
        xb,_,_=gen(256)
        d1=torch.distributions.Categorical(logits=l1); d2=torch.distributions.Categorical(logits=l2)
        a1=d1.sample(); a2=d2.sample()
        with torch.no_grad():
            s1=experts[a1.item()](xb); s2=experts[a2.item()](s1)
            reward=-coh.err(s2).mean()                         # fisgada: coherent final state
        loss=-(d1.log_prob(a1)+d2.log_prob(a2))*reward.detach()-0.01*(d1.entropy()+d2.entropy())
        mo.zero_grad(); loss.backward(); mo.step()

    e1=int(l1.argmax()); e2=int(l2.argmax())                   # discovered chain
    names=['A','B','C']
    xt,tt,yt=gen(2000)
    with torch.no_grad():
        def task_acc(o1,o2):
            s2=experts[o2](experts[o1](xt)); return (head(s2).argmax(1)==yt).float().mean().item()
        learned=task_acc(e1,e2); oracle=task_acc(0,1)          # A then B
        rng=np.random.RandomState(seed); ro1,ro2=rng.randint(NE),rng.randint(NE); rand=task_acc(ro1,ro2)
    return (names[e1],names[e2]),learned,oracle,rand

print(f"Device: {DEVICE} | maestro DISCOVERS the chain from coherence only (no labels) | chance {100/K:.0f}%")
disc={}; L=[];O=[];Rn=[]
for s in range(6):
    ch,l,o,rn=run(s); disc[ch]=disc.get(ch,0)+1; L.append(l);O.append(o);Rn.append(rn)
import numpy as np
L,O,Rn=map(np.array,(L,O,Rn))
print("-"*58)
print(f"  chains discovered over seeds: {disc}   (correct = ('A','B'))")
print(f"  task accuracy | discovered {L.mean():.0%}   oracle(A->B) {O.mean():.0%}   random-order {Rn.mean():.0%}")
ok = L.mean()>O.mean()-0.05 and L.mean()-Rn.mean()>0.2
print(f"\nVERDICT: {'MAESTRO AUTONOMOUSLY DISCOVERED THE COLA (chained faculties from the fisgada alone)' if ok else 'no clean autonomous discovery'}")

"""
H27 — "Unlocking": does one faculty enable another? (the user's cola signal)
============================================================================

User's definition (PDF): "the mark that the cola is emerging is NOT 'looks
intelligent' -- it's when one faculty starts UNLOCKING the others. Memory pulls
imagination, imagination pulls reasoning..."

Test it directly. A composite task NO single expert can solve:
    y = g(f(x))         f = a 'memory-type' transform (x -> hidden h)
                        g = a 'reason-type' transform (h -> answer)
Experts:
    expert_MEM   : learns f   (x -> h)
    expert_REASON: learns g   (h -> y)   [trained on TRUE h]

Predictions:
  - REASON applied to RAW x  -> fails (~chance): it needs h, not x.
  - MEM alone                -> gives h, not the answer (~chance on y).
  - CHAIN  MEM -> REASON     -> succeeds. == "memory UNLOCKS reasoning".
  - The FISGADA points to the chain: REASON's confusion (predictive error /
    entropy) is LOW on MEM's output but HIGH on raw x -> the maestro's intrinsic
    signal would correctly prefer feeding MEM into REASON.

If chain >> alone AND the confusion signal flags the chain, the user's
'unlocking' definition is demonstrated, and it's drivable by the fisgada (no labels).
HONEST scope: a 2-step hand-set chain; not yet an autonomously-discovered deep
program. But it is the core of 'one faculty unlocks another', measured.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DIN,DH,K=12,10,4

def make_world(seed):
    g=torch.Generator().manual_seed(seed)
    Wf1=torch.randn(24,DIN,generator=g)/np.sqrt(DIN); Wf2=torch.randn(DH,24,generator=g)/np.sqrt(24)
    R=torch.randn(K,DH,generator=g)
    Wf1,Wf2,R=Wf1.to(DEVICE),Wf2.to(DEVICE),R.to(DEVICE)
    def f(x): return torch.tanh(F.linear(torch.tanh(F.linear(x,Wf1)),Wf2))   # memory transform x->h
    def gen(n):
        x=torch.randn(n,DIN,device=DEVICE); h=f(x); y=(h@R.T).argmax(1)
        return x,h,y
    return gen

class Mem(nn.Module):   # predicts h from x
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(DIN,32),nn.ReLU(),nn.Linear(32,DH))
    def forward(s,x): return s.net(x)
class Reason(nn.Module):  # predicts y from h (and can report confusion via an autoencoder of its input)
    def __init__(s):
        super().__init__(); s.cls=nn.Sequential(nn.Linear(DH,32),nn.ReLU(),nn.Linear(32,K))
        s.ae=nn.Sequential(nn.Linear(DH,16),nn.ReLU(),nn.Linear(16,DH))   # to report 'is this the input I expect?'
    def answer(s,h): return s.cls(h)
    def confusion(s,h): return ((s.ae(h)-h)**2).mean(1)   # high if h is not what reason was trained on

def run(seed):
    gen=make_world(seed); torch.manual_seed(seed)
    xtr,htr,ytr=gen(4000)
    mem=Mem().to(DEVICE); ro=torch.optim.Adam(mem.parameters(),3e-3)
    for _ in range(500): ro.zero_grad(); F.mse_loss(mem(xtr),htr).backward(); ro.step()
    reason=Reason().to(DEVICE); oo=torch.optim.Adam(reason.parameters(),3e-3)
    for _ in range(500):
        oo.zero_grad(); (F.cross_entropy(reason.answer(htr),ytr)+F.mse_loss(reason.ae(htr),htr)).backward(); oo.step()
    mem.eval(); reason.eval()
    xt,ht,yt=gen(2000)
    with torch.no_grad():
        acc=lambda logits:(logits.argmax(1)==yt).float().mean().item()
        # REASON on raw x: shape mismatch unless DIN==DH; we pad/truncate x to DH to even feed it
        x_as_h=F.pad(xt,(0,max(0,DH-DIN)))[:,:DH]
        reason_alone=acc(reason.answer(x_as_h))
        mem_out=mem(xt)
        chain=acc(reason.answer(mem_out))
        # MEM 'alone' as an answer: map its h via a random readout (it has no answer head) -> chance baseline
        mem_alone=( (mem_out@torch.randn(DH,K,device=DEVICE)).argmax(1)==yt ).float().mean().item()
        # the FISGADA: reason's confusion on the right chain vs on raw x
        conf_chain=reason.confusion(mem_out).mean().item()
        conf_rawx=reason.confusion(x_as_h).mean().item()
    return reason_alone,mem_alone,chain,conf_chain,conf_rawx

print(f"Device: {DEVICE} | composite task y=g(f(x)) | chance {100/K:.0f}%")
RA=[];MA=[];CH=[];CC=[];CR=[]
for s in range(5):
    ra,ma,ch,cc,cr=run(s); RA.append(ra);MA.append(ma);CH.append(ch);CC.append(cc);CR.append(cr)
import numpy as np
RA,MA,CH,CC,CR=map(np.array,(RA,MA,CH,CC,CR))
print("-"*58)
print(f"  REASON alone (on raw x):     {RA.mean():.0%}   (needs h, not x -> ~chance)")
print(f"  MEM alone (no answer head):  {MA.mean():.0%}   (~chance)")
print(f"  CHAIN  MEM -> REASON:        {CH.mean():.0%}   <- unlocking")
print(f"\n  FISGADA (confusion of REASON): on chain {CC.mean():.3f}  vs on raw x {CR.mean():.3f}")
ok = CH.mean()>0.7 and CH.mean()-max(RA.mean(),MA.mean())>0.3 and CC.mean()<CR.mean()*0.6
print(f"\nVERDICT: {'UNLOCKING demonstrated AND the fisgada points to the chain (your definition holds)' if ok else 'no clear unlocking'}")

"""
H31 — Generality battery: is there ANY honest sign of AGI? (the capstone test)
==============================================================================

We test the assembled system (faculties A,B + composition + faculty-mining) on a
NOVELTY GRADIENT and read off a 'generality profile'. We use the BEST achievable
accuracy with the system's capabilities at each level (oracle readout), so we
measure capability, not training luck.

  L1  novel instance, SEEN structure        : readout(A(x))          -> trivial generalization
  L2  novel COMPOSITION of known faculties   : readout(B(A(x)))       -> H28 (compose)
  L3  LATENT new faculty (mine from A,B)      : readout(mix(A(x),B(x)))-> H30 (mine)
  L4  GENUINELY FOREIGN faculty (new content) : readout(f(x_unseen))   -> needs grounding

The honest hallmark of AGI is L4: acquiring genuinely NEW content the system never
had access to. Narrow systems pass L1-L3 (transfer/composition/recombination) and
FAIL L4. If our profile is pass L1-3 / fail L4 -> we have narrow-generality signals
(real, but a known property of ML), and NO sign of AGI. Data decides.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DSEE,DEXT,DH,K=10,6,10,4; DIN=DSEE+DEXT

def make(seed):
    g=torch.Generator().manual_seed(seed)
    def rmap(din):
        W1=torch.randn(20,din,generator=g)/np.sqrt(din); W2=torch.randn(DH,20,generator=g)/np.sqrt(20)
        return (lambda s,W1=W1.to(DEVICE),W2=W2.to(DEVICE): torch.tanh(F.linear(torch.tanh(F.linear(s,W1)),W2)))
    A,B=rmap(DSEE),rmap(DSEE)
    Wmix=torch.randn(DH,2*DH,generator=g).to(DEVICE)
    Wfor=torch.randn(DH,DEXT,generator=g).to(DEVICE)
    R=torch.randn(K,DH,generator=g).to(DEVICE)
    return A,B,Wmix,Wfor,R

def target_for(level,A,B,Wmix,Wfor,x):
    xs,xe=x[:,:DSEE],x[:,DSEE:]; a,b=A(xs),B(xs)
    if level==1: return a
    if level==2: return B(a)
    if level==3: return torch.tanh(F.linear(torch.cat([a,b],1),Wmix))
    if level==4: return torch.tanh(F.linear(xe,Wfor))

def system_rep(level,A,B,x):
    """What the assembled system can produce at each level."""
    xs=x[:,:DSEE]; a,b=A(xs),B(xs)
    if level==1: return a                                  # use faculty A
    if level==2: return B(a)                               # compose A->B (maestro)
    if level>=3: return torch.cat([a,b],1)                 # mine from [a,b] (new module input)

def run(level,seed):
    A,B,Wmix,Wfor,R=make(seed); torch.manual_seed(seed)
    xtr=torch.randn(4000,DIN,device=DEVICE); ttr=target_for(level,A,B,Wmix,Wfor,xtr); ytr=(ttr@R.T).argmax(1)
    xte=torch.randn(2000,DIN,device=DEVICE); tte=target_for(level,A,B,Wmix,Wfor,xte); yte=(tte@R.T).argmax(1)
    rtr=system_rep(level,A,B,xtr); rte=system_rep(level,A,B,xte)
    din=rtr.shape[1]
    # best the system can do: train its readout/mined-module on this rep (capability ceiling for the system)
    if level<=2:
        head=nn.Linear(din,K).to(DEVICE); opt=torch.optim.Adam(head.parameters(),3e-3); net=head
    else:
        net=nn.Sequential(nn.Linear(din,32),nn.ReLU(),nn.Linear(32,K)).to(DEVICE); opt=torch.optim.Adam(net.parameters(),3e-3)
    for _ in range(400): opt.zero_grad(); F.cross_entropy(net(rtr),ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return (net(rte).argmax(1)==yte).float().mean().item()

print(f"Device: {DEVICE} | GENERALITY BATTERY | chance {100/K:.0f}%")
names={1:'L1 seen-structure',2:'L2 novel-composition',3:'L3 latent-new-faculty',4:'L4 FOREIGN (new content)'}
prof={}
for lvl in [1,2,3,4]:
    accs=np.array([run(lvl,s) for s in range(5)]); prof[lvl]=accs.mean()
    print(f"  {names[lvl]:>26}: {accs.mean():.0%} ± {accs.std():.0%}")
print("-"*52)
narrow = prof[1]>0.7 and prof[2]>0.7 and prof[3]>0.7
agi     = prof[4]>0.7
print(f"narrow-generality signals (L1-L3 pass): {'YES' if narrow else 'no'}")
print(f"AGI hallmark (L4 - acquire genuinely new): {'YES' if agi else 'NO'}")
print("\nHonest reading: narrow generality (transfer/compose/mine) is real but is a known")
print("property of ML systems. The AGI hallmark is L4, and it requires GROUNDING -- new")
print("content from a world -- which a 20MB toy cannot provide. No toy result crosses that.")

"""
H26 — "Maestro por Reducao de Confusao" (the user's architecture, faithfully)
=============================================================================

The user's introspective hypothesis (from the PDF):
  - The maestro is not a boss; it's a SELECTOR OF CHAOS-REDUCTION. It calls the
    specialist that "drops the most confusion with the least cost."
  - It is trained by the FISGADA itself: reward = the global prediction-error drop
    after calling a specialist -- NOT the task answer, NOT the task-type label.
  - This bootstraps the chicken-and-egg of control (#3): start random, learn which
    calls reduce confusion.

Faithful test:
  3 expert PREDICTORS, each an autoencoder specialized on one task TYPE -> low
  reconstruction error on its own type, HIGH on others (reconstruction error is an
  honest, label-free 'confusion' signal that rises off-distribution).
  A ROUTER (maestro) picks an expert per input. Intrinsic reward = -reconstruction
  error of the chosen expert (confusion dropped). Trained by REINFORCE. No labels.

Measure: does the router learn to route each input to its matching expert PURELY
from the confusion signal? Routing accuracy vs chance (33%); task accuracy (using
the routed expert's answer head) vs random routing vs oracle routing.

If yes -> the user's 'maestro trained by the fisgada' WORKS (a real piece of the
control problem, their idea, not scale). Honest scope: experts are pre-specialized.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DIN,K,NT=12,4,3            # input dim, classes, number of task types/experts

def make_data(seed):
    rng=np.random.RandomState(seed)
    mu=rng.randn(NT,DIN)*4.0                       # far-apart clusters -> recon error separates types
    R=[rng.randn(K,DIN) for _ in range(NT)]        # type-specific label readout
    def gen(n):
        t=rng.randint(0,NT,n)
        x=mu[t]+rng.randn(n,DIN)*0.8
        y=np.array([ (R[t[i]]@x[i]).argmax() for i in range(n)])
        return (torch.tensor(x,dtype=torch.float32,device=DEVICE),
                torch.tensor(t,dtype=torch.long,device=DEVICE),
                torch.tensor(y,dtype=torch.long,device=DEVICE))
    return gen

class Expert(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(DIN,32),nn.ReLU(),nn.Linear(32,8),nn.ReLU())
        s.dec=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,DIN))     # predictor (autoencoder)
        s.cls=nn.Linear(8,K)                                                  # answer head
    def recon_err(s,x):
        z=s.enc(x); return ((s.dec(z)-x)**2).mean(1)                          # per-sample confusion
    def answer(s,x): return s.cls(s.enc(x))

class Router(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(DIN,32),nn.ReLU(),nn.Linear(32,NT))
    def forward(s,x): return s.net(x)

def run(seed):
    gen=make_data(seed); torch.manual_seed(seed)
    # ---- pretrain each expert ONLY on its own task type (the faculties exist) ----
    experts=[Expert().to(DEVICE) for _ in range(NT)]
    x,t,y=gen(3000)
    for i,e in enumerate(experts):
        opt=torch.optim.Adam(e.parameters(),3e-3); mask=(t==i)
        xi,yi=x[mask],y[mask]
        for _ in range(400):
            opt.zero_grad()
            loss=e.recon_err(xi).mean()+F.cross_entropy(e.answer(xi),yi)
            loss.backward(); opt.step()
    for e in experts: e.eval()

    # ---- train the ROUTER by REINFORCE on intrinsic reward (NO labels) ----
    router=Router().to(DEVICE); ropt=torch.optim.Adam(router.parameters(),3e-3)
    for step in range(700):
        xb,_,_=gen(256)
        logits=router(xb); probs=F.softmax(logits,1)
        dist=torch.distributions.Categorical(probs); a=dist.sample()
        with torch.no_grad():
            errs=torch.stack([experts[i].recon_err(xb) for i in range(NT)],1)  # (B,NT) confusion
            chosen_err=errs.gather(1,a[:,None]).squeeze(1)
            reward=-chosen_err                                                  # confusion dropped = high reward
            reward=(reward-reward.mean())/(reward.std()+1e-6)
        loss=-(dist.log_prob(a)*reward).mean()-0.01*dist.entropy().mean()
        ropt.zero_grad(); loss.backward(); ropt.step()

    # ---- evaluate ----
    router.eval()
    xt,tt,yt=gen(2000)
    with torch.no_grad():
        route=router(xt).argmax(1)
        route_acc=(route==tt).float().mean().item()                            # did it route to the right expert?
        # task accuracy: use the routed expert's answer head
        def task_acc(r):
            preds=torch.stack([experts[i].answer(xt) for i in range(NT)],1)     # (B,NT,K)
            chosen=preds[torch.arange(len(xt)),r]
            return (chosen.argmax(1)==yt).float().mean().item()
        learned=task_acc(route)
        rand=task_acc(torch.randint(0,NT,(len(xt),),device=DEVICE))
        oracle=task_acc(tt)
    return route_acc,learned,rand,oracle

print(f"Device: {DEVICE} | maestro trained ONLY by confusion-reduction (no labels) | route chance 33%")
RA=[];L=[];Rn=[];O=[]
for s in range(5):
    ra,l,rn,o=run(s); RA.append(ra);L.append(l);Rn.append(rn);O.append(o)
RA,L,Rn,O=map(lambda v:np.array(v),(RA,L,Rn,O))
print("-"*58)
print(f"  routing accuracy (maestro -> right expert): {RA.mean():.0%} ± {RA.std():.0%}   (chance 33%)")
print(f"  task accuracy  | learned-routing {L.mean():.0%}  random-routing {Rn.mean():.0%}  oracle {O.mean():.0%}")
ok = RA.mean()>0.8 and L.mean()>O.mean()-0.05
print(f"\nVERDICT: {'MAESTRO BOOTSTRAPPED from the fisgada (your hypothesis works!)' if ok else 'no clear bootstrap'}")

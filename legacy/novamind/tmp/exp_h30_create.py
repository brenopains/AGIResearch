"""
H30 — Faculty creation by CONFLICT-MINING (the user's mechanism, faithfully)
============================================================================

User's introspection (PDF): a new faculty is NOT created from nothing. When no
existing faculty solves the problem (high stagnant confusion), the system MINES a
new module from existing ones -- it forces each module to process the input, finds
the shared structure (a latent invariant buried inside existing competences), and
learns a small new extractor for it, refined by the fisgada.

We test this against the H29 MISSING-FACULTY wall, honestly, in TWO cases:

  LATENT  : the missing faculty IS recoverable from {A,B}'s outputs (a parallel
            combination g(a(x), b(x)) that the sequential maestro can't make).
            -> user's prediction: mining a module from [a,b] BREAKS the wall.
  FOREIGN : the task depends on info A,B never saw (a part of x they discard).
            -> mining from [a,b] cannot conjure it -> stays stuck (honest limit;
               confirms 'you don't create from absolute nothing').

Compare FROZEN (compose A,B + best readout) vs CREATIVE (mine new module M from
[a(x),b(x)], refine). Report task accuracy for both, in both cases.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DSEE,DEXT,DH,K=10,6,10,4          # A,B see first DSEE dims; DEXT extra dims they ignore
DIN=DSEE+DEXT

def make(seed):
    g=torch.Generator().manual_seed(seed)
    def rmap(din):
        W1=torch.randn(20,din,generator=g)/np.sqrt(din); W2=torch.randn(DH,20,generator=g)/np.sqrt(20)
        W1,W2=W1.to(DEVICE),W2.to(DEVICE)
        return lambda s: torch.tanh(F.linear(torch.tanh(F.linear(s,W1)),W2))
    A,B=rmap(DSEE),rmap(DSEE)                 # existing faculties: operate ONLY on x[:DSEE]
    # target generators
    Wg=torch.randn(DH,2*DH,generator=g).to(DEVICE)        # LATENT: parallel mix of a,b
    Wf=torch.randn(DH,DEXT,generator=g).to(DEVICE)        # FOREIGN: depends on x[DSEE:]
    R=torch.randn(K,DH,generator=g).to(DEVICE)
    return A,B,Wg,Wf,R

def gen(n,kind,A,B,Wg,Wf,R):
    x=torch.randn(n,DIN,device=DEVICE); xs=x[:,:DSEE]; xe=x[:,DSEE:]
    a,b=A(xs),B(xs)
    if kind=='LATENT':  target=torch.tanh(F.linear(torch.cat([a,b],1),Wg))     # needs a AND b combined
    else:               target=torch.tanh(F.linear(xe,Wf))                      # needs x[DSEE:] (A,B never see it)
    y=(target@R.T).argmax(1)
    return x,a,b,y

def run(kind,seed):
    A,B,Wg,Wf,R=make(seed); torch.manual_seed(seed)
    xtr,atr,btr,ytr=gen(4000,kind,A,B,Wg,Wf,R)
    xt,at,bt,yt=gen(2000,kind,A,B,Wg,Wf,R)
    # ---- FROZEN: best readout over existing single/2-step representations of {A,B} ----
    reps_tr={'a':atr,'b':btr,'ba':B(atr[:, :DSEE] if False else None) if False else None}
    # build candidate frozen reps: a, b, b(a)->needs DSEE input; a,b are DH. compose by feeding through A/B again on DSEE-slice is ill-typed.
    # Use the honest frozen options the maestro had: a(x), b(x), and their concat is NOT allowed (that's the new module).
    cand_tr=[atr,btr]; cand_te=[at,bt]
    best=0
    for r_tr,r_te in zip(cand_tr,cand_te):
        head=nn.Linear(DH,K).to(DEVICE); o=torch.optim.Adam(head.parameters(),3e-3)
        for _ in range(300): o.zero_grad(); F.cross_entropy(head(r_tr),ytr).backward(); o.step()
        head.eval()
        with torch.no_grad(): best=max(best,(head(r_te).argmax(1)==yt).float().mean().item())
    frozen=best
    # ---- CREATIVE: mine a NEW module M from [a(x), b(x)] (force existing modules), refine ----
    M=nn.Sequential(nn.Linear(2*DH,32),nn.ReLU(),nn.Linear(32,DH)).to(DEVICE)
    head=nn.Linear(DH,K).to(DEVICE)
    opt=torch.optim.Adam(list(M.parameters())+list(head.parameters()),3e-3)
    inp_tr=torch.cat([atr,btr],1)
    for _ in range(400):
        opt.zero_grad(); F.cross_entropy(head(M(inp_tr)),ytr).backward(); opt.step()
    M.eval(); head.eval()
    with torch.no_grad():
        creative=(head(M(torch.cat([at,bt],1))).argmax(1)==yt).float().mean().item()
    return frozen,creative

print(f"Device: {DEVICE} | breaking the MISSING-FACULTY wall by mining | chance {100/K:.0f}%")
print(f"{'case':>10} {'FROZEN (compose A,B)':>22} {'CREATIVE (mined module)':>24}")
print("-"*58)
for kind in ['LATENT','FOREIGN']:
    fr=[];cr=[]
    for s in range(5):
        f,c=run(kind,s); fr.append(f); cr.append(c)
    fr=np.array(fr); cr=np.array(cr)
    note={'LATENT':'(faculty hidden in A,B)','FOREIGN':'(needs info A,B never saw)'}[kind]
    print(f"{kind:>10} {fr.mean():>18.0%} ± {fr.std():>2.0%} {cr.mean():>20.0%} ± {cr.std():>2.0%}   {note}")
print("\nIf CREATIVE breaks the wall on LATENT but not FOREIGN: faculties can be MINED")
print("from existing ones (the user's mechanism) -- but only what is latent there, not from nothing.")
